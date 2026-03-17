# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/detr.py
"""
MaskFormer criterion.
"""

import logging

import torch
import torch.nn.functional as F
from torch import nn

from detectron2.utils.comm import get_world_size
from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)

from ..utils.misc import is_dist_avail_and_initialized, nested_tensor_from_tensor_list


def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(dice_loss)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(sigmoid_ce_loss)  # type: torch.jit.ScriptModule


def calculate_uncertainty(logits):
    """
    We estimate uncertainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


class SetCriterion(nn.Module):
    """
    Loss for Mask2Former-like architectures.

    分成两类：
    - 基础监督：labels / masks（走 self.losses + get_loss）
    - 物理先验：垂直 / 支撑 / 拓扑（在 forward 末尾单独算，再自适应融合）
    """

    def __init__(
        self,
        num_classes,
        matcher,
        weight_dict,
        eos_coef,
        losses,
        num_points,
        oversample_ratio,
        importance_sample_ratio,
        # 先验相关参数
        use_priors: bool = False,
        class_vertical_ranges=None,  # [num_classes, 2]
        ground_classes=None,         # list[int]  可通行表面类（road, sidewalk, terrain）
        heavy_classes=None,          # list[int]  有重量的物体类（person, car, truck...）
        sky_class: int = None,       # sky 类 id
        road_class: int = None,      # road 类 id
        gamma_prior: float = 0.0,    # 外部整体先验权重（真正起作用的是 weight_dict["loss_prior"]）
    ):
        """
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses

        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

        # pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio

        # ===== 先验相关 =====
        self.use_priors = use_priors
        self.gamma_prior = gamma_prior  # 仅用于 repr，真正权重在 weight_dict["loss_prior"]

        # 垂直先验区间 [num_classes, 2]
        if class_vertical_ranges is not None:
            vr = torch.as_tensor(class_vertical_ranges, dtype=torch.float32)
            assert vr.shape[0] == self.num_classes
            assert vr.shape[1] == 2
            self.register_buffer("class_vertical_ranges", vr, persistent=False)
        else:
            self.class_vertical_ranges = None

        # 类别集合
        self.ground_classes = ground_classes  # list[int] or None
        self.heavy_classes = heavy_classes    # list[int] or None
        self.sky_class = sky_class
        self.road_class = road_class

        # 3 个先验的自适应融合权重（softmax 之后是 α_vert, α_sup, α_topo）
        if self.use_priors:
            self.prior_logits = nn.Parameter(torch.zeros(3))
        else:
            self.register_parameter("prior_logits", None)

    # ------------------------------------------------------------------------
    # 基础监督 loss
    # ------------------------------------------------------------------------
    def loss_labels(self, outputs, targets, indices, num_masks):
        """Classification loss (NLL).
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float()

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        )
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(
            src_logits.transpose(1, 2), target_classes, self.empty_weight
        )
        losses = {"loss_ce": loss_ce}
        return losses

    def loss_masks(self, outputs, targets, indices, num_masks):
        """Compute the losses related to the masks: the BCE loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]

        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # N x 1 x H x W
        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]

        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        losses = {
            "loss_mask": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks),
            "loss_dice": dice_loss_jit(point_logits, point_labels, num_masks),
        }

        del src_masks
        del target_masks
        return losses

    # ------------------------------------------------------------------------
    # helper: 从 query 输出组合出 per-pixel per-class 概率 P[b,c,h,w]
    # ------------------------------------------------------------------------
    def _compute_class_prob_maps(self, outputs):
        """
        outputs["pred_logits"]: [B, Q, C+1] (最后一类为 no-object)
        outputs["pred_masks"]:  [B, Q, H, W]
        返回:
            P: [B, C, H, W]，按类归一化（每个像素所有类和为 1）
        """
        pred_logits = outputs["pred_logits"]  # [B, Q, C+1]
        pred_masks = outputs["pred_masks"]    # [B, Q, H, W]

        B, Q, C_plus_1 = pred_logits.shape
        C = C_plus_1 - 1

        class_prob = F.softmax(pred_logits, dim=-1)[..., :C]  # [B, Q, C]
        mask_prob = pred_masks.sigmoid()                      # [B, Q, H, W]

        # P[b,c,h,w] = sum_q P(class=c | q) * P(mask=1 | q,h,w)
        P = torch.einsum("bqc,bqhw->bchw", class_prob, mask_prob)

        # 按类归一化，避免多个 query 堆叠过大
        denom = P.sum(dim=1, keepdim=True) + 1e-6
        P = P / denom
        return P  # [B, C, H, W]

    # ------------------------------------------------------------------------
    # 先验 1：垂直分布约束（vertical prior），基于全图 P[b,c,h,w]
    # ------------------------------------------------------------------------
    def prior_vertical_scalar(self, outputs):
        """
        scalar 垂直先验 loss，用于后面自适应加权。
        思路：某类在“不合理高度”上的概率质量越大，惩罚越大。
        """
        if (not self.use_priors) or (self.class_vertical_ranges is None):
            return outputs["pred_logits"].sum() * 0.0

        P = self._compute_class_prob_maps(outputs)  # [B, C, H, W]
        B, C, H, W = P.shape
        device = P.device

        # 归一化行坐标 y ∈ (0,1)，0 在最上，1 在最下
        y = (torch.arange(H, device=device, dtype=torch.float32) + 0.5) / H  # [H]
        y = y.view(1, 1, H)  # [1,1,H]

        ranges = self.class_vertical_ranges.to(device)  # [C,2]
        y_min = ranges[:, 0].view(1, C, 1)  # [1,C,1]
        y_max = ranges[:, 1].view(1, C, 1)  # [1,C,1]

        # 哪些行对该类是“不合理”的
        invalid = ((y < y_min) | (y > y_max)).float()  # [1,C,H]

        # p_c,y = 按列平均后的概率
        p_cy = P.mean(dim=3)  # [B,C,H]

        penalty = (p_cy * invalid).sum(dim=[1, 2]).mean()  # batch 取平均
        return penalty

    # ------------------------------------------------------------------------
    # 先验 2：支撑 / 接触先验（support prior），基于类存在强度
    # ------------------------------------------------------------------------
    def prior_support_scalar(self, outputs):
        """
        简化版支撑先验：
        - heavy 类（车、人等）强存在时，ground 类（road/sidewalk/terrain）也应该强存在；
        否则惩罚 q_heavy > q_ground 的情况。
        """
        if (not self.use_priors) or (self.ground_classes is None) or (self.heavy_classes is None):
            return outputs["pred_logits"].sum() * 0.0

        P = self._compute_class_prob_maps(outputs)  # [B,C,H,W]
        B, C, H, W = P.shape
        device = P.device

        # 每一类在整图上的“存在强度” q_c = max_{h,w} P[b,c,h,w]
        q = P.view(B, C, -1).max(dim=2).values  # [B,C]

        ground = torch.as_tensor(self.ground_classes, dtype=torch.long, device=device)
        heavy = torch.as_tensor(self.heavy_classes, dtype=torch.long, device=device)

        if ground.numel() == 0 or heavy.numel() == 0:
            return outputs["pred_logits"].sum() * 0.0

        q_ground = q[:, ground].max(dim=1).values  # [B]
        q_heavy = q[:, heavy].max(dim=1).values    # [B]

        # 如果 heavy 明显强于 ground，则认为“不着地”，惩罚
        violation = F.relu(q_heavy - q_ground)
        return violation.mean()

    # ------------------------------------------------------------------------
    # 先验 3：拓扑先验（topology prior），示例：sky 必须在 road 上方
    # ------------------------------------------------------------------------
    def prior_topology_sky_road_scalar(self, outputs):
        """
        简化拓扑先验：
        - 当 sky 和 road 都明显存在时，要求 sky 的垂直重心在 road 之上：
          L_topo = mean_b ReLU( y_sky - y_road )
        """
        if (not self.use_priors) or (self.sky_class is None) or (self.road_class is None):
            return outputs["pred_logits"].sum() * 0.0

        P = self._compute_class_prob_maps(outputs)  # [B,C,H,W]
        B, C, H, W = P.shape
        device = P.device

        # p_c,y
        p_cy = P.mean(dim=3)  # [B,C,H]

        y = (torch.arange(H, device=device, dtype=torch.float32) + 0.5) / H  # [H]
        y = y.view(1, 1, H)  # [1,1,H]

        denom = p_cy.sum(dim=2, keepdim=True) + 1e-6  # [B,C,1]
        y_expect = (p_cy * y).sum(dim=2, keepdim=True) / denom  # [B,C,1]

        sky_y = y_expect[:, self.sky_class, 0]   # [B]
        road_y = y_expect[:, self.road_class, 0] # [B]

        # 只在二者都“存在明显”的图上施加约束，避免噪声
        p_tot = P.view(B, C, -1).mean(dim=2)  # [B,C]
        sky_present = p_tot[:, self.sky_class] > 1e-3
        road_present = p_tot[:, self.road_class] > 1e-3
        valid = sky_present & road_present

        if valid.sum() == 0:
            return outputs["pred_logits"].sum() * 0.0

        v_sky = sky_y[valid]
        v_road = road_y[valid]

        violation = F.relu(v_sky - v_road)  # sky 要更“高”，否则惩罚
        return violation.mean()

    # ------------------------------------------------------------------------
    # 索引工具
    # ------------------------------------------------------------------------
    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat(
            [torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)]
        )
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    # ------------------------------------------------------------------------
    # 基础 loss 映射（注意：这里只保留 labels / masks）
    # ------------------------------------------------------------------------
    def get_loss(self, loss, outputs, targets, indices, num_masks):
        loss_map = {
            "labels": self.loss_labels,
            "masks": self.loss_masks,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks)

    # ------------------------------------------------------------------------
    # 总 forward
    # ------------------------------------------------------------------------
    def forward(self, outputs, targets):
        """
        This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {
            k: v for k, v in outputs.items() if k != "aux_outputs"
        }

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # 1) 基础 loss（只对主输出）
        losses = {}
        for loss in self.losses:  # 建议外部只传 ["labels", "masks"]
            losses.update(
                self.get_loss(loss, outputs, targets, indices, num_masks)
            )

        # 2) aux 输出的基础 loss（不加先验）
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices_aux = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(
                        loss, aux_outputs, targets, indices_aux, num_masks
                    )
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # 3) 物理先验：只在主输出上施加，自适应融合
        if self.use_priors:
            main_outputs = {
                "pred_logits": outputs_without_aux["pred_logits"],
                "pred_masks": outputs_without_aux["pred_masks"],
            }

            L_vert = self.prior_vertical_scalar(main_outputs)
            L_sup = self.prior_support_scalar(main_outputs)
            L_topo = self.prior_topology_sky_road_scalar(main_outputs)

            prior_vec = torch.stack([L_vert, L_sup, L_topo], dim=0)  # [3]

            alpha = F.softmax(self.prior_logits, dim=0)  # [3]
            L_prior = (alpha * prior_vec).sum()

            # 真正用于反向传播的先验总 loss
            losses["loss_prior"] = L_prior

            # 方便日志观察各子项（不参与加权，不一定要写到 weight_dict）
            losses["loss_prior_vertical_raw"] = L_vert.detach()
            losses["loss_prior_support_raw"] = L_sup.detach()
            losses["loss_prior_topology_raw"] = L_topo.detach()

        return losses

    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
            "num_points: {}".format(self.num_points),
            "oversample_ratio: {}".format(self.oversample_ratio),
            "importance_sample_ratio: {}".format(self.importance_sample_ratio),
            "use_priors: {}".format(self.use_priors),
            "gamma_prior (external): {}".format(self.gamma_prior),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
