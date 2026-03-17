# mask2former/data/datasets/darkzurich_semantic.py
import os
import glob
from detectron2.data import DatasetCatalog, MetadataCatalog

_DARKZURICH_ROOT = "/home/lab106-2/data/baichen/sun/Mask2Former/datasets/DarkZurich"  # 你实际的根路径

def _load_darkzurich_semantic(split: str):
    """
    暂时只支持带标注的 night val（50 张），作为 supervised 语义分割数据集。
    """
    assert split in ["val"], split

    base = os.path.join(_DARKZURICH_ROOT, "Dark_Zurich_val_anon")
    img_dir = os.path.join(base, "rgb_anon", "val", "night")
    gt_dir  = os.path.join(base, "gt",       "val", "night")

    dataset_dicts = []

    # sequence 目录，例如 GOPR0351、GP010364_ref 等
    for seq in sorted(os.listdir(img_dir)):
        seq_img_dir = os.path.join(img_dir, seq)
        if not os.path.isdir(seq_img_dir):
            continue

        # 图像名：{sequence}_frame_{frame:0>6}_rgb_anon.png
        for img_path in sorted(glob.glob(os.path.join(seq_img_dir, "*_rgb_anon.png"))):
            file_name = os.path.basename(img_path)
            stem = file_name.replace("_rgb_anon.png", "")

            gt_path = os.path.join(gt_dir, seq, stem + "_gt_labelTrainIds.png")
            if not os.path.isfile(gt_path):
                raise FileNotFoundError(gt_path)

            record = {
                "file_name": img_path,
                "sem_seg_file_name": gt_path,
                # Dark Zurich 分辨率一般是 1080x1920，你也可以实际读一张图再写
                "height": 1080,
                "width": 1920,
            }
            dataset_dicts.append(record)

    return dataset_dicts


def register_darkzurich_semantic(root: str = "/home/lab106-2/data/baichen/sun/Mask2Former/datasets/DarkZurich"):
    global _DARKZURICH_ROOT
    _DARKZURICH_ROOT = root

    print("[DARKZURICH-NEW] auto registering Dark Zurich semantic datasets...")

    split = "val"
    name = f"darkzurich_semantic_{split}"

    DatasetCatalog.register(
        name,
        # 用默认参数捕获 split
        lambda s=split: _load_darkzurich_semantic(s),
    )

    # 直接复用 Cityscapes 的类别与颜色，保证 19 类对齐
    city_meta = MetadataCatalog.get("cityscapes_fine_sem_seg_train")
    meta = MetadataCatalog.get(name)
    meta.set(
        stuff_classes=getattr(city_meta, "stuff_classes", None),
        thing_classes=getattr(city_meta, "thing_classes", None),
        stuff_colors=getattr(city_meta, "stuff_colors", None),
        evaluator_type="sem_seg",
        ignore_label=255,
    )
