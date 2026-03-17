import os
from detectron2.data import DatasetCatalog, MetadataCatalog

# Cityscapes / ACDC 的 19 个 trainIds 类别
ACDC_CLASSES = [
    "road", "sidewalk", "building", "wall", "fence",
    "pole", "traffic light", "traffic sign",
    "vegetation", "terrain", "sky",
    "person", "rider",
    "car", "truck", "bus", "train", "motorcycle", "bicycle",
]

# 和 config 里的名字保持一致
ACDC_SPLITS = {
    "acdc_semantic_train": ("train", ["fog", "night", "rain", "snow"]),
    "acdc_semantic_val":   ("val",   ["fog", "night", "rain", "snow"]),
}


def _get_acdc_root():
    """
    返回 ACDC 数据集根目录绝对路径：<Mask2Former 根>/datasets/acdc
    """
    this_dir = os.path.dirname(__file__)                 # .../mask2former/data/datasets
    proj_root = os.path.abspath(os.path.join(this_dir, "..", "..", ".."))
    return os.path.join(proj_root, "datasets", "acdc")


def load_acdc_semantic(split_name: str):
    """
    根据 split_name 构建 Detectron2 的 dataset dict 列表。
    只加入能在 gt_trainval/gt/... 找到 _gt_labelTrainIds.png 的样本。
    """
    split, conditions = ACDC_SPLITS[split_name]
    root = _get_acdc_root()

    records = []
    img_id = 0

    for cond in conditions:
        # 图像：rgb_anon_trainvaltest/rgb_anon/<cond>/<train|val>/**
        img_root = os.path.join(
            root, "rgb_anon_trainvaltest", "rgb_anon", cond, split
        )
        # 标签：gt_trainval/gt/<cond>/<train|val>/**
        gt_root = os.path.join(
            root, "gt_trainval", "gt", cond, split
        )

        if not os.path.isdir(img_root):
            print(f"[ACDC-NEW] img_root not found, skip: {img_root}")
            continue
        if not os.path.isdir(gt_root):
            print(f"[ACDC-NEW] gt_root not found, skip: {gt_root}")
            continue

        for dirpath, _, filenames in os.walk(img_root):
            rel_dir = os.path.relpath(dirpath, img_root)  # 例如 GP050176
            for fname in sorted(filenames):
                if not fname.endswith("_rgb_anon.png"):
                    continue

                img_path = os.path.join(dirpath, fname)

                # GOPR0604_frame_000496_rgb_anon.png -> GOPR0604_frame_000496_gt_labelTrainIds.png
                stem = fname.replace("_rgb_anon.png", "")
                gt_name = stem + "_gt_labelTrainIds.png"
                gt_path = os.path.join(gt_root, rel_dir, gt_name)

                if not os.path.isfile(gt_path):
                    print(
                        f"[ACDC-NEW] cannot find gt for {img_path}, expected:\n"
                        f"    {gt_path}"
                    )
                    continue

                records.append(
                    {
                        "file_name": img_path,
                        "sem_seg_file_name": gt_path,
                        "image_id": img_id,
                    }
                )
                img_id += 1

    print(f"[ACDC-NEW] loaded {len(records)} samples for split '{split_name}'")
    return records


def register_acdc_semantic():
    """
    在 Detectron2 的 DatasetCatalog / MetadataCatalog 中注册 ACDC 语义分割。
    """
    root = _get_acdc_root()
    if not os.path.isdir(root):
        print(f"[ACDC-NEW] root dir not found: {root}")
        return

    for name in ACDC_SPLITS.keys():
        DatasetCatalog.register(
            name,
            lambda n=name: load_acdc_semantic(n),
        )
        MetadataCatalog.get(name).set(
            stuff_classes=ACDC_CLASSES,
            ignore_label=255,
            evaluator_type="sem_seg",
            name=name,
        )

# ========= 关键：模块 import 时自动调用注册 =========
print("[ACDC-NEW] auto registering ACDC semantic datasets...")
register_acdc_semantic()
# ==================================================

