#!/usr/bin/env python3
"""
Generate Cityscapes image-level weak-supervision labels.

Cityscapes semantic labels are converted from labelIds to official trainIds
0..18. Void and ignored labels are excluded from the JSON.

Output:
    <CITYSCAPES_ROOT>/ImageSets/ImageLevel/train_imagelevel.json
    <CITYSCAPES_ROOT>/ImageSets/ImageLevel/val_imagelevel.json

Usage:
    python tools/generate_cityscapes_imagelevel_train_val.py --cityscapes-root data/CityScapes
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image


DEFAULT_CITYSCAPES_ROOT = "data/CityScapes"
DEFAULT_CLASS_CONFIG = "configs/cls_city_scapes.txt"

CITYSCAPES_ID_TO_TRAIN_ID = np.full(256, 255, dtype=np.uint8)
for _label_id, _train_id in {
    7: 0,
    8: 1,
    11: 2,
    12: 3,
    13: 4,
    17: 5,
    19: 6,
    20: 7,
    21: 8,
    22: 9,
    23: 10,
    24: 11,
    25: 12,
    26: 13,
    27: 14,
    28: 15,
    31: 16,
    32: 17,
    33: 18,
}.items():
    CITYSCAPES_ID_TO_TRAIN_ID[_label_id] = _train_id

CITYSCAPES_NAME_TO_TRAIN_ID = {
    "road": 0,
    "sidewalk": 1,
    "building": 2,
    "wall": 3,
    "fence": 4,
    "pole": 5,
    "trafficlight": 6,
    "traffic light": 6,
    "trafficsign": 7,
    "traffic sign": 7,
    "vegetation": 8,
    "terrain": 9,
    "sky": 10,
    "person": 11,
    "people": 11,
    "rider": 12,
    "car": 13,
    "truck": 14,
    "bus": 15,
    "train": 16,
    "motorcycle": 17,
    "bicycle": 18,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Cityscapes image-level label JSON")
    parser.add_argument(
        "--cityscapes-root",
        default=DEFAULT_CITYSCAPES_ROOT,
        help="Cityscapes dataset root",
    )
    parser.add_argument(
        "--config-path",
        default=DEFAULT_CLASS_CONFIG,
        help="Cityscapes 19-class definition file",
    )
    return parser.parse_args()


def load_classes(config_path: str) -> List[str]:
    """Load Cityscapes class names. The file should not include background."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Class config not found: {path}")

    classes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            classes.append(line.split(";")[0].strip())
    return classes


def cityscapes_image_to_gt_path(img_path: Path, cityscapes_root: Path) -> Path:
    """Infer the gtFine labelIds path from a leftImg8bit image path."""
    split = img_path.parts[img_path.parts.index("leftImg8bit") + 1]
    city = img_path.parent.name
    base = img_path.stem.replace("_leftImg8bit", "")
    return cityscapes_root / "gtFine" / split / city / f"{base}_gtFine_labelIds.png"


def label_name_to_train_id(label_name: str) -> Optional[int]:
    """Convert a Cityscapes polygon label name to a trainId."""
    normalized = label_name.strip().lower().replace("_", " ")
    normalized = " ".join(normalized.split())
    compact = normalized.replace(" ", "")

    if compact.endswith("group"):
        compact = compact[:-5]
        normalized = normalized[:-5].strip()

    return CITYSCAPES_NAME_TO_TRAIN_ID.get(normalized, CITYSCAPES_NAME_TO_TRAIN_ID.get(compact))


def valid_train_ids_from_label_ids(label_ids: np.ndarray) -> List[int]:
    """Extract valid trainIds from a Cityscapes labelIds mask."""
    label_ids = np.asarray(label_ids)
    valid = (label_ids >= 0) & (label_ids < len(CITYSCAPES_ID_TO_TRAIN_ID))
    train_ids = CITYSCAPES_ID_TO_TRAIN_ID[label_ids[valid].astype(np.int64)]
    return sorted(int(train_id) for train_id in np.unique(train_ids) if int(train_id) != 255)


def extract_labels_from_polygons(json_path: Path) -> List[int]:
    """Extract image-level trainIds from a Cityscapes polygons.json file."""
    data = json.loads(json_path.read_text(encoding="utf-8"))

    labels = set()
    for obj in data.get("objects", []):
        train_id = label_name_to_train_id(obj.get("label", ""))
        if train_id is not None:
            labels.add(int(train_id))
    return sorted(labels)


def build_payload(cityscapes_root: Path, split: str, classes: List[str]):
    """Build the image-level JSON payload for one Cityscapes split."""
    image_dir = cityscapes_root / "leftImg8bit" / split
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    img_paths = sorted(image_dir.glob("*/*_leftImg8bit.png"))
    print(f"Processing Cityscapes {split}: {len(img_paths)} images")

    images = []
    for idx, img_path in enumerate(img_paths, start=1):
        gt_path = cityscapes_image_to_gt_path(img_path, cityscapes_root)
        if not gt_path.is_file():
            print(f"Warning: missing annotation, skipped {gt_path}")
            continue

        polygon_path = Path(str(gt_path).replace("_gtFine_labelIds.png", "_gtFine_polygons.json"))
        if polygon_path.is_file():
            labels = extract_labels_from_polygons(polygon_path)
        else:
            label_ids = np.array(Image.open(gt_path))
            labels = valid_train_ids_from_label_ids(label_ids)

        images.append({
            "img_id": img_path.stem,
            "file_name": img_path.relative_to(cityscapes_root).as_posix(),
            "gt_file_name": gt_path.relative_to(cityscapes_root).as_posix(),
            "labels": labels,
        })

        if idx % 500 == 0:
            print(f"  {idx}/{len(img_paths)}")

    return {
        "data_root": str(cityscapes_root),
        "label_protocol": "cityscapes_trainId_0_18_without_background",
        "images": images,
        "classes": classes,
    }


def main():
    args = parse_args()
    cityscapes_root = Path(args.cityscapes_root)
    classes = load_classes(args.config_path)
    if len(classes) != 19:
        raise ValueError(f"Cityscapes should have 19 classes, got {len(classes)}")

    out_root = cityscapes_root / "ImageSets" / "ImageLevel"
    out_root.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val"):
        payload = build_payload(cityscapes_root, split, classes)
        out_path = out_root / f"{split}_imagelevel.json"
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote {len(payload['images'])} entries -> {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
