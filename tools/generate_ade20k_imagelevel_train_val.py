#!/usr/bin/env python3
"""
Generate ADE20K image-level weak-supervision labels.

ADE20K masks use 0 as void and 1..150 as semantic classes. The JSON labels use
reduced indices 0..149 so CorrCLIP filtering matches the model output.

Output:
    <ADE_ROOT>/ImageSets/ImageLevel/train_imagelevel.json
    <ADE_ROOT>/ImageSets/ImageLevel/val_imagelevel.json

Usage:
    python tools/generate_ade20k_imagelevel_train_val.py --ade-root data/ADEChallengeData2016
"""

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image


DEFAULT_ADE_ROOT = "data/ADEChallengeData2016"
DEFAULT_CLASS_CONFIG = "configs/cls_ade20k.txt"


def parse_args():
    parser = argparse.ArgumentParser(description="Generate ADE20K image-level label JSON")
    parser.add_argument("--ade-root", default=DEFAULT_ADE_ROOT, help="ADEChallengeData2016 dataset root")
    parser.add_argument("--config-path", default=DEFAULT_CLASS_CONFIG, help="ADE20K class definition file")
    return parser.parse_args()


def load_classes(config_path: str) -> List[str]:
    """Load ADE20K class names. The file should not include background."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Class config not found: {path}")

    classes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            classes.append(line.split(";")[0].strip())
    return classes


def extract_reduced_labels(mask_path: Path, num_classes: int) -> List[int]:
    """Extract labels from an ADE20K mask and convert 1..150 to 0..149."""
    mask = np.array(Image.open(mask_path))
    labels = {
        int(value) - 1
        for value in np.unique(mask)
        if 1 <= int(value) <= num_classes
    }
    return sorted(labels)


def build_payload(ade_root: Path, split_dir: str, split_name: str, classes: List[str]):
    image_dir = ade_root / "images" / split_dir
    ann_dir = ade_root / "annotations" / split_dir
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not ann_dir.is_dir():
        raise FileNotFoundError(f"Annotation directory not found: {ann_dir}")

    img_paths = sorted(image_dir.glob("*.jpg"))
    print(f"Processing ADE20K {split_name} ({split_dir}): {len(img_paths)} images")

    images = []
    for idx, img_path in enumerate(img_paths, start=1):
        mask_path = ann_dir / f"{img_path.stem}.png"
        if not mask_path.is_file():
            print(f"Warning: missing annotation, skipped {mask_path}")
            continue

        images.append({
            "img_id": img_path.stem,
            "file_name": f"images/{split_dir}/{img_path.name}",
            "labels": extract_reduced_labels(mask_path, len(classes)),
        })

        if idx % 1000 == 0:
            print(f"  {idx}/{len(img_paths)}")

    return {
        "data_root": str(ade_root),
        "label_protocol": "ade20k_reduce_zero_label_for_corrclip",
        "images": images,
        "classes": classes,
    }


def main():
    args = parse_args()
    ade_root = Path(args.ade_root)
    classes = load_classes(args.config_path)
    if len(classes) != 150:
        raise ValueError(f"ADE20K should have 150 classes, got {len(classes)}")

    out_root = ade_root / "ImageSets" / "ImageLevel"
    out_root.mkdir(parents=True, exist_ok=True)

    for split_dir, split_name in [("training", "train"), ("validation", "val")]:
        payload = build_payload(ade_root, split_dir, split_name, classes)
        out_path = out_root / f"{split_name}_imagelevel.json"
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote {len(payload['images'])} entries -> {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
