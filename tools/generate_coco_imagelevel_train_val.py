#!/usr/bin/env python3
"""
Generate COCO2014 image-level weak-supervision labels.

Extracts per-image class indices from instances_{train,val}2014.json.
Only images present in SegmentationClass/ are included (train=82081, val=40137).

COCO original category IDs (1-90, non-contiguous, 80 valid) are mapped to
contiguous indices 1-80. Index 0 = background (from cls_coco_object.txt line 1).

Output:
    <COCO_ROOT>/annotations/train_imagelevel.json  (82081 entries)
    <COCO_ROOT>/annotations/val_imagelevel.json    (40137 entries)

Usage:
    python tools/generate_coco_imagelevel_train_val.py --coco-root data/COCO2014
"""

import argparse
import os
import os.path as osp
import json
import glob
from typing import List, Dict, Set
from collections import defaultdict

# ---------------------------------------------------------------------------
# Configurable paths
# ---------------------------------------------------------------------------
DEFAULT_COCO_ROOT = 'data/COCO2014'
DEFAULT_CLASS_CONFIG = 'configs/cls_coco_object.txt'


def parse_args():
    ap = argparse.ArgumentParser(description='Generate COCO2014 image-level label JSON')
    ap.add_argument('--coco-root', default=DEFAULT_COCO_ROOT, help='COCO2014 dataset root')
    ap.add_argument('--config-path', default=DEFAULT_CLASS_CONFIG, help='Class definition file')
    return ap.parse_args()


def load_classes(config_path: str) -> List[str]:
    """Load class names from config file (semicolon-separated synonyms; first token used)."""
    if not osp.isfile(config_path):
        raise FileNotFoundError(f'Class config not found: {config_path}')
    with open(config_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    classes = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        classes.append(line.split(';')[0].strip())
    return classes


def build_coco_to_contiguous_mapping() -> Dict[int, int]:
    """Map COCO original category IDs (1-90, 80 valid) to contiguous indices 1-80."""
    coco_80_ids = [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20,
        21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
        41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
        59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
        80, 81, 82, 84, 85, 86, 87, 88, 89, 90
    ]
    return {coco_id: idx for idx, coco_id in enumerate(coco_80_ids, start=1)}


def extract_imagelevel_labels(instances_json: str, coco_to_contiguous: Dict[int, int]) -> Dict[int, Set[int]]:
    """Extract per-image class indices from instances JSON. Returns {image_id: set of class indices}."""
    if not osp.isfile(instances_json):
        raise FileNotFoundError(f'Annotation file not found: {instances_json}')
    with open(instances_json, 'r') as f:
        data = json.load(f)

    image_labels = defaultdict(set)
    for ann in data['annotations']:
        coco_cat_id = ann['category_id']
        if coco_cat_id in coco_to_contiguous:
            image_labels[ann['image_id']].add(coco_to_contiguous[coco_cat_id])

    # Add background (index 0) to all annotated images
    for img_id in image_labels:
        image_labels[img_id].add(0)

    return image_labels


def load_segmentationclass_list(coco_root: str, split: str) -> Set[str]:
    """Load image stems that have segmentation masks (from ImageSets list or SegmentationClass dir)."""
    split_name = 'train' if 'train' in split else 'val'
    list_file = osp.join(coco_root, 'ImageSets', f'coco_{split_name}.txt')

    if osp.isfile(list_file):
        with open(list_file, 'r') as f:
            return {line.strip() for line in f if line.strip()}

    segclass_dir = osp.join(coco_root, 'SegmentationClass', split)
    if osp.isdir(segclass_dir):
        return {osp.splitext(osp.basename(f))[0] for f in glob.glob(osp.join(segclass_dir, '*.png'))}

    print(f"Warning: cannot find {list_file} or {segclass_dir}, no filtering applied")
    return set()


def build_payload(coco_root: str, split: str, instances_json: str,
                  coco_to_contiguous: Dict[int, int], classes: List[str]):
    """Build JSON payload. Only includes images present in SegmentationClass."""
    image_labels = extract_imagelevel_labels(instances_json, coco_to_contiguous)
    valid_images = load_segmentationclass_list(coco_root, split)
    print(f"SegmentationClass valid images: {len(valid_images)}")

    with open(instances_json, 'r') as f:
        data = json.load(f)

    images = []
    skipped = 0
    print(f"Processing {split} ({len(data['images'])} total images)...")

    for i, img_info in enumerate(data['images']):
        img_id = img_info['id']
        file_name = img_info['file_name']
        img_stem = osp.splitext(file_name)[0]

        # Only keep images with segmentation annotations
        if valid_images and img_stem not in valid_images:
            skipped += 1
            continue

        labels = sorted(list(image_labels.get(img_id, {0})))
        images.append({
            'img_id': str(img_id),
            'file_name': f'images/{split}/{file_name}',
            'labels': labels
        })
        if (i + 1) % 10000 == 0:
            print(f"  {i + 1}/{len(data['images'])}")

    print(f"  After filtering: {len(images)} (skipped {skipped})")
    return {'data_root': coco_root, 'images': images, 'classes': classes}


def main():
    args = parse_args()
    coco_root = args.coco_root

    classes = load_classes(args.config_path)
    print(f"Loaded {len(classes)} classes from {args.config_path}")

    coco_to_contiguous = build_coco_to_contiguous_mapping()
    print(f"COCO ID mapping built (e.g. COCO ID 1 -> Index {coco_to_contiguous[1]})")

    out_root = osp.join(coco_root, 'annotations')
    os.makedirs(out_root, exist_ok=True)

    for split in ['train2014', 'val2014']:
        instances_json = osp.join(coco_root, 'annotations', f'instances_{split}.json')
        try:
            payload = build_payload(coco_root, split, instances_json, coco_to_contiguous, classes)
            split_name = 'train' if 'train' in split else 'val'
            out_path = osp.join(out_root, f'{split_name}_imagelevel.json')
            with open(out_path, 'w') as f:
                json.dump(payload, f, indent=2)
            print(f"Wrote {len(payload['images'])} entries -> {out_path}")
        except FileNotFoundError as e:
            print(f"Skipping {split}: {e}")

    print("Done.")


if __name__ == '__main__':
    main()
