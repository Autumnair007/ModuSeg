#!/usr/bin/env python3
"""
Generate COCO2014 segmentation split lists from SegmentationClass directory.

Scans SegmentationClass/{train2014,val2014}/ for existing PNG masks and writes
image name lists. Only images with segmentation annotations are included.

Expected counts: train=82081, val=40137 (vs full COCO train=82783, val=40504).

Output:
    <COCO_ROOT>/ImageSets/coco_train.txt
    <COCO_ROOT>/ImageSets/coco_val.txt

Usage:
    python tools/generate_coco_split_from_segmentationclass.py --coco-root data/COCO2014
"""

import argparse
from pathlib import Path

# ---------------------------------------------------------------------------
# Configurable paths
# ---------------------------------------------------------------------------
DEFAULT_COCO_ROOT = 'data/COCO2014'


def parse_args():
    ap = argparse.ArgumentParser(description='Generate COCO split lists from SegmentationClass')
    ap.add_argument('--coco-root', default=DEFAULT_COCO_ROOT, help='COCO2014 dataset root')
    ap.add_argument('--dry-run', action='store_true', help='Only count, do not write files')
    return ap.parse_args()


def generate_split_files(coco_root: str, dry_run: bool = False):
    coco_root = Path(coco_root)
    segclass_dir = coco_root / 'SegmentationClass'
    if not segclass_dir.exists():
        raise FileNotFoundError(f'SegmentationClass not found: {segclass_dir}')

    imageset_dir = coco_root / 'ImageSets'
    imageset_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for split in ['train2014', 'val2014']:
        split_dir = segclass_dir / split
        if not split_dir.exists():
            print(f'Warning: {split_dir} not found, skipping')
            continue

        png_files = sorted(split_dir.glob('*.png'))
        image_names = [f.stem for f in png_files]

        out_name = 'coco_train.txt' if 'train' in split else 'coco_val.txt'
        out_path = imageset_dir / out_name

        print(f'[{split}] Found {len(image_names)} annotated images')
        if not dry_run:
            with open(out_path, 'w') as f:
                for name in image_names:
                    f.write(f'{name}\n')
            print(f'  -> Wrote: {out_path}')

        results[split] = len(image_names)

    print('\n' + '=' * 50)
    print('Summary:')
    for split, count in results.items():
        expected = 82081 if 'train' in split else 40137
        status = 'OK' if count == expected else 'MISMATCH'
        print(f'  {split}: {count} ({status}, expected: {expected})')
    print('=' * 50)
    return results


def main():
    args = parse_args()
    generate_split_files(args.coco_root, args.dry_run)


if __name__ == '__main__':
    main()
