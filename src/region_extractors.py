"""Region extraction utilities for feature bank building."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from configs.config import (
    DATASET_TYPE,
    DATASET_CLASSES,
    MIN_MASK_AREA,
    BUILD_MASK_EROSION_KERNEL_SIZE,
    BUILD_MASK_EROSION_ITERATIONS,
    PSEUDO_MASK_ROOT,
)


def _get_erosion_kernel():
    return np.ones((BUILD_MASK_EROSION_KERNEL_SIZE, BUILD_MASK_EROSION_KERNEL_SIZE), np.uint8)


def _erode_if_enabled(mask, kernel):
    return cv2.erode(mask, kernel, iterations=BUILD_MASK_EROSION_ITERATIONS)


def _extract_class_regions(seg_map: np.ndarray, min_area: int, kernel):
    """Extract per-class binary masks from a segmentation map."""
    regions = []
    for cid in range(1, len(DATASET_CLASSES) + 1):
        mask_c = _erode_if_enabled((seg_map == cid).astype(np.uint8), kernel)
        if mask_c.sum() >= min_area:
            regions.append({"cls_id": cid, "mask": mask_c})

    bg = _erode_if_enabled((seg_map == 0).astype(np.uint8), kernel)
    if bg.sum() >= min_area:
        regions.append({"cls_id": 0, "mask": bg})
    return regions


def get_regions_from_corrclip(mask_generator, img_path: str, min_area: int = MIN_MASK_AREA):
    """Generate semantic segmentation via CorrCLIP and return per-class regions."""
    seg_pred_np = mask_generator.segment(img_path)
    return _extract_class_regions(seg_pred_np, min_area, _get_erosion_kernel())


def get_regions_from_pseudo(img_path: str, min_area: int = MIN_MASK_AREA):
    """Load pre-generated pseudo masks and split into per-class regions."""
    from configs.config import DATASET_TRAIN_SPLIT

    img_path = Path(img_path)
    pseudo_root = Path(PSEUDO_MASK_ROOT)

    if DATASET_TYPE == "coco":
        pseudo_path = pseudo_root / DATASET_TRAIN_SPLIT / f"{img_path.name.replace('.jpg', '.png')}"
    else:
        pseudo_path = pseudo_root / f"{img_path.stem}.png"

    if not pseudo_path.is_file():
        raise FileNotFoundError(
            f"Pseudo mask not found: {pseudo_path}\n"
            f"Run: python tools/generate_pseudo_masks.py --dataset {DATASET_TYPE} --model {PSEUDO_MASK_ROOT.split('/')[-1]}"
        )

    pseudo_mask = np.array(Image.open(pseudo_path))
    return _extract_class_regions(pseudo_mask, min_area, _get_erosion_kernel())


__all__ = [
    "get_regions_from_corrclip",
    "get_regions_from_pseudo",
]
