import json
import os
from typing import Dict, Iterable, Optional, Set

from configs.config import (
    IMAGELEVEL_JSON_PATH,
    IMAGELEVEL_IMG_ID_KEY,
    IMAGELEVEL_LABELS_KEY,
)


def load_imagelevel_labels(
    json_path: Optional[str] = None,
    img_id_key: Optional[str] = None,
    labels_key: Optional[str] = None,
) -> Dict[str, Set[int]]:
    """Load image-level label mapping: img_id -> set of allowed class indices."""
    path = json_path or IMAGELEVEL_JSON_PATH
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"Image-level label file not found: {path}")

    img_key = img_id_key or IMAGELEVEL_IMG_ID_KEY
    lbl_key = labels_key or IMAGELEVEL_LABELS_KEY

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    mapping: Dict[str, Set[int]] = {}
    for item in payload.get("images", []):
        img_id = str(item.get(img_key, "")).strip()
        labels = item.get(lbl_key, [])
        if not img_id:
            continue
        allowed = {int(v) for v in labels if isinstance(v, (int, float))}
        if not allowed:
            continue
        mapping[img_id] = allowed
    
    if not mapping:
        raise ValueError(f"Image-level label file is empty or malformed: {path}")
    
    return mapping


def build_class_mask(
    allowed: Optional[Iterable[int]],
    num_classes: int,
    device=None,
):
    """Build a boolean mask tensor from allowed class indices."""
    import torch

    if not allowed:
        return None
    mask = torch.zeros(num_classes, dtype=torch.bool, device=device)
    for cid in allowed:
        if 0 <= int(cid) < num_classes:
            mask[int(cid)] = True
    return mask
