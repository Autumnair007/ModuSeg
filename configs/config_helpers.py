from __future__ import annotations

import os
from typing import Optional, List, Union


def _is_none_string(val: Optional[str]) -> bool:
    return val is None or val.strip().lower() == 'none' or val.strip() == ''


def env_get_str(key: str, default: str) -> str:
    val = os.environ.get(key)
    return default if _is_none_string(val) else val


def env_get_int(key: str, default: int) -> int:
    val = os.environ.get(key)
    if _is_none_string(val):
        return default
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def env_get_float(key: str, default: float) -> float:
    val = os.environ.get(key)
    if _is_none_string(val):
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def env_get_bool(key: str, default: bool) -> bool:
    val = os.environ.get(key)
    if _is_none_string(val):
        return default
    return val.strip().lower() == 'true'


def load_classes(name_file: str) -> List[str]:
    """Load foreground class list from class file (skip first background line)."""
    with open(name_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    classes: List[str] = []
    for line in lines[1:]:
        parts = line.strip().split(";")
        if parts and parts[0].strip():
            classes.append(parts[0].strip())
    return classes


def get_model_version_str(
    backbone_type: Optional[str] = None,
    model_name: Optional[str] = None,
) -> str:
    """Return feature extractor version string for output file naming."""
    return "c-radiov4-so400m"


def get_mask_quality_str(
    pseudo_mask_type: Optional[str] = None,
) -> str:
    """Return mask quality identifier for output file naming."""
    if pseudo_mask_type is None:
        from configs import config as _config
        pseudo_mask_type = str(getattr(_config, "PSEUDO_MASK_TYPE", "corrclip"))
    return f"{str(pseudo_mask_type)}-pseudo"
