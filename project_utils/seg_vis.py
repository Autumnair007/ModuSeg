"""Semantic segmentation visualization utilities."""

from typing import List

import numpy as np
import cv2

from configs.config import DATASET_CLASSES


def voc_color_map(n=256, normalized=False):
    """Create VOC color map."""

    def bitget(byteval, idx):
        return (byteval & (1 << idx)) != 0

    cmap = np.zeros((n, 3), dtype="float32" if normalized else "uint8")
    for i in range(n):
        r = g = b = 0
        c = i
        for j in range(8):
            r |= (1 << (7 - j)) if bitget(c, 0) else 0
            g |= (1 << (7 - j)) if bitget(c, 1) else 0
            b |= (1 << (7 - j)) if bitget(c, 2) else 0
            c >>= 3
        cmap[i] = np.array([r, g, b])
    if normalized:
        cmap = cmap / 255.0
    return cmap


def render_segmentation(seg: np.ndarray) -> np.ndarray:
    """Render segmentation as color image."""
    cmap = voc_color_map(256, normalized=False)
    h, w = seg.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    color[:] = 0
    vals = np.unique(seg)
    for v in vals:
        if v == 255:
            continue
        color[seg == v] = cmap[v]
    return color


def overlay_segmentation(img_rgb: np.ndarray, seg: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """Overlay segmentation on original image."""
    color = render_segmentation(seg)
    color = cv2.resize(color, (img_rgb.shape[1], img_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    overlay = (alpha * color + (1 - alpha) * img_rgb).astype(np.uint8)
    return overlay[..., ::-1].copy()  # RGB->BGR


def render_legend_panel(class_ids: List[int], height: int) -> np.ndarray:
    """Render right-side legend panel (BGR format)."""
    class_ids = sorted([cid for cid in class_ids if cid > 0])
    if len(class_ids) == 0:
        panel = np.full((height, 200, 3), 255, dtype=np.uint8)
        cv2.putText(
            panel,
            "No classes",
            (20, min(40, height - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        return panel

    pad = 14
    box_w = 26
    row_h = 30
    names = ["background"] + DATASET_CLASSES
    max_text = max((names[cid] if cid < len(names) else f"cls_{cid}") for cid in class_ids)
    text_size, _ = cv2.getTextSize(max_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    legend_w = pad + box_w + 8 + max(160, text_size[0]) + pad
    legend_w = max(200, min(legend_w, 420))
    panel = np.full((height, legend_w, 3), 255, dtype=np.uint8)

    cmap = voc_color_map(256, normalized=False)
    y0 = pad
    for i, cid in enumerate(class_ids):
        y = y0 + i * row_h
        if y + box_w + pad >= height:
            break
        color_rgb = cmap[cid]
        color_bgr = (int(color_rgb[2]), int(color_rgb[1]), int(color_rgb[0]))
        x = pad
        cv2.rectangle(panel, (x, y), (x + box_w, y + box_w), color_bgr, thickness=-1)
        cv2.rectangle(panel, (x, y), (x + box_w, y + box_w), (0, 0, 0), thickness=1)
        txt = names[cid] if cid < len(names) else f"cls_{cid}"
        cv2.putText(
            panel,
            txt,
            (x + box_w + 8, y + int(box_w * 0.8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
    return panel


__all__ = [
    "voc_color_map",
    "render_segmentation",
    "overlay_segmentation",
    "render_legend_panel",
]
