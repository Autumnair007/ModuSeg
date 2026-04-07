"""Evaluation metrics utility."""

from typing import List

import numpy as np
import torch

from configs.config import DATASET_CLASSES


def evaluate_dataset_miou(
    pred_list: List[np.ndarray],
    gt_list: List[np.ndarray],
    num_classes: int,
    ignore_index: int,
):
    """Compute mIoU using torchmetrics."""
    try:
        from torchmetrics.classification import MulticlassJaccardIndex
    except ImportError:
        raise RuntimeError(
            "torchmetrics not installed. Run: pip install torchmetrics"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    metric = MulticlassJaccardIndex(num_classes=num_classes, ignore_index=ignore_index, average=None).to(device)

    for pred, gt in zip(pred_list, gt_list):
        gt = np.asarray(gt).squeeze()
        assert pred.shape == gt.shape, f"Shape mismatch: pred {pred.shape} vs gt {gt.shape}"

        pred_tensor = torch.from_numpy(pred).long().to(device)
        gt_tensor = torch.from_numpy(gt).long().to(device)
        metric.update(pred_tensor, gt_tensor)

    per_class_iou = metric.compute()
    miou = per_class_iou.mean()

    results = {"mIoU": float(miou.cpu().numpy()), "per_class_iou": per_class_iou.cpu().numpy()}

    summary_lines = [f"mIoU: {results['mIoU']:.4f}"]
    summary_lines.append("\nPer-class IoU:")
    class_names = ["background"] + DATASET_CLASSES
    for i, iou_val in enumerate(per_class_iou.cpu().numpy()):
        if i < len(class_names):
            summary_lines.append(f"  {class_names[i]}: {iou_val:.4f}")

    summary_str = "\n".join(summary_lines)
    return summary_str, results, class_names


__all__ = [
    "evaluate_dataset_miou",
]
