#!/usr/bin/env python3
"""Inference script: EntitySeg proposals + C-RADIOv4 features + feature bank retrieval."""

import inspect
import os
import time

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

import configs.config as cfg_module
from configs.config import (
    DATASET_CLASSES, DATASET_TYPE, INF_CRADIOV4_CKPT, INF_CRADIOV4_IMG_SIZE,
    INF_CRADIOV4_MODEL_NAME, INF_DATASET_ROOT, INF_FEATURE_BANK_ROOT,
    INF_IGNORE_INDEX, INF_NMS_IOU, INF_NUM_CLASSES, INF_SAVE_DIR,
    INF_TOPK_NEIGH, INF_VIS_LIMIT, PSEUDO_MASK_TYPE, SEED,
)
from configs.config_helpers import get_mask_quality_str, get_model_version_str
from project_utils.config_logger import record_config
from project_utils.metrics import evaluate_dataset_miou
from project_utils.seed_utils import set_global_seed
from project_utils.seg_vis import overlay_segmentation, render_legend_panel
from src.feature_extractor_c_radiov4 import CRADIOv4FeatureExtractor
from src.inference import (
    build_faiss_index,
    init_mask_proposer,
    load_class_features,
    load_class_mapping,
    load_eval_samples,
    predict_instances,
    rasterize_instances,
    reduce_ade20k_gt_for_eval,
    reduce_ade20k_pred_for_eval,
    reduce_cityscapes_gt_for_eval,
    reduce_cityscapes_pred_for_eval,
)


def main():
    print("=" * 80)
    print(f"Inference - {DATASET_TYPE.upper()}")
    print("=" * 80)

    set_global_seed(SEED)

    dataset_root = INF_DATASET_ROOT
    feature_bank_root = INF_FEATURE_BANK_ROOT
    sample_infos = load_eval_samples(dataset_root)

    # Load feature bank
    _, fg_names, bg_names = load_class_mapping(os.path.join(feature_bank_root, 'meta'))
    class_to_feats = load_class_features(
        os.path.join(feature_bank_root, 'features'),
        list(set(fg_names + bg_names)),
    )
    assert len(class_to_feats) > 0, "Empty feature bank. Please run run_build_feature_bank.py first."

    # Build gallery features and FAISS index
    gallery_feats = []
    gallery_labels = []
    for name, mat in class_to_feats.items():
        if mat is None or mat.shape[0] == 0:
            continue
        gallery_feats.append(mat.astype(np.float32))
        gallery_labels.extend([name] * mat.shape[0])
    if len(gallery_feats) == 0:
        raise RuntimeError("No gallery features available.")

    all_feats = np.vstack(gallery_feats)
    all_feats = all_feats / (np.linalg.norm(all_feats, axis=1, keepdims=True) + 1e-8)
    all_feats = np.ascontiguousarray(all_feats.astype(np.float32))
    faiss_index = build_faiss_index(all_feats)

    # Load feature extractor and mask proposer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dino = CRADIOv4FeatureExtractor(
        INF_CRADIOV4_CKPT,
        device,
        INF_CRADIOV4_MODEL_NAME,
        INF_CRADIOV4_IMG_SIZE,
    )
    model_version = get_model_version_str()

    mask_proposer = init_mask_proposer()
    record_config(os.path.join(INF_FEATURE_BANK_ROOT, 'meta'), 'run_inference.py')

    save_dir = os.path.join(INF_FEATURE_BANK_ROOT, INF_SAVE_DIR)
    os.makedirs(save_dir, exist_ok=True)

    preds = []
    gts = []

    for sample in tqdm(sample_infos, desc=f"Infer ({model_version})", ncols=100):
        img_id = sample['id']
        img_path = sample['img']
        gt_path = sample['gt']

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        gt = None
        if gt_path and os.path.isfile(gt_path):
            gt = np.array(Image.open(gt_path))

        instances = predict_instances(
            img_rgb,
            dino,
            mask_proposer,
            all_feats,
            gallery_labels,
            faiss_index,
            INF_NMS_IOU,
            topk_neigh=INF_TOPK_NEIGH,
        )

        pred_before = rasterize_instances(instances, h, w)

        if gt is not None:
            if DATASET_TYPE == "ade20k":
                preds.append(reduce_ade20k_pred_for_eval(pred_before))
                gts.append(reduce_ade20k_gt_for_eval(gt))
            elif DATASET_TYPE == "cityscapes":
                pred_eval = reduce_cityscapes_pred_for_eval(pred_before)
                gt_eval = reduce_cityscapes_gt_for_eval(gt)
                gt_eval[pred_eval == INF_IGNORE_INDEX] = INF_IGNORE_INDEX
                preds.append(pred_eval)
                gts.append(gt_eval)
            else:
                preds.append(pred_before)
                gts.append(gt)

        # Save visualization for the first N evaluated images.
        vis_count = len([p for p, g in zip(preds, gts) if g is not None])
        if vis_count <= INF_VIS_LIMIT:
            overlay = overlay_segmentation(img_rgb, pred_before)
            present_ids = sorted(list(set(int(v) for v in np.unique(pred_before) if v not in (0, 255))))
            panel = render_legend_panel(present_ids, overlay.shape[0])
            canvas = np.full((overlay.shape[0], overlay.shape[1] + panel.shape[1], 3), 255, dtype=np.uint8)
            canvas[:, :overlay.shape[1]] = overlay
            canvas[:, overlay.shape[1]:overlay.shape[1] + panel.shape[1]] = panel
            save_path = os.path.join(save_dir, f'{img_id}.jpg')
            cv2.imwrite(save_path, canvas)

    # Evaluate
    if len(preds) == 0:
        print(f"\nNo samples for evaluation. Results saved to: {save_dir}")
        return

    print(f"\n[Eval] {len(preds)} samples ...")
    eval_num_classes = len(DATASET_CLASSES) if DATASET_TYPE in ("ade20k", "cityscapes") else INF_NUM_CLASSES
    eval_class_names = DATASET_CLASSES if DATASET_TYPE in ("ade20k", "cityscapes") else None
    summary, metrics, class_names = evaluate_dataset_miou(
        preds,
        gts,
        eval_num_classes,
        INF_IGNORE_INDEX,
        class_names=eval_class_names,
    )
    print(summary)

    per_class = metrics['per_class_iou']
    print("\nPer-class IoU Table:")
    print("+----------------------+---------+")
    print("| Class                | IoU     |")
    print("+----------------------+---------+")
    for name, iou_val in zip(class_names, per_class):
        print(f"| {name:20s} | {iou_val:7.4f} |")
    print("+----------------------+---------+")
    print(f"| {'Mean(mIoU)':20s} | {metrics['mIoU']:7.4f} |")
    print("+----------------------+---------+")

    # Save results
    mask_quality = get_mask_quality_str()
    ts = time.strftime('%Y%m%d_%H%M%S')
    result_txt = os.path.join(save_dir, f"{ts}_{DATASET_TYPE}_{mask_quality}_{model_version}_results.txt")

    with open(result_txt, 'w', encoding='utf-8') as f:
        f.write(f"{DATASET_TYPE.upper()} - {mask_quality} Inference Results\n")
        f.write("=" * 40 + "\n")
        f.write(f"Dataset: {DATASET_TYPE.upper()}\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Mask Quality: {mask_quality}\n")
        f.write(f"Feature Extractor: {model_version}\n")
        f.write(f"Feature Bank Root: {INF_FEATURE_BANK_ROOT}\n")
        f.write(f"Save Dir: {save_dir}\n")
        f.write(f"Pseudo Mask Type: {PSEUDO_MASK_TYPE}\n")
        f.write("-" * 40 + "\n")

        for name, value in vars(cfg_module).items():
            if name.startswith('__') or inspect.ismodule(value) or inspect.isfunction(value) or inspect.isclass(value):
                continue
            f.write(f"{name}: {value}\n")

        f.write("-" * 40 + "\n")
        f.write(summary + "\n")
        f.write("\nPer-class IoU Table:\n")
        f.write("+----------------------+---------+\n")
        f.write("| Class                | IoU     |\n")
        f.write("+----------------------+---------+\n")
        for name, iou_val in zip(class_names, per_class):
            f.write(f"| {name:20s} | {iou_val:7.4f} |\n")
        f.write("+----------------------+---------+\n")
        f.write(f"| {'Mean(mIoU)':20s} | {metrics['mIoU']:7.4f} |\n")
        f.write("+----------------------+---------+\n")

    print(f"\nResults saved to {result_txt}")


if __name__ == '__main__':
    main()
