#!/usr/bin/env python3
"""Inference script: EntitySeg proposals + C-RADIOv4 features + feature bank retrieval."""
import os
import time
import json
from pathlib import Path

import numpy as np
import cv2
import torch
from tqdm import tqdm
from PIL import Image

from src.feature_extractor_c_radiov4 import CRADIOv4FeatureExtractor
from project_utils.config_logger import record_config
from src.inference import (
    init_mask_proposer, load_class_mapping, load_class_features,
    predict_instances, rasterize_instances,
)
from project_utils.seg_vis import overlay_segmentation, render_legend_panel
from project_utils.metrics import evaluate_dataset_miou
from configs.config import (
    INF_DATASET_ROOT, INF_FEATURE_BANK_ROOT, INF_SAVE_DIR, INF_VIS_LIMIT,
    INF_CRADIOV4_CKPT, INF_CRADIOV4_MODEL_NAME, INF_CRADIOV4_IMG_SIZE,
    INF_NMS_IOU, INF_NUM_CLASSES, INF_IGNORE_INDEX, INF_TOPK_NEIGH,
    INF_USE_FAISS, INF_FAISS_USE_GPU, INF_FAISS_DEVICE, INF_FAISS_DETERMINISTIC,
    DATASET_CLASSES, PSEUDO_MASK_TYPE,
    DATASET_TYPE,
    SEED,
)
from configs.config_helpers import get_model_version_str, get_mask_quality_str
from project_utils.seed_utils import set_global_seed


def main():
    print(f"=" * 80)
    print(f"Inference - {DATASET_TYPE.upper()}")
    print(f"=" * 80)

    set_global_seed(SEED)
    
    dataset_root = INF_DATASET_ROOT
    feature_bank_root = INF_FEATURE_BANK_ROOT

    # Load validation samples by dataset type
    if DATASET_TYPE == "coco":
        from configs.config import COCO_IMAGELEVEL_VAL_JSON, COCO_VAL_LIST
        if Path(COCO_IMAGELEVEL_VAL_JSON).is_file():
            with open(COCO_IMAGELEVEL_VAL_JSON, 'r') as f:
                val_data = json.load(f)
            sample_infos = []
            for img_info in val_data['images']:
                img_id = img_info['img_id']
                img_path = os.path.join(dataset_root, img_info['file_name'])
                # COCO GT: SegmentationClass/{split}/{filename}.png
                gt_filename = os.path.basename(img_info['file_name']).replace('.jpg', '.png')
                gt_split = 'train2014' if 'train' in img_info['file_name'] else 'val2014'
                gt_path = os.path.join(dataset_root, 'SegmentationClass', gt_split, gt_filename)
                sample_infos.append({'id': img_id, 'img': img_path, 'gt': gt_path})
        elif os.path.isfile(COCO_VAL_LIST):
            with open(COCO_VAL_LIST, 'r') as f:
                val_ids = [line.strip() for line in f if line.strip()]
            sample_infos = []
            for img_stem in val_ids:
                img_path = os.path.join(dataset_root, 'images', 'val2014', f'{img_stem}.jpg')
                gt_path = os.path.join(dataset_root, 'SegmentationClass', 'val2014', f'{img_stem}.png')
                sample_infos.append({'id': img_stem, 'img': img_path, 'gt': gt_path})
        else:
            # Fallback: scan SegmentationClass/val2014
            segclass_dir = os.path.join(dataset_root, 'SegmentationClass', 'val2014')
            sample_infos = []
            for gt_file in sorted(Path(segclass_dir).glob('*.png')):
                img_stem = gt_file.stem
                img_path = os.path.join(dataset_root, 'images', 'val2014', f'{img_stem}.jpg')
                sample_infos.append({'id': img_stem, 'img': img_path, 'gt': str(gt_file)})
        print(f"Loaded {len(sample_infos)} COCO val images")
        
    else:
        # VOC
        val_list = os.path.join(dataset_root, 'ImageSets', 'Segmentation', 'val.txt')
        gt_dir = os.path.join(dataset_root, 'SegmentationClassAug')
        img_dir = os.path.join(dataset_root, 'JPEGImages')
        
        with open(val_list, 'r') as f:
            ids = [line.strip() for line in f if line.strip()]
        
        sample_infos = []
        for img_id in ids:
            img_path = os.path.join(img_dir, f'{img_id}.jpg')
            gt_path = os.path.join(gt_dir, f'{img_id}.png')
            sample_infos.append({'id': img_id, 'img': img_path, 'gt': gt_path})

    # Load feature bank
    _, fg_names, bg_names = load_class_mapping(os.path.join(feature_bank_root, 'meta'))
    class_to_feats = load_class_features(
        os.path.join(feature_bank_root, 'features'), 
        list(set(fg_names + bg_names))
    )
    assert len(class_to_feats) > 0, "Empty feature bank. Please run build_demo.py first."

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
    # Normalize features
    all_feats = all_feats / (np.linalg.norm(all_feats, axis=1, keepdims=True) + 1e-8)
    all_feats = np.ascontiguousarray(all_feats.astype(np.float32))

    # Build FAISS index
    faiss_index = None
    if INF_USE_FAISS:
        import faiss
        try:
            if hasattr(faiss, "set_random_seed"):
                faiss.set_random_seed(SEED)
        except Exception:
            pass
        dim = all_feats.shape[1]
        n_samples = all_feats.shape[0]

        use_ivf = not INF_FAISS_DETERMINISTIC and n_samples > 10000
        
        if INF_FAISS_DETERMINISTIC:
            print(f"[FAISS] Deterministic mode: IndexFlatIP (exact search)")
        
        if INF_FAISS_USE_GPU and faiss.get_num_gpus() > 0:
            # GPU FAISS
            res = faiss.StandardGpuResources()
            res.setTempMemory(1536 * 1024 * 1024)
            
            if use_ivf:
                nlist = min(int(np.sqrt(n_samples)), 1024)
                print(f"[FAISS] GPU mode: IndexIVFFlat (nlist={nlist})")
                quantizer = faiss.IndexFlatIP(dim)
                cpu_index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
                cpu_index.train(all_feats)
                cpu_index.add(all_feats)
                cpu_index.nprobe = min(32, nlist)
                faiss_index = faiss.index_cpu_to_gpu(res, INF_FAISS_DEVICE, cpu_index)
            else:
                print(f"[FAISS] GPU mode: IndexFlatIP (deterministic)")
                cpu_index = faiss.IndexFlatIP(dim)
                cpu_index.add(all_feats)
                faiss_index = faiss.index_cpu_to_gpu(res, INF_FAISS_DEVICE, cpu_index)
        else:
            # CPU FAISS
            if use_ivf:
                nlist = min(int(np.sqrt(n_samples)), 1024)
                print(f"[FAISS] CPU mode: IndexIVFFlat (nlist={nlist})")
                quantizer = faiss.IndexFlatIP(dim)
                faiss_index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
                faiss_index.train(all_feats)
                faiss_index.add(all_feats)
                faiss_index.nprobe = min(32, nlist)
            else:
                print(f"[FAISS] CPU mode: IndexFlatIP (deterministic)")
                faiss_index = faiss.IndexFlatIP(dim)
                faiss_index.add(all_feats)

    # Load feature extractor
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dino = CRADIOv4FeatureExtractor(
        INF_CRADIOV4_CKPT, device, INF_CRADIOV4_MODEL_NAME, INF_CRADIOV4_IMG_SIZE
    )
    model_version = get_model_version_str()

    mask_proposer = init_mask_proposer()
    record_config(os.path.join(INF_FEATURE_BANK_ROOT, 'meta'), 'inference_demo.py')

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
            img_rgb, dino, mask_proposer,
            all_feats, gallery_labels, faiss_index,
            INF_NMS_IOU,
            topk_neigh=INF_TOPK_NEIGH,
        )

        pred_before = rasterize_instances(instances, h, w)

        if gt is not None:
            preds.append(pred_before)
            gts.append(gt)

        # Save visualization (first N images)
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
    summary, metrics, class_names = evaluate_dataset_miou(
        preds, gts, INF_NUM_CLASSES, INF_IGNORE_INDEX
    )
    print(summary)

    per_class = metrics['per_class_iou']
    class_names = ['background'] + DATASET_CLASSES
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

    with open(result_txt, 'w') as f:
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

        import configs.config as cfg_module
        import inspect
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
