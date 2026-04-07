"""Inference module: EntitySeg mask proposals + C-RADIOv4 features + feature bank retrieval."""
import os
import json
from typing import List, Dict, Any
from contextlib import nullcontext

import numpy as np
import torch

from src.feature_extractor_c_radiov4 import CRADIOv4FeatureExtractor
from project_utils.seed_utils import set_global_seed
from CropFormer.demo_mask2former.demo import get_entityseg
from configs.config import (
    INF_MASK_BACKEND,
    INF_ENTITYSEG_CFG, INF_ENTITYSEG_CKPT,
    INF_INSTANCE_SCORE_THR,
    MIN_MASK_AREA, DATASET_CLASSES, SEED,
)

# ============================================================================
# Mask proposer and feature bank loading
# ============================================================================

def init_mask_proposer():
    """Initialize EntitySeg mask proposer for inference."""
    set_global_seed(SEED, verbose=False)

    backend = (INF_MASK_BACKEND or "entityseg").lower()
    if backend == "entityseg":
        return get_entityseg(cfg_file=INF_ENTITYSEG_CFG, ckpt_path=INF_ENTITYSEG_CKPT)

    raise ValueError(f"Unknown INF_MASK_BACKEND: {INF_MASK_BACKEND}. Only 'entityseg' is supported.")


def load_class_mapping(meta_dir: str):
    """Load class mapping from meta directory."""
    path = os.path.join(meta_dir, 'class_mapping.json')
    assert os.path.isfile(path), f"Mapping not found: {path}. Please run build_demo.py first."
    meta = json.load(open(path))
    id_to_dirname = {int(k): v for k, v in meta['id_to_dirname'].items()}
    fg_class_names = [id_to_dirname[i] for i in sorted(id_to_dirname.keys()) 
                      if i != 0 and not id_to_dirname[i].startswith('background')]
    bg_class_names = [id_to_dirname[i] for i in sorted(id_to_dirname.keys()) 
                      if id_to_dirname[i] == 'background' or id_to_dirname[i].startswith('background_')]
    return id_to_dirname, fg_class_names, bg_class_names


def load_class_features(features_dir: str, class_names: List[str]):
    """Load class features (.npz and .npy formats)."""
    from pathlib import Path
    out = {}
    for name in class_names:
        cls_dir = Path(features_dir) / name
        if not cls_dir.is_dir():
            continue
        
        npz_files = list(cls_dir.glob('*.npz'))
        if npz_files:
            feats = []
            for npz_file in npz_files:
                try:
                    data = np.load(npz_file)
                    for key in sorted(data.keys()):
                        arr = data[key].astype(np.float32)
                        arr = arr.reshape(1, -1)
                        feats.append(arr)
                except Exception:
                    continue
            if len(feats) > 0:
                out[name] = np.vstack(feats)
            continue
        
        # Fallback to .npy
        feats = []
        for f in cls_dir.iterdir():
            if f.suffix == '.npy':
                try:
                    arr = np.load(f).astype(np.float32)
                    arr = arr.reshape(1, -1)
                    feats.append(arr)
                except Exception:
                    continue
        if len(feats) > 0:
            out[name] = np.vstack(feats)
    return out

# ============================================================================
# Core inference logic
# ============================================================================

def mask_iou(m1: np.ndarray, m2: np.ndarray) -> float:
    inter = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    return float(inter) / (float(union) + 1e-8)


def nms_masks(candidates: List[Dict[str, Any]], iou_thr: float) -> List[Dict[str, Any]]:
    """NMS filtering within a single class."""
    keep = []
    candidates = sorted(candidates, key=lambda x: x['score'], reverse=True)
    for c in candidates:
        if c['mask'].sum() < MIN_MASK_AREA:
            continue
        ok = True
        for k in keep:
            if mask_iou(c['mask'], k['mask']) > iou_thr:
                ok = False
                break
        if ok:
            keep.append(c)
    return keep


def nms_masks_per_class(candidates: List[Dict[str, Any]], iou_thr: float) -> List[Dict[str, Any]]:
    """Per-class NMS to avoid cross-class suppression."""
    from collections import defaultdict
    
    # Group by class
    cls_groups = defaultdict(list)
    for c in candidates:
        cls_groups[c['cls_name']].append(c)
    
    # Per-class NMS
    keep_all = []
    for cls_name, cls_candidates in cls_groups.items():
        keep = nms_masks(cls_candidates, iou_thr)
        keep_all.extend(keep)
    
    return keep_all


def predict_instances(img_rgb: np.ndarray,
                      dino: CRADIOv4FeatureExtractor,
                      mask_proposer,
                      all_feats: np.ndarray,
                      all_labels: List[str],
                      faiss_index,
                      nms_iou: float,
                      topk_neigh: int = 9) -> List[Dict[str, Any]]:
    """Full inference: EntitySeg proposals -> feature extraction -> retrieval -> per-class NMS."""
    # 1) Generate candidate masks
    img_bgr = img_rgb[..., ::-1].copy()
    amp_ctx = (
        torch.autocast("cuda", dtype=torch.float16)
        if torch.cuda.is_available()
        else nullcontext()
    )
    with torch.inference_mode(), amp_ctx:
        predictions = mask_proposer(img_bgr)

    pred_masks = predictions["instances"].pred_masks
    pred_scores = predictions["instances"].scores
    selected_indexes = (pred_scores >= INF_INSTANCE_SCORE_THR)
    selected_masks = pred_masks[selected_indexes]

    masks_original = []
    for m in selected_masks:
        m_np = m.detach().cpu().numpy().astype(np.uint8)
        if m_np.sum() < MIN_MASK_AREA:
            continue
        masks_original.append(m_np.astype(bool))

    if len(masks_original) == 0 or all_feats is None or all_feats.shape[0] == 0:
        return []

    # 2) Batch feature extraction
    feature_list = []
    if hasattr(dino, 'extract_features_batch'):
        try:
            feature_list = dino.extract_features_batch(img_rgb, masks_original)
        except Exception:
            feature_list = []
    
    if len(feature_list) != len(masks_original):
        feature_list = []
        for m in masks_original:
            try:
                feats = dino.extract_features(img_rgb, m)
                vec = feats.detach().cpu().numpy().reshape(-1)
                feature_list.append(vec)
            except Exception:
                feature_list.append(np.zeros(1, dtype=np.float32))
    
    # Stack and normalize features
    valid_features = []
    valid_masks_original = []
    for m_orig, vec in zip(masks_original, feature_list):
        if vec.shape[0] == all_feats.shape[1]:
            valid_features.append(vec)
            valid_masks_original.append(m_orig)
    
    if len(valid_features) == 0:
        return []
    
    # Normalize query features
    query_feats = np.vstack([f.reshape(1, -1) for f in valid_features]).astype(np.float32)
    query_feats = query_feats / (np.linalg.norm(query_feats, axis=1, keepdims=True) + 1e-8)
    query_feats = np.ascontiguousarray(query_feats)

    # 3) Batch top-K search
    k = min(topk_neigh, all_feats.shape[0])
    if k <= 0:
        return []
    
    if faiss_index is not None:
        # Batch FAISS search
        D, I = faiss_index.search(query_feats, k)
        sims_batch = D
        idxs_batch = I
    else:
        # Batch matrix multiply fallback
        sims_all = query_feats @ all_feats.T  # (N, M)
        idxs_batch = []
        sims_batch = []
        for i in range(sims_all.shape[0]):
            sims = sims_all[i]
            idxs = np.argpartition(sims, -k)[-k:]
            idxs = idxs[np.argsort(sims[idxs])[::-1]]
            idxs_batch.append(idxs)
            sims_batch.append(sims[idxs])
        sims_batch = np.array(sims_batch)
        idxs_batch = np.array(idxs_batch)

    # 4) Batch voting classification
    candidates = []
    for m, sims, idxs in zip(valid_masks_original, sims_batch, idxs_batch):
        votes, label_sums = {}, {}
        for rank, idx in enumerate(idxs):
            lab = all_labels[idx]
            lab = 'background' if (lab == 'background' or str(lab).startswith('background_')) else lab
            s = float(sims[rank])
            votes[lab] = votes.get(lab, 0) + 1
            label_sums[lab] = label_sums.get(lab, 0.0) + s

        best_lab, best_votes, best_sum = None, -1, -1e9
        for lab in votes:
            v = votes[lab]
            s = label_sums[lab]
            if v > best_votes or (v == best_votes and s > best_sum):
                best_votes, best_sum, best_lab = v, s, lab
        avg_score = label_sums[best_lab] / max(votes[best_lab], 1)
        candidates.append({'mask': m, 'cls_name': best_lab, 'score': float(avg_score)})

    # 5) Per-class NMS
    final_instances = nms_masks_per_class(candidates, nms_iou)
    
    # 6) Sort by score descending
    final_instances = sorted(final_instances, key=lambda x: x['score'], reverse=True)
    
    return final_instances


def rasterize_instances(instances: List[Dict[str, Any]], h: int, w: int) -> np.ndarray:
    """Convert instance list to semantic segmentation map."""
    label_map = {name: i+1 for i, name in enumerate(DATASET_CLASSES)}
    sem = np.zeros((h, w), dtype=np.uint8)
    for inst in instances:
        cls = inst['cls_name']
        if cls == 'background' or cls not in label_map:
            continue
        m = inst['mask']
        sem[m & (sem == 0)] = label_map[cls]
    return sem


__all__ = [
    'init_mask_proposer',
    'load_class_mapping',
    'load_class_features',
    'predict_instances',
    'rasterize_instances',
]
