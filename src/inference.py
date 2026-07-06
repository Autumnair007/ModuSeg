"""Inference module: EntitySeg mask proposals + C-RADIOv4 features + feature bank retrieval."""
import os
import json
from typing import List, Dict, Any
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from src.feature_bank_storage import load_feature_matrix_npz
from src.feature_extractor_c_radiov4 import CRADIOv4FeatureExtractor
from project_utils.seed_utils import set_global_seed
from CropFormer.demo_mask2former.demo import get_entityseg
from configs.config import (
    INF_USE_FAISS, INF_FAISS_USE_GPU, INF_FAISS_DEVICE, INF_FAISS_EXACT_SEARCH_THRESHOLD,
    DATASET_TYPE,
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
    assert os.path.isfile(path), f"Mapping not found: {path}. Please run run_build_feature_bank.py first."
    with open(path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    id_to_dirname = {int(k): v for k, v in meta['id_to_dirname'].items()}
    fg_class_names = [id_to_dirname[i] for i in sorted(id_to_dirname.keys()) 
                      if i != 0 and not id_to_dirname[i].startswith('background')]
    bg_class_names = [id_to_dirname[i] for i in sorted(id_to_dirname.keys()) 
                      if id_to_dirname[i] == 'background' or id_to_dirname[i].startswith('background_')]
    return id_to_dirname, fg_class_names, bg_class_names


def load_class_features(features_dir: str, class_names: List[str]):
    """Load matrix-format class features."""
    out = {}
    for name in class_names:
        cls_dir = Path(features_dir) / name
        if not cls_dir.is_dir():
            continue
        
        npz_files = sorted(cls_dir.glob('*.npz'))
        if not npz_files:
            continue
        if len(npz_files) != 1:
            raise RuntimeError(f"Expected exactly one v2 feature file under {cls_dir}, got {len(npz_files)}")

        features, _sample_ids = load_feature_matrix_npz(npz_files[0])
        if features.shape[0] > 0:
            out[name] = features
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


def reduce_ade20k_gt_for_eval(gt_seg: np.ndarray) -> np.ndarray:
    """Convert ADE20K GT from 0/1..150 to 255/0..149."""
    gt = np.asarray(gt_seg).astype(np.int64)
    out = np.full(gt.shape, 255, dtype=np.int64)
    valid = (gt >= 1) & (gt <= 150)
    out[valid] = gt[valid] - 1
    return out


def reduce_ade20k_pred_for_eval(pred_seg: np.ndarray) -> np.ndarray:
    """Convert internal ADE20K prediction labels from 0..150 to 0..149."""
    pred = np.asarray(pred_seg).astype(np.int64)
    return np.clip(pred - 1, 0, 149)


_CITYSCAPES_ID_TO_TRAIN_ID = np.full(256, 255, dtype=np.uint8)
for _label_id, _train_id in {
    7: 0,
    8: 1,
    11: 2,
    12: 3,
    13: 4,
    17: 5,
    19: 6,
    20: 7,
    21: 8,
    22: 9,
    23: 10,
    24: 11,
    25: 12,
    26: 13,
    27: 14,
    28: 15,
    31: 16,
    32: 17,
    33: 18,
}.items():
    _CITYSCAPES_ID_TO_TRAIN_ID[_label_id] = _train_id


def cityscapes_image_to_gt_path(img_path: Path, dataset_root: Path) -> Path:
    """Infer the Cityscapes labelIds path from a leftImg8bit image path."""
    img_path = Path(img_path)
    split = img_path.parts[img_path.parts.index("leftImg8bit") + 1]
    city = img_path.parent.name
    base = img_path.stem.replace("_leftImg8bit", "")
    return dataset_root / "gtFine" / split / city / f"{base}_gtFine_labelIds.png"


def reduce_cityscapes_gt_for_eval(gt_seg: np.ndarray) -> np.ndarray:
    """Convert Cityscapes labelIds GT to trainIds 0..18 with 255 ignored."""
    gt = np.asarray(gt_seg)
    out = np.full(gt.shape, 255, dtype=np.int64)
    valid = (gt >= 0) & (gt < len(_CITYSCAPES_ID_TO_TRAIN_ID))
    out[valid] = _CITYSCAPES_ID_TO_TRAIN_ID[gt[valid].astype(np.int64)].astype(np.int64)
    return out


def reduce_cityscapes_pred_for_eval(pred_seg: np.ndarray) -> np.ndarray:
    """Convert internal CityScapes labels 1..19 to trainIds 0..18."""
    pred = np.asarray(pred_seg).astype(np.int64)
    out = np.full(pred.shape, 255, dtype=np.int64)
    valid = (pred >= 1) & (pred <= 19)
    out[valid] = pred[valid] - 1
    return out


def load_eval_samples(dataset_root: str) -> List[Dict[str, str]]:
    """Load validation samples for the configured dataset."""
    if DATASET_TYPE == "coco":
        from configs.config import COCO_IMAGELEVEL_VAL_JSON, COCO_VAL_LIST

        if Path(COCO_IMAGELEVEL_VAL_JSON).is_file():
            with open(COCO_IMAGELEVEL_VAL_JSON, 'r', encoding='utf-8') as f:
                val_data = json.load(f)
            sample_infos = []
            for img_info in val_data['images']:
                img_id = img_info['img_id']
                img_path = os.path.join(dataset_root, img_info['file_name'])
                gt_filename = os.path.basename(img_info['file_name']).replace('.jpg', '.png')
                gt_split = 'train2014' if 'train' in img_info['file_name'] else 'val2014'
                gt_path = os.path.join(dataset_root, 'SegmentationClass', gt_split, gt_filename)
                sample_infos.append({'id': img_id, 'img': img_path, 'gt': gt_path})
        elif os.path.isfile(COCO_VAL_LIST):
            with open(COCO_VAL_LIST, 'r', encoding='utf-8') as f:
                val_ids = [line.strip() for line in f if line.strip()]
            sample_infos = []
            for img_stem in val_ids:
                img_path = os.path.join(dataset_root, 'images', 'val2014', f'{img_stem}.jpg')
                gt_path = os.path.join(dataset_root, 'SegmentationClass', 'val2014', f'{img_stem}.png')
                sample_infos.append({'id': img_stem, 'img': img_path, 'gt': gt_path})
        else:
            segclass_dir = os.path.join(dataset_root, 'SegmentationClass', 'val2014')
            sample_infos = []
            for gt_file in sorted(Path(segclass_dir).glob('*.png')):
                img_stem = gt_file.stem
                img_path = os.path.join(dataset_root, 'images', 'val2014', f'{img_stem}.jpg')
                sample_infos.append({'id': img_stem, 'img': img_path, 'gt': str(gt_file)})
        print(f"Loaded {len(sample_infos)} COCO val images")
        return sample_infos

    if DATASET_TYPE == "ade20k":
        from configs.config import ADE20K_IMAGELEVEL_VAL_JSON

        sample_infos = []
        if Path(ADE20K_IMAGELEVEL_VAL_JSON).is_file():
            with open(ADE20K_IMAGELEVEL_VAL_JSON, 'r', encoding='utf-8') as f:
                val_data = json.load(f)
            for img_info in val_data['images']:
                img_id = img_info['img_id']
                img_path = os.path.join(dataset_root, img_info['file_name'])
                gt_path = os.path.join(
                    dataset_root,
                    'annotations',
                    'validation',
                    f"{Path(img_info['file_name']).stem}.png",
                )
                sample_infos.append({'id': img_id, 'img': img_path, 'gt': gt_path})
        else:
            image_dir = Path(dataset_root) / 'images' / 'validation'
            for img_file in sorted(image_dir.glob('*.jpg')):
                gt_path = Path(dataset_root) / 'annotations' / 'validation' / f'{img_file.stem}.png'
                sample_infos.append({'id': img_file.stem, 'img': str(img_file), 'gt': str(gt_path)})
        print(f"Loaded {len(sample_infos)} ADE20K val images")
        return sample_infos

    if DATASET_TYPE == "cityscapes":
        from configs.config import CITYSCAPES_IMAGELEVEL_VAL_JSON

        sample_infos = []
        if Path(CITYSCAPES_IMAGELEVEL_VAL_JSON).is_file():
            with open(CITYSCAPES_IMAGELEVEL_VAL_JSON, 'r', encoding='utf-8') as f:
                val_data = json.load(f)
            for img_info in val_data['images']:
                img_id = img_info['img_id']
                img_path = os.path.join(dataset_root, img_info['file_name'])
                gt_path = os.path.join(dataset_root, img_info['gt_file_name'])
                sample_infos.append({'id': img_id, 'img': img_path, 'gt': gt_path})
        else:
            image_dir = Path(dataset_root) / 'leftImg8bit' / 'val'
            for img_file in sorted(image_dir.glob('*/*_leftImg8bit.png')):
                gt_path = cityscapes_image_to_gt_path(img_file, Path(dataset_root))
                sample_infos.append({'id': img_file.stem, 'img': str(img_file), 'gt': str(gt_path)})
        print(f"Loaded {len(sample_infos)} CityScapes val images")
        return sample_infos

    val_list = os.path.join(dataset_root, 'ImageSets', 'Segmentation', 'val.txt')
    gt_dir = os.path.join(dataset_root, 'SegmentationClassAug')
    img_dir = os.path.join(dataset_root, 'JPEGImages')

    with open(val_list, 'r', encoding='utf-8') as f:
        ids = [line.strip() for line in f if line.strip()]

    sample_infos = []
    for img_id in ids:
        img_path = os.path.join(img_dir, f'{img_id}.jpg')
        gt_path = os.path.join(gt_dir, f'{img_id}.png')
        sample_infos.append({'id': img_id, 'img': img_path, 'gt': gt_path})
    return sample_infos


def build_faiss_index(all_feats: np.ndarray):
    """Build the optional FAISS search index for gallery features."""
    if not INF_USE_FAISS:
        return None

    import faiss

    try:
        if hasattr(faiss, "set_random_seed"):
            faiss.set_random_seed(SEED)
        elif hasattr(faiss, "rand") and hasattr(faiss.rand, "seed"):
            faiss.rand.seed(SEED)
        elif hasattr(faiss, "cvar") and hasattr(faiss.cvar, "rand_seed"):
            faiss.cvar.rand_seed = int(SEED)
    except Exception:
        pass

    dim = all_feats.shape[1]
    n_samples = all_feats.shape[0]
    use_ivf = n_samples > INF_FAISS_EXACT_SEARCH_THRESHOLD

    if n_samples <= INF_FAISS_EXACT_SEARCH_THRESHOLD:
        print(f"[FAISS] {n_samples} features <= {INF_FAISS_EXACT_SEARCH_THRESHOLD}: IndexFlatIP (exact search)")

    if INF_FAISS_USE_GPU and faiss.get_num_gpus() > 0:
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
            return faiss.index_cpu_to_gpu(res, INF_FAISS_DEVICE, cpu_index)

        print("[FAISS] GPU mode: IndexFlatIP (exact search)")
        cpu_index = faiss.IndexFlatIP(dim)
        cpu_index.add(all_feats)
        return faiss.index_cpu_to_gpu(res, INF_FAISS_DEVICE, cpu_index)

    if use_ivf:
        nlist = min(int(np.sqrt(n_samples)), 1024)
        print(f"[FAISS] CPU mode: IndexIVFFlat (nlist={nlist})")
        faiss_index = faiss.IndexIVFFlat(
            faiss.IndexFlatIP(dim),
            dim,
            nlist,
            faiss.METRIC_INNER_PRODUCT,
        )
        faiss_index.train(all_feats)
        faiss_index.add(all_feats)
        faiss_index.nprobe = min(32, nlist)
        return faiss_index

    print("[FAISS] CPU mode: IndexFlatIP (exact search)")
    faiss_index = faiss.IndexFlatIP(dim)
    faiss_index.add(all_feats)
    return faiss_index


__all__ = [
    'build_faiss_index',
    'cityscapes_image_to_gt_path',
    'init_mask_proposer',
    'load_class_mapping',
    'load_class_features',
    'load_eval_samples',
    'predict_instances',
    'rasterize_instances',
    'reduce_ade20k_gt_for_eval',
    'reduce_ade20k_pred_for_eval',
    'reduce_cityscapes_gt_for_eval',
    'reduce_cityscapes_pred_for_eval',
]
