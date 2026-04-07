"""Project configuration for FeatureBank-CorrCLIP pipeline."""
from __future__ import annotations
import os
import torch

from configs.config_helpers import (
    env_get_str, env_get_int, env_get_float, env_get_bool
)


class ConfigDefaults:
    """Default configuration values (single source of truth)."""
    # Base
    SEED = 42
    DATASET_TYPE = "voc"  # "voc" | "coco"
    FEATURE_BACKBONE = "c-radiov4"
    NUM_WORKERS = 8

    # Build stage
    MIN_MASK_AREA = 300
    BUILD_MASK_EROSION_KERNEL_SIZE = 3
    BUILD_MASK_EROSION_ITERATIONS = 20
    FILTER_FOREGROUND_DROP_RATIO = 0.25
    FILTER_BACKGROUND_DROP_RATIO = 0.0

    # Inference stage
    INF_VIS_LIMIT = 20
    INF_MASK_BACKEND = "entityseg"
    INF_NMS_IOU = 0.4
    INF_TOPK_NEIGH = 25
    INF_FAISS_USE_GPU = True
    INF_FAISS_DEVICE = 0
    INF_FAISS_DETERMINISTIC = True
    INF_ENTITYSEG_SCORE_THR = 0.6

    # Model
    CRADIOV4_CKPT = "C_RADIOv4_SO400M/model.safetensors"
    CRADIOV4_MODEL_NAME = "c-radio_v4-so400m"
    CRADIOV4_IMG_SIZE = 512



_D = ConfigDefaults

# ==========================================================================
# 1. Base
# ==========================================================================

SEED = env_get_int("OVERRIDE_SEED", _D.SEED)
DATASET_TYPE = env_get_str("OVERRIDE_DATASET_TYPE", _D.DATASET_TYPE)
FEATURE_BACKBONE = env_get_str("OVERRIDE_FEATURE_BACKBONE", _D.FEATURE_BACKBONE)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_WORKERS = env_get_int("OVERRIDE_NUM_WORKERS", _D.NUM_WORKERS)

# ==========================================================================
# 2. Build stage
# ==========================================================================

PSEUDO_MASK_TYPE = "corrclip"
MIN_MASK_AREA = env_get_int("OVERRIDE_MIN_MASK_AREA", _D.MIN_MASK_AREA)

BUILD_MASK_EROSION_KERNEL_SIZE = env_get_int("OVERRIDE_BUILD_MASK_EROSION_KERNEL_SIZE", _D.BUILD_MASK_EROSION_KERNEL_SIZE)
BUILD_MASK_EROSION_ITERATIONS = env_get_int("OVERRIDE_BUILD_MASK_EROSION_ITERATIONS", _D.BUILD_MASK_EROSION_ITERATIONS)

FILTER_FOREGROUND_DROP_RATIO = env_get_float("OVERRIDE_FILTER_FOREGROUND_DROP_RATIO", _D.FILTER_FOREGROUND_DROP_RATIO)
FILTER_BACKGROUND_DROP_RATIO = env_get_float("OVERRIDE_FILTER_BACKGROUND_DROP_RATIO", _D.FILTER_BACKGROUND_DROP_RATIO)
FILTER_BACKGROUND_PREFIX = 'background'

# ==========================================================================
# 3. Inference stage
# ==========================================================================

INF_SAVE_DIR = 'inference_vis'
INF_VIS_LIMIT = env_get_int("OVERRIDE_INF_VIS_LIMIT", _D.INF_VIS_LIMIT)
INF_MASK_BACKEND = env_get_str("OVERRIDE_INF_MASK_BACKEND", _D.INF_MASK_BACKEND).lower()
INF_INSTANCE_SCORE_THR = env_get_float("OVERRIDE_INF_INSTANCE_SCORE_THR", _D.INF_ENTITYSEG_SCORE_THR)

INF_NMS_IOU = env_get_float("OVERRIDE_INF_NMS_IOU", _D.INF_NMS_IOU)
INF_IGNORE_INDEX = 255
INF_TOPK_NEIGH = env_get_int("OVERRIDE_INF_TOPK_NEIGH", _D.INF_TOPK_NEIGH)

INF_USE_FAISS = True
INF_FAISS_USE_GPU = env_get_bool("OVERRIDE_INF_FAISS_USE_GPU", _D.INF_FAISS_USE_GPU)
INF_FAISS_DEVICE = env_get_int("OVERRIDE_INF_FAISS_DEVICE", _D.INF_FAISS_DEVICE)
INF_FAISS_DETERMINISTIC = env_get_bool("OVERRIDE_INF_FAISS_DETERMINISTIC", _D.INF_FAISS_DETERMINISTIC)

INF_ENTITYSEG_CFG = "mask2former_hornet_3x.yaml"
INF_ENTITYSEG_CKPT = os.path.join('pretrain_model', 'Mask2Former_hornet_3x_576d0b.pth')
INF_ENTITYSEG_SCORE_THR = env_get_float("OVERRIDE_INF_ENTITYSEG_SCORE_THR", _D.INF_ENTITYSEG_SCORE_THR)

# ==========================================================================
# 4. Model
# ==========================================================================

CRADIOV4_CKPT = env_get_str("OVERRIDE_CRADIOV4_CKPT", _D.CRADIOV4_CKPT)
CRADIOV4_MODEL_NAME = env_get_str("OVERRIDE_CRADIOV4_MODEL_NAME", _D.CRADIOV4_MODEL_NAME)
CRADIOV4_IMG_SIZE = env_get_int("OVERRIDE_CRADIOV4_IMG_SIZE", _D.CRADIOV4_IMG_SIZE)

INF_CRADIOV4_CKPT = CRADIOV4_CKPT
INF_CRADIOV4_MODEL_NAME = CRADIOV4_MODEL_NAME
INF_CRADIOV4_IMG_SIZE = CRADIOV4_IMG_SIZE

# CorrCLIP
CORRCLIP_CLIP_TYPE = 'metaclip_fullcc'
CORRCLIP_MODEL_TYPE = 'ViT-B-16-quickgelu'  
CORRCLIP_DINO_TYPE = 'dino_vitb8'
CORRCLIP_MASK_BACKEND = 'entityseg'

# ==========================================================================
# 5. Paths (auto-configured by dataset type)
# ==========================================================================
from configs.config_helpers import load_classes

CLS_VOC21_PATH = os.path.join('configs', 'cls_voc21.txt')
CLS_COCO_PATH = os.path.join('configs', 'cls_coco_object.txt')

if DATASET_TYPE == "coco":
    COCO_ROOT = 'data/COCO2014'
    COCO_TRAIN_SPLIT = 'train2014'
    COCO_VAL_SPLIT = 'val2014'
    COCO_CLASSES = load_classes(CLS_COCO_PATH)
    COCO_NUM_CLASSES = len(COCO_CLASSES) + 1
    COCO_IMAGELEVEL_TRAIN_JSON = os.path.join(COCO_ROOT, 'annotations', 'train_imagelevel.json')
    COCO_IMAGELEVEL_VAL_JSON = os.path.join(COCO_ROOT, 'annotations', 'val_imagelevel.json')
    COCO_TRAIN_LIST = os.path.join(COCO_ROOT, 'ImageSets', 'coco_train.txt')
    COCO_VAL_LIST = os.path.join(COCO_ROOT, 'ImageSets', 'coco_val.txt')
    COCO_OUTPUT_ROOT = os.environ.get("OVERRIDE_COCO_OUTPUT_ROOT", "feature_bank_coco")

    DATASET_ROOT = COCO_ROOT
    DATASET_TRAIN_SPLIT = COCO_TRAIN_SPLIT
    DATASET_VAL_SPLIT = COCO_VAL_SPLIT
    CLS_PATH = CLS_COCO_PATH
    DATASET_CLASSES = COCO_CLASSES
    NUM_CLASSES = COCO_NUM_CLASSES
    IMAGELEVEL_JSON_PATH = COCO_IMAGELEVEL_TRAIN_JSON
    OUTPUT_ROOT = env_get_str("OVERRIDE_OUTPUT_ROOT", COCO_OUTPUT_ROOT)
else:
    DATASET_ROOT = "data/VOC2012"
    DATASET_TRAIN_SPLIT = "train"
    DATASET_VAL_SPLIT = "val"
    CLS_PATH = CLS_VOC21_PATH
    DATASET_CLASSES = load_classes(CLS_VOC21_PATH)
    NUM_CLASSES = 21
    IMAGELEVEL_JSON_PATH = os.path.join(DATASET_ROOT, 'ImageSets', 'ImageLevel', 'train_imagelevel.json')
    OUTPUT_ROOT = env_get_str("OVERRIDE_OUTPUT_ROOT", "feature_bank")

# Image-level label JSON key names
IMAGELEVEL_IMG_ID_KEY = "img_id"
IMAGELEVEL_LABELS_KEY = "labels"

# Derived paths
FEATURES_DIR = os.path.join(OUTPUT_ROOT, "features")
INDEX_DIR = os.path.join(OUTPUT_ROOT, "index")
META_DIR = os.path.join(OUTPUT_ROOT, "meta")
CORRCLIP_NAME_FILE = CLS_PATH
CORRCLIP_INSTANCE_MASK_ROOT = os.path.join(DATASET_ROOT, 'instance_masks')

PSEUDO_MASK_ROOT = os.path.join(DATASET_ROOT, 'pseudo', 'corrclip')

INF_DATASET_ROOT = DATASET_ROOT
INF_FEATURE_BANK_ROOT = env_get_str("OVERRIDE_INF_FEATURE_BANK_ROOT", OUTPUT_ROOT)
INF_NUM_CLASSES = NUM_CLASSES

print(f"[Config] DATASET_TYPE={DATASET_TYPE}, BACKBONE={FEATURE_BACKBONE}, OUTPUT={OUTPUT_ROOT}, PSEUDO={PSEUDO_MASK_TYPE}")