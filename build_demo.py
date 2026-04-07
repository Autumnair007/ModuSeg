#!/usr/bin/env python3
"""Feature bank building script (pseudo masks from CorrCLIP or pre-generated)."""
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import json
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.Mask_generator_CorrCLIP import CorrCLIPMaskGenerator
from src.feature_extractor_c_radiov4 import CRADIOv4FeatureExtractor
from src.filter_stage import run_filter_stage
from project_utils.config_logger import record_config
from src.build_feature_bank import (
    create_dirs, save_class_mapping, FeatureBank
)
from src.feature_bank_dataset import FeatureBankDataset, collate_fn
from configs.config import (
    DATASET_ROOT, OUTPUT_ROOT, DATASET_TYPE, CLS_PATH,
    CRADIOV4_CKPT, CRADIOV4_MODEL_NAME, CRADIOV4_IMG_SIZE,
    DATASET_CLASSES, NUM_CLASSES,
    PSEUDO_MASK_TYPE, PSEUDO_MASK_ROOT, META_DIR,
    NUM_WORKERS,
    SEED,
)
from project_utils.seed_utils import set_global_seed, seed_worker


def main():
    print("=" * 80)
    print(f"Feature Bank Build - {DATASET_TYPE.upper()}")
    set_global_seed(SEED)
    record_config(META_DIR, 'build_demo.py')

    # Check for pre-generated pseudo masks
    has_pseudo_masks = False
    pseudo_root = Path(PSEUDO_MASK_ROOT)
    if pseudo_root.exists():
        if DATASET_TYPE == 'voc':
            sample_masks = list(pseudo_root.glob('*.png'))[:5]
        else:
            from configs.config import DATASET_TRAIN_SPLIT
            train_dir = pseudo_root / DATASET_TRAIN_SPLIT
            sample_masks = list(train_dir.glob('*.png'))[:5] if train_dir.exists() else []
        if sample_masks:
            has_pseudo_masks = True
            print(f"Found pre-generated pseudo masks: {PSEUDO_MASK_ROOT}")

    if has_pseudo_masks:
        print(f"Mode: pre-generated pseudo masks ({PSEUDO_MASK_TYPE})")
    else:
        print("Mode: CorrCLIP real-time pseudo mask generation")
    print("=" * 80)

    create_dirs()

    if has_pseudo_masks:
        print(f"\n[1/6] Using pre-generated pseudo masks, skip CorrCLIP init ({PSEUDO_MASK_TYPE})")
    else:
        print("\n[1/6] Will initialize CorrCLIP pseudo mask generator")

    # Load feature extractor
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("\n[2/6] Loading C-RADIOv4...")
    feature_extractor = CRADIOv4FeatureExtractor(
        model_path=CRADIOV4_CKPT,
        device=device,
        model_name=CRADIOV4_MODEL_NAME,
        img_size=CRADIOV4_IMG_SIZE,
    )

    # Save class mapping
    save_class_mapping(NUM_CLASSES, CLS_PATH)

    # Scan dataset
    print(f"\n[3/6] Scanning {DATASET_TYPE.upper()} dataset (train split)...")
    dataset_root = Path(DATASET_ROOT)
    
    if DATASET_TYPE == "coco":
        from configs.config import IMAGELEVEL_JSON_PATH
        from configs.config import COCO_TRAIN_LIST
        
        imagelevel_data = None
        if Path(IMAGELEVEL_JSON_PATH).is_file():
            with open(IMAGELEVEL_JSON_PATH, 'r') as f:
                imagelevel_data = json.load(f)
            
            image_paths = []
            for img_info in imagelevel_data['images']:
                img_path = dataset_root / img_info['file_name']
                if img_path.is_file():
                    image_paths.append(img_path)
            
            print(f"Loaded {len(image_paths)} train images from {IMAGELEVEL_JSON_PATH}")
        
        elif Path(COCO_TRAIN_LIST).is_file():
            with open(COCO_TRAIN_LIST, 'r') as f:
                train_ids = [line.strip() for line in f if line.strip()]
            image_paths = []
            for img_stem in train_ids:
                img_path = dataset_root / 'images' / 'train2014' / f'{img_stem}.jpg'
                if img_path.is_file():
                    image_paths.append(img_path)
            print(f"Loaded {len(image_paths)} train images from {COCO_TRAIN_LIST}")
        
        else:
            # Fallback: scan SegmentationClass/train2014
            segclass_dir = dataset_root / 'SegmentationClass' / 'train2014'
            image_paths = []
            for gt_file in sorted(segclass_dir.glob('*.png')):
                img_path = dataset_root / 'images' / 'train2014' / f'{gt_file.stem}.jpg'
                if img_path.is_file():
                    image_paths.append(img_path)
            print(f"Loaded {len(image_paths)} train images from SegmentationClass/train2014")
        
    else:
        # VOC: read from ImageSets/Segmentation/train.txt
        jpeg_dir = dataset_root / 'JPEGImages'
        all_image_paths = sorted(jpeg_dir.glob("*.jpg"))
        train_list_path = dataset_root / 'ImageSets' / 'Segmentation' / 'train.txt'
        
        train_ids = {line.strip() for line in train_list_path.read_text().splitlines() if line.strip()}
        image_paths = [p for p in all_image_paths if p.stem in train_ids]
        print(f"JPEGImages total: {len(all_image_paths)}; train: {len(image_paths)}")
    
    if len(image_paths) == 0:
        print("Error: no training images found.")
        return

    feature_bank = FeatureBank()

    corrclip_gen = None
    if not has_pseudo_masks:
        print("\n[4/6] Initializing CorrCLIP pseudo mask generator...")
        corrclip_gen = CorrCLIPMaskGenerator()

    # Initialize DataLoader
    print("\n[4/6] Initializing data loader...")

    if has_pseudo_masks:
        mode = 'pseudo'
        num_workers = NUM_WORKERS
    else:
        mode = 'corrclip'
        num_workers = 0  # CorrCLIP uses GPU, cannot use multiprocessing
    
    # Create Dataset
    dataset = FeatureBankDataset(
        image_paths=image_paths,
        mode=mode,
        pseudo_gen=corrclip_gen
    )

    g = torch.Generator()
    g.manual_seed(SEED)

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
        worker_init_fn=seed_worker,
        generator=g,
    )
    print(f"  Dataset size: {len(dataset)} images")

    # Extract features
    print("\n[5/6] Extracting features...")

    desc_text = f"Extract ({PSEUDO_MASK_TYPE})" if has_pseudo_masks else "Extract (CorrCLIP)"
    
    # Iterate DataLoader
    for batch in tqdm(dataloader, desc=desc_text):
        if not batch:
            continue
        for img_path_str, regions in batch:
            img_name = Path(img_path_str).stem
            clsid_to_feats = feature_extractor.extract_features_for_regions(
                img_path_str, regions, 0
            )

            for cls_id, feat_list in clsid_to_feats.items():
                if cls_id == 0:
                    cls_name = 'background'
                elif 1 <= cls_id <= len(DATASET_CLASSES):
                    cls_name = DATASET_CLASSES[cls_id - 1]
                else:
                    continue
                for feat_vec in feat_list:
                    if cls_id == 0:
                        feature_bank.add_background(feat_vec, img_name)
                    else:
                        feature_bank.add_feature(cls_name, feat_vec, img_name)

    # Save stats
    print("\n[6/6] Saving statistics...")
    feature_bank.save_stats()

    print("\nBuilding FAISS indices...")
    feature_bank.build_indices()

    print("\n[Post] Running filter stage...")
    run_filter_stage(drop_ratio=None)

    print("\n" + "=" * 80)
    print("Summary:")
    print("-" * 80)
    for cls_name, stat in sorted(feature_bank.stats.items()):
        print(f"{cls_name:20s}: {stat['count']:6d} samples")
    print("=" * 80)
    print(f"Feature bank saved to: {OUTPUT_ROOT}")
    print("Done!")


if __name__ == '__main__':
    main()
