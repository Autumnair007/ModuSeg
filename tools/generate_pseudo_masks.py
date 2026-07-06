"""
Generate CorrCLIP pseudo masks for weakly-supervised semantic segmentation.

Uses CorrCLIP (ViT-B-16 + DINO) with EntitySeg as the instance mask backend.
Image-level label filtering is always enabled.

Output layout:
    VOC:  data/VOC2012/pseudo/corrclip/{img_id}.png
    COCO: data/COCO2014/pseudo/corrclip/{split}/{img_id}.png
    ADE:  data/ADEChallengeData2016/pseudo/corrclip/{split}/{img_id}.png
    Cityscapes: data/CityScapes/pseudo/corrclip/{split}/{img_id}.png

Usage:
    python tools/generate_pseudo_masks.py --dataset voc
    python tools/generate_pseudo_masks.py --dataset coco --split train2014
    python tools/generate_pseudo_masks.py --dataset ade20k --split training
    python tools/generate_pseudo_masks.py --dataset cityscapes --split train
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import json
from PIL import Image

import torch
from torchvision import transforms

from corrclip_segmentor import CorrCLIPSegmentation
from configs.config import CORRCLIP_CLIP_TYPE, CORRCLIP_MODEL_TYPE, CORRCLIP_DINO_TYPE, CORRCLIP_MASK_BACKEND
from project_utils.imagelevel_utils import load_imagelevel_labels

# ---------------------------------------------------------------------------
# Configurable paths (edit these if your layout differs)
# ---------------------------------------------------------------------------
VOC_ROOT = Path('data/VOC2012')
COCO_ROOT = Path('data/COCO2014')
ADE20K_ROOT = Path('data/ADEChallengeData2016')
CITYSCAPES_ROOT = Path('data/CityScapes')

# Pseudo mask output directory name under {dataset_root}/pseudo/
PSEUDO_SUBDIR = 'corrclip'

VOC_CLASS_CONFIG = 'configs/cls_voc21.txt'
COCO_CLASS_CONFIG = 'configs/cls_coco_object.txt'
ADE20K_CLASS_CONFIG = 'configs/cls_ade20k.txt'
CITYSCAPES_CLASS_CONFIG = 'configs/cls_city_scapes.txt'


class PseudoMaskGenerator:
    """CorrCLIP pseudo mask generator (ViT-B + EntitySeg, image-level filter always on)."""

    def __init__(self, dataset_type='voc'):
        self.dataset_type = dataset_type.lower()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Dataset-specific paths
        if self.dataset_type == 'voc':
            self.dataset_root = VOC_ROOT
            self.name_file = VOC_CLASS_CONFIG
            self.imagelevel_json = self.dataset_root / 'ImageSets' / 'ImageLevel' / 'train_imagelevel.json'
        elif self.dataset_type == 'coco':
            self.dataset_root = COCO_ROOT
            self.name_file = COCO_CLASS_CONFIG
            self.imagelevel_json = self.dataset_root / 'annotations' / 'train_imagelevel.json'
        elif self.dataset_type == 'ade20k':
            self.dataset_root = ADE20K_ROOT
            self.name_file = ADE20K_CLASS_CONFIG
            self.imagelevel_json = self.dataset_root / 'ImageSets' / 'ImageLevel' / 'train_imagelevel.json'
        elif self.dataset_type == 'cityscapes':
            self.dataset_root = CITYSCAPES_ROOT
            self.name_file = CITYSCAPES_CLASS_CONFIG
            self.imagelevel_json = self.dataset_root / 'ImageSets' / 'ImageLevel' / 'train_imagelevel.json'
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")

        # Output directory: {dataset_root}/pseudo/corrclip/
        self.output_dir = self.dataset_root / 'pseudo' / PSEUDO_SUBDIR

        print("=" * 80)
        print(f"CorrCLIP Pseudo Mask Generator")
        print(f"  Dataset:      {self.dataset_type.upper()}")
        print(f"  CLIP model:   {CORRCLIP_MODEL_TYPE}")
        print(f"  DINO type:    {CORRCLIP_DINO_TYPE}")
        print(f"  Mask backend: {CORRCLIP_MASK_BACKEND}")
        print(f"  Output dir:   {self.output_dir}")
        print(f"  Image-level filter: ON (fixed)")
        print("=" * 80)

        # Initialize CorrCLIP model (model type and backend are fixed via config)
        self.model = CorrCLIPSegmentation(
            clip_type=CORRCLIP_CLIP_TYPE,
            model_type=CORRCLIP_MODEL_TYPE,
            dino_type=CORRCLIP_DINO_TYPE,
            name_path=self.name_file,
            device=self.device,
            mask_generator=CORRCLIP_MASK_BACKEND,
            instance_mask_path=None,
            imagelevel_json_path=str(self.imagelevel_json),
            dataset_type=self.dataset_type,
        )

        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                [0.48145466, 0.4578275, 0.40821073],
                [0.26862954, 0.26130258, 0.27577711],
            ),
        ])

    def get_image_list(self, split='train'):
        """Get image paths from image-level label JSON."""
        if self.dataset_type == 'voc':
            imagelevel_file = self.dataset_root / 'ImageSets' / 'ImageLevel' / f'{split}_imagelevel.json'
        elif self.dataset_type == 'coco':
            split_name = 'train' if 'train' in split else 'val'
            imagelevel_file = self.dataset_root / 'annotations' / f'{split_name}_imagelevel.json'
        elif self.dataset_type == 'ade20k':
            split_dir = self._normalize_ade20k_split(split)
            split_name = 'train' if split_dir == 'training' else 'val'
            imagelevel_file = self.dataset_root / 'ImageSets' / 'ImageLevel' / f'{split_name}_imagelevel.json'
        elif self.dataset_type == 'cityscapes':
            split_dir = self._normalize_cityscapes_split(split)
            imagelevel_file = self.dataset_root / 'ImageSets' / 'ImageLevel' / f'{split_dir}_imagelevel.json'
        else:
            raise ValueError(f"Unsupported dataset type: {self.dataset_type}")

        if not imagelevel_file.exists():
            raise FileNotFoundError(f"Image-level label file not found: {imagelevel_file}")

        with open(imagelevel_file, 'r') as f:
            imagelevel_data = json.load(f)

        img_paths = []
        for img_info in imagelevel_data['images']:
            img_path = self.dataset_root / img_info['file_name']
            if not img_path.is_file():
                raise FileNotFoundError(f"Image not found: {img_path}")
            img_paths.append(img_path)

        self.imagelevel_json = imagelevel_file
        print(f"Loaded {len(img_paths)} images from {imagelevel_file}")
        return img_paths

    def _sync_imagelevel_filter(self):
        """Update CorrCLIP image-level filtering when a non-default split is used."""
        if not getattr(self.model, "use_imagelevel_filter", False):
            return
        json_path = str(self.imagelevel_json)
        if getattr(self.model, "imagelevel_json_path", None) == json_path:
            return
        self.model.imagelevel_json_path = json_path
        self.model.imagelevel_labels = load_imagelevel_labels(json_path=json_path)

    @torch.inference_mode()
    def generate_mask(self, img_path):
        """Generate pseudo mask for a single image."""
        img_path = str(img_path)
        img = Image.open(img_path).convert('RGB')
        img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)

        seg_pred = self.model.predict([img_tensor, img_path], data_samples=None)

        if isinstance(seg_pred, torch.Tensor):
            seg_np = seg_pred.squeeze().detach().cpu().numpy().astype(np.uint8)
        else:
            seg_np = np.array(seg_pred).astype(np.uint8)
        if self.dataset_type in ('ade20k', 'cityscapes'):
            seg_np = (seg_np.astype(np.uint16) + 1).astype(np.uint8)
        return seg_np

    def save_mask(self, mask, img_path, split='train'):
        """Save mask as PNG."""
        img_path = Path(img_path)
        filename = f'{img_path.stem}.png'

        if self.dataset_type == 'coco':
            output_path = self.output_dir / split / filename
        elif self.dataset_type == 'ade20k':
            output_path = self.output_dir / self._normalize_ade20k_split(split) / filename
        elif self.dataset_type == 'cityscapes':
            output_path = self.output_dir / self._normalize_cityscapes_split(split) / filename
        else:
            output_path = self.output_dir / filename

        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(mask).save(output_path)
        return output_path

    def generate_all(self, split='train', resume=True):
        """Batch-generate pseudo masks for all images in a split."""
        img_paths = self.get_image_list(split)
        self._sync_imagelevel_filter()
        print(f"\nGenerating pseudo masks for {len(img_paths)} images (split={split})...")

        success_count = 0
        error_count = 0

        for img_path in tqdm(img_paths, desc="Generating pseudo masks"):
            try:
                # Resume: skip if output already exists
                if resume:
                    if self.dataset_type == 'coco':
                        output_path = self.output_dir / split / f'{img_path.stem}.png'
                    elif self.dataset_type == 'ade20k':
                        output_path = self.output_dir / self._normalize_ade20k_split(split) / f'{img_path.stem}.png'
                    elif self.dataset_type == 'cityscapes':
                        output_path = self.output_dir / self._normalize_cityscapes_split(split) / f'{img_path.stem}.png'
                    else:
                        output_path = self.output_dir / f'{img_path.stem}.png'
                    if output_path.exists():
                        success_count += 1
                        continue

                mask = self.generate_mask(img_path)
                self.save_mask(mask, img_path, split)
                success_count += 1

            except Exception as e:
                print(f"\nFailed: {img_path.name}, error: {e}")
                error_count += 1

        print("\n" + "=" * 80)
        print(f"Pseudo mask generation complete.")
        print(f"  Success: {success_count}")
        print(f"  Failed:  {error_count}")
        print(f"  Output:  {self.output_dir}")
        print("=" * 80)

    @staticmethod
    def _normalize_ade20k_split(split: str) -> str:
        split = str(split).lower()
        if split in {"train", "training"}:
            return "training"
        if split in {"val", "validation"}:
            return "validation"
        return split

    @staticmethod
    def _normalize_cityscapes_split(split: str) -> str:
        split = str(split).lower()
        if split in {"train", "training"}:
            return "train"
        if split in {"val", "validation"}:
            return "val"
        return split


def main():
    parser = argparse.ArgumentParser(description='Generate CorrCLIP pseudo masks')
    parser.add_argument('--dataset', type=str, default='voc', choices=['voc', 'coco', 'ade20k', 'cityscapes'],
                        help='Dataset type (voc, coco, ade20k, or cityscapes)')
    parser.add_argument('--split', type=str, default='train',
                        help='Split (VOC: train/val, COCO: train2014/val2014, ADE20K: training/validation, Cityscapes: train/val)')
    parser.add_argument('--no-resume', action='store_true',
                        help='Disable resume, regenerate all masks')

    args = parser.parse_args()

    generator = PseudoMaskGenerator(dataset_type=args.dataset)
    generator.generate_all(split=args.split, resume=not args.no_resume)


if __name__ == '__main__':
    main()
