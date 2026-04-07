import os
from typing import Optional

import numpy as np
from PIL import Image

import torch
from torchvision import transforms

from corrclip_segmentor import CorrCLIPSegmentation
from configs.config import (
    CORRCLIP_CLIP_TYPE,
    CORRCLIP_MODEL_TYPE,
    CORRCLIP_DINO_TYPE,
    CORRCLIP_NAME_FILE,
    CORRCLIP_MASK_BACKEND,
    CORRCLIP_INSTANCE_MASK_ROOT,
    DATASET_TYPE,
    IMAGELEVEL_JSON_PATH,
)


class CorrCLIPMaskGenerator:
    """CorrCLIP semantic segmentation wrapper.
    Given an image path, returns a full-image semantic segmentation map (HxW int).
    """

    def __init__(self) -> None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if CORRCLIP_MASK_BACKEND in (None, "precomputed"):
            mask_generator = None
            instance_mask_path = CORRCLIP_INSTANCE_MASK_ROOT
        else:
            mask_generator = CORRCLIP_MASK_BACKEND
            instance_mask_path = None

        print("=" * 60)
        print(f"[CorrCLIPMaskGenerator] Initializing:")
        print(f"  clip_type     = {CORRCLIP_CLIP_TYPE}")
        print(f"  model_type    = {CORRCLIP_MODEL_TYPE}")
        print(f"  dino_type     = {CORRCLIP_DINO_TYPE}")
        print(f"  mask_backend  = {CORRCLIP_MASK_BACKEND}")
        print("=" * 60)

        self.model = CorrCLIPSegmentation(
            clip_type=CORRCLIP_CLIP_TYPE,
            model_type=CORRCLIP_MODEL_TYPE,
            dino_type=CORRCLIP_DINO_TYPE,
            name_path=CORRCLIP_NAME_FILE,
            device=self.device,
            mask_generator=mask_generator,
            instance_mask_path=instance_mask_path,
            imagelevel_json_path=IMAGELEVEL_JSON_PATH,
            dataset_type=DATASET_TYPE,
        )

        self._preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.48145466, 0.4578275, 0.40821073],
                    [0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )

    @torch.inference_mode()
    def segment(self, img_path: str) -> np.ndarray:
        """Run semantic segmentation. Returns HxW class index map (int64)."""
        img_path = str(img_path)
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        img = Image.open(img_path).convert("RGB")
        img_tensor = self._preprocess(img).unsqueeze(0).to(self.device)

        seg_pred = self.model.predict([img_tensor, img_path], data_samples=None)
        if isinstance(seg_pred, torch.Tensor):
            seg_np = seg_pred.detach().cpu().numpy()
        else:
            # When data_samples is None, postprocess_result returns tensor directly
            seg_np = torch.as_tensor(seg_pred).detach().cpu().numpy()

        seg_np = np.squeeze(seg_np).astype(np.int64)
        return seg_np


__all__ = ["CorrCLIPMaskGenerator"]
