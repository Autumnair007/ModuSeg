"""C-RADIOv4 feature extractor based on NVIDIA C-RADIOv4-SO400M model."""
import os
import sys
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from src.feature_extractor import BaseFeatureExtractor
from configs.config import CRADIOV4_CKPT


class CRADIOv4FeatureExtractor(BaseFeatureExtractor):
    """C-RADIOv4 feature extractor. Subclasses BaseFeatureExtractor."""

    def __init__(
        self,
        model_path: str,
        device: torch.device,
        model_name: Optional[str] = None,
        img_size: Optional[int] = None,
    ) -> None:
        model_path = model_path or CRADIOV4_CKPT
        img_size = img_size or 512
        
        print(f"[C-RADIOv4] Loading model: C-RADIOv4-SO400M, base size: {img_size}")
        
        model = self._load_local_model(model_path)
        model = model.to(device).eval()
        for p in model.parameters():
            p.requires_grad = False
        
        embed_dim = self._get_embed_dim(model, device, img_size)

        # C-RADIOv4 expects [0, 1] input; internal normalization
        super().__init__(
            device=device,
            patch_size=(16, 16),
            embed_dim=embed_dim,
            model_prefix='C-RADIOv4',
            norm_mean=(0.0, 0.0, 0.0),  # no normalization (handled internally)
            norm_std=(1.0, 1.0, 1.0),
        )
        
        self.model = model
        self.img_size = img_size
        print(f"[{self.model_prefix}] Feature dim: {self.embed_dim}, strategy: mask_pooling")

    def _load_local_model(self, model_path: str):
        """Load C-RADIOv4 model from local checkpoint."""
        import json
        from safetensors.torch import load_file
        
        model_dir = os.path.dirname(model_path) or 'C_RADIOv4_SO400M'
        model_dir = os.path.abspath(model_dir)
        model_path = os.path.abspath(model_path)
        
        if model_dir not in sys.path:
            sys.path.insert(0, model_dir)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"[C-RADIOv4] Weight file not found: {model_path}")
        
        config_path = os.path.join(model_dir, 'config.json')
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"[C-RADIOv4] Config file not found: {config_path}")
        
        from C_RADIOv4_SO400M.hf_model import RADIOConfig, RADIOModel
        
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        config = RADIOConfig(**{k: v for k, v in config_dict.items() 
                                if k not in ['auto_map', 'architectures', 'torch_dtype', 'transformers_version']})
        
        model = RADIOModel(config)
        
        print(f"[C-RADIOv4] Loading weights: {model_path}")
        state_dict = load_file(model_path)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(f"[C-RADIOv4] Warning: {len(missing_keys)} missing keys")
        
        print(f"[C-RADIOv4] Model loaded successfully")
        return model

    def _get_embed_dim(self, model, device, img_size: int) -> int:
        """Probe spatial feature embedding dimension."""
        try:
            dummy_input = torch.zeros(1, 3, img_size, img_size, device=device)
            with torch.no_grad():
                summary, spatial_features = model(dummy_input)
            if spatial_features.dim() == 3:
                return spatial_features.shape[-1]
            elif spatial_features.dim() == 4:
                return spatial_features.shape[1]
        except Exception as e:
            print(f"[C-RADIOv4] Failed to probe embed dim: {e}, using default 1280")
        return 1280

    def _forward_tokens(self, inp: torch.Tensor) -> Tuple[Optional[torch.Tensor], Tuple[int, int]]:
        """Forward pass returning spatial feature tokens."""
        Hp, Wp = self._get_patch_grid(inp)
        
        with torch.no_grad():
            summary, spatial_features = self.model(inp)
        

        if spatial_features.dim() == 4:
            B, C, H, W = spatial_features.shape
            spatial_features = spatial_features.permute(0, 2, 3, 1).reshape(B, H * W, C)
        
        return spatial_features, (Hp, Wp)

    def _fallback_global_feature(self, inp: torch.Tensor) -> torch.Tensor:
        """Fallback: use summary feature."""
        with torch.no_grad():
            summary, _ = self.model(inp)
        return F.normalize(summary.squeeze(0), p=2, dim=0)


__all__ = ["CRADIOv4FeatureExtractor"]
