"""Feature bank dataset for parallel data loading with PyTorch DataLoader."""
from pathlib import Path
from typing import List, Optional, Tuple

from torch.utils.data import Dataset

from configs.config import MIN_MASK_AREA
from src.region_extractors import (
    get_regions_from_pseudo,
    get_regions_from_corrclip,
)


class FeatureBankDataset(Dataset):
    """Dataset for feature bank building.
    
    Modes:
    - 'pseudo': pre-generated pseudo masks (supports multiprocessing)
    - 'corrclip': real-time CorrCLIP generation (num_workers=0)
    """

    def __init__(
        self, 
        image_paths: List[Path], 
        mode: str = 'pseudo',
        pseudo_gen: Optional[object] = None
    ):
        self.image_paths = image_paths
        self.mode = mode
        self.pseudo_gen = pseudo_gen

        if mode not in ['pseudo', 'corrclip']:
            raise ValueError(f"Invalid mode: {mode}, must be 'pseudo' or 'corrclip'")
        if mode == 'corrclip' and pseudo_gen is None:
            raise ValueError("CorrCLIP mode requires pseudo_gen parameter")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[str, list]:
        img_path = self.image_paths[idx]
        regions = []
        try:
            if self.mode == 'pseudo':
                regions = get_regions_from_pseudo(str(img_path), min_area=MIN_MASK_AREA)
            elif self.mode == 'corrclip':
                regions = get_regions_from_corrclip(
                    self.pseudo_gen, str(img_path), min_area=MIN_MASK_AREA
                )
        except Exception as e:
            print(f"\nWarning: error processing {img_path.name}: {e}")
            regions = []
        return str(img_path), regions


def collate_fn(batch):
    """Filter out samples with no regions."""
    return [item for item in batch if item[1]]


__all__ = ['FeatureBankDataset', 'collate_fn']
