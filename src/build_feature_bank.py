"""Feature bank building logic."""
import os
import json
from collections import defaultdict
from typing import Optional

import numpy as np

from configs.config import (
    FEATURES_DIR, INDEX_DIR, META_DIR, DATASET_CLASSES,
)



def create_dirs():
    """Create feature bank output directory structure."""
    for d in [FEATURES_DIR, INDEX_DIR, META_DIR]:
        os.makedirs(d, exist_ok=True)
    for cls_name in DATASET_CLASSES:
        os.makedirs(os.path.join(FEATURES_DIR, cls_name), exist_ok=True)
    os.makedirs(os.path.join(FEATURES_DIR, "background"), exist_ok=True)


def save_class_mapping(num_classes: Optional[int], name_file_path: str):
    """Save class ID to directory name mapping."""
    if num_classes is None:
        num_classes = len(DATASET_CLASSES) + 1
    mapping = {0: 'background'}
    for i in range(1, num_classes):
        idx = i - 1
        if 0 <= idx < len(DATASET_CLASSES):
            mapping[i] = DATASET_CLASSES[idx]
        else:
            mapping[i] = f'cls_{i}'
    meta = {
        'name_file': name_file_path,
        'num_classes': num_classes,
        'id_to_dirname': mapping
    }
    os.makedirs(META_DIR, exist_ok=True)
    with open(os.path.join(META_DIR, 'class_mapping.json'), 'w') as f:
        json.dump(meta, f, indent=2)


class FeatureBank:
    """Feature bank: saves to disk + maintains statistics (no in-memory caching)."""

    def __init__(self):
        self.stats = defaultdict(lambda: {'count': 0, 'mean': None, 'var': None})
        self.bg_retained = 0
        self.bg_discarded = 0

    def _ensure_class_dir(self, cls_name: str):
        os.makedirs(os.path.join(FEATURES_DIR, cls_name), exist_ok=True)

    def _is_valid_feature(self, feature_vec: np.ndarray) -> bool:
        if np.any(np.isnan(feature_vec)) or np.any(np.isinf(feature_vec)):
            return False
        if np.linalg.norm(feature_vec) < 1e-6:
            return False
        return True

    def add_feature(self, cls_name, feature_vec, img_name):
        """Add feature to bank (disk only, no memory caching)."""
        feature_vec = feature_vec.astype(np.float32)

        if not self._is_valid_feature(feature_vec):
            return

        self._ensure_class_dir(cls_name)
        save_path = os.path.join(FEATURES_DIR, cls_name, f"{img_name}_{self.stats[cls_name]['count']:06d}.npy")
        np.save(save_path, feature_vec)
        self._update_stats(cls_name, feature_vec)

    def add_background(self, feature_vec, img_name):
        """Add background feature."""
        cls_name = "background"
        self.add_feature(cls_name, feature_vec, img_name)
        self.bg_retained += 1

    def _update_stats(self, cls_name, feature_vec):
        stat = self.stats[cls_name]
        n = stat['count']
        if n == 0:
            stat['mean'] = feature_vec.copy()
            stat['var'] = np.zeros_like(feature_vec)
        else:
            delta = feature_vec - stat['mean']
            stat['mean'] += delta / (n + 1)
            delta2 = feature_vec - stat['mean']
            stat['var'] += delta * delta2
        stat['count'] = n + 1

    def save_stats(self):
        stats_dict = {}
        for cls_name, stat in self.stats.items():
            stats_dict[cls_name] = {
                'count': int(stat['count']),
                'mean': stat['mean'].flatten().tolist() if stat['mean'] is not None else None,
                'var': stat['var'].flatten().tolist() if stat['var'] is not None else None,
            }
        fg_means = []
        fg_names = []
        for name in DATASET_CLASSES:
            if name in self.stats and self.stats[name]['mean'] is not None and self.stats[name]['count'] > 0:
                fg_names.append(name)
                fg_means.append(self.stats[name]['mean'].reshape(1, -1))
        if len(fg_means) >= 2:
            X = np.vstack(fg_means)
            X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
            cos_mat = (X @ X.T).tolist()
            stats_dict['pairwise_fg_center_cos'] = {
                'classes': fg_names,
                'cosine': cos_mat
            }
        stats_dict['background_info'] = {
            'bg_retained': int(self.bg_retained),
            'bg_discarded': int(self.bg_discarded),
        }
        stats_path = os.path.join(META_DIR, 'stats.json')
        with open(stats_path, 'w') as f:
            json.dump(stats_dict, f, indent=2)
        print(f"\nSaved statistics to {stats_path}")

    def build_indices(self):
        """Build FAISS indices from disk .npy files and pack into .npz."""
        import faiss
        from pathlib import Path
        from tqdm import tqdm
        
        print("\nBuilding FAISS indices from disk...")
        
        # Iterate all class directories
        features_root = Path(FEATURES_DIR)
        class_dirs = [d for d in features_root.iterdir() if d.is_dir()]

        for cls_dir in tqdm(class_dirs, desc="Processing classes"):
            cls_name = cls_dir.name

            npy_files = list(cls_dir.glob("*.npy"))
            if not npy_files:
                continue
            
            feat_list = []
            all_features = {}

            for npy_file in npy_files:
                try:
                    feat = np.load(npy_file)
                    if len(feat.shape) == 1 and feat.shape[0] > 0:
                        feat_list.append(feat)
                        all_features[npy_file.stem] = feat
                except Exception:
                    pass
            
            if not feat_list:
                continue

            X = np.vstack(feat_list).astype(np.float32)
            dim = X.shape[1]
            index = faiss.IndexFlatIP(dim)
            index.add(X)
            index_path = os.path.join(INDEX_DIR, f"faiss_{cls_name}.index")
            faiss.write_index(index, index_path)
            # Pack into .npz and remove .npy files
            npz_path = cls_dir / f"{cls_name}_features.npz"
            np.savez_compressed(npz_path, **all_features)
            
            # Remove original .npy files
            for npy_file in npy_files:
                try:
                    npy_file.unlink()
                except Exception:
                    pass
        
        print("Index building and packing complete.")

__all__ = [
    'create_dirs',
    'save_class_mapping',
    'FeatureBank',
]
