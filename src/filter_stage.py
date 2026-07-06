"""In-place feature bank filtering by dropping outlier samples."""
from pathlib import Path
import json
import numpy as np
from configs.config import (
    OUTPUT_ROOT,
    FILTER_BACKGROUND_PREFIX,
    FILTER_FOREGROUND_DROP_RATIO,
    FILTER_BACKGROUND_DROP_RATIO,
)
from src.feature_bank_storage import load_feature_matrix_npz, save_feature_matrix_npz


FEATURE_BANK_ROOT = OUTPUT_ROOT
FEATURES_DIR = str(Path(FEATURE_BANK_ROOT) / "features")
INDEX_DIR = str(Path(FEATURE_BANK_ROOT) / "index")
META_DIR = str(Path(FEATURE_BANK_ROOT) / "meta")


def _load_class_dirs(features_dir: str):
    """Load all class subdirectories."""
    p = Path(features_dir)
    if not p.is_dir():
        raise FileNotFoundError(f"Features directory not found: {features_dir}")
    dirs = [x for x in sorted(p.iterdir()) if x.is_dir()]
    if not dirs:
        print(f"[Filter] No class subdirectories under {features_dir}")
    return dirs


def _load_features_with_sample_ids(cls_dir: Path):
    """Load one matrix-format feature file from a class directory."""
    npz_files = sorted(cls_dir.glob('*.npz'))
    if not npz_files:
        return np.array([], dtype=str), np.empty((0,), dtype=np.float32)
    if len(npz_files) != 1:
        raise RuntimeError(f"Expected exactly one v2 feature file under {cls_dir}, got {len(npz_files)}")
    X, sample_ids = load_feature_matrix_npz(npz_files[0])
    return sample_ids, X


def _compute_center_mean(X: np.ndarray) -> np.ndarray:
    """Compute feature centroid."""
    if X.ndim != 2 or X.shape[0] == 0:
        return np.zeros((X.shape[1] if X.ndim == 2 else 1,), dtype=np.float32)
    return X.mean(axis=0)


def _filter_by_top_ratio(sample_ids, X, ratio_drop: float):
    """Drop the most distant samples by ratio."""
    if X.ndim != 2 or X.shape[0] == 0:
        return np.array([], dtype=str), X, np.array([]), np.array([])
    center = _compute_center_mean(X)
    diffs = X - center[None, :]
    dists = np.sqrt((diffs * diffs).sum(axis=1))
    n = X.shape[0]
    k_drop = int(np.floor(n * ratio_drop))
    if k_drop <= 0:
        keep_idx = np.arange(n)
    else:
        # Sort by distance descending, drop farthest k_drop
        order = np.argsort(dists)[::-1]
        drop_idx = set(order[:k_drop].tolist())
        keep_idx = np.array([i for i in range(n) if i not in drop_idx], dtype=np.int64)
    X_keep = X[keep_idx]
    sample_ids_keep = sample_ids[keep_idx]
    return sample_ids_keep, X_keep, center, dists

def _build_faiss_index_ip(cls_name: str, X: np.ndarray):
    """Build and save FAISS inner-product index."""
    try:
        import faiss  # type: ignore
    except Exception:
        print("[Filter] faiss not installed, skip index for:", cls_name)
        return
    if X.shape[0] == 0:
        return
    Xn = X.astype(np.float32)
    # L2 normalize for cosine similarity
    Xn = Xn / (np.linalg.norm(Xn, axis=1, keepdims=True) + 1e-8)
    dim = int(Xn.shape[1])
    index = faiss.IndexFlatIP(dim)
    index.add(np.ascontiguousarray(Xn))
    out_path = Path(INDEX_DIR) / f'faiss_{cls_name}.index'
    faiss.write_index(index, str(out_path))


def _recompute_stats(class_to_features: dict) -> dict:
    """Recompute statistics after filtering."""
    stats = {}
    for cls, X in class_to_features.items():
        if X.ndim != 2 or X.shape[0] == 0:
            stats[cls] = {"count": 0, "mean": [], "var": []}
            continue
        mean = X.mean(axis=0)
        var = X.var(axis=0)
        stats[cls] = {
            "count": int(X.shape[0]),
            "mean": mean.astype(float).tolist(),
            "var": var.astype(float).tolist(),
        }
    return stats


def run_filter_stage(drop_ratio: float | None = None):
    """Run in-place feature bank filtering."""
    print("\n[Filter] Running in-place feature bank filtering...")

    class_dirs = _load_class_dirs(FEATURES_DIR)
    class_to_features = {}
    filter_report = {}

    fg_ratio_used = drop_ratio if drop_ratio is not None else FILTER_FOREGROUND_DROP_RATIO

    for cls_dir in class_dirs:
        cls_name = cls_dir.name
        sample_ids, X = _load_features_with_sample_ids(cls_dir)
        original_count = X.shape[0] if (X.ndim == 2) else 0
        if X.ndim != 2 or X.shape[0] == 0:
            (Path(FEATURES_DIR) / cls_name).mkdir(parents=True, exist_ok=True)
            class_to_features[cls_name] = np.empty((0,), dtype=np.float32)
            filter_report[cls_name] = {
                "original_count": int(original_count),
                "kept_count": 0,
                "applied_drop_ratio": 0.0,
                "is_background": bool(cls_name.startswith(FILTER_BACKGROUND_PREFIX))
            }
            continue
        
        # Determine drop ratio
        applied_ratio = FILTER_BACKGROUND_DROP_RATIO if cls_name.startswith(FILTER_BACKGROUND_PREFIX) else fg_ratio_used
        
        # Execute filtering
        sample_ids_keep, X_keep, center, dists = _filter_by_top_ratio(sample_ids, X, applied_ratio)

        # Save filtered features as matrix-format .npz
        cls_dir_path = Path(FEATURES_DIR) / cls_name
        cls_dir_path.mkdir(parents=True, exist_ok=True)

        npz_path = cls_dir_path / f"{cls_name}_features.npz"
        save_feature_matrix_npz(npz_path, X_keep, sample_ids_keep)

        for p in cls_dir_path.glob('*.npy'):
            try:
                p.unlink()
            except Exception:
                pass
        
        # Rebuild FAISS index
        class_to_features[cls_name] = X_keep
        _build_faiss_index_ip(cls_name, X_keep)

        filter_report[cls_name] = {
            "original_count": int(original_count),
            "kept_count": int(X_keep.shape[0]),
            "applied_drop_ratio": float(applied_ratio),
            "is_background": bool(cls_name.startswith(FILTER_BACKGROUND_PREFIX))
        }

    # Save stats and report
    stats = _recompute_stats(class_to_features)
    with open(Path(META_DIR) / 'filtered_stats.json', 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    with open(Path(META_DIR) / 'filter_report.json', 'w', encoding='utf-8') as f:
        json.dump(filter_report, f, indent=2, ensure_ascii=False)
    print(f"[Filter] Filtering complete, results saved in: {FEATURE_BANK_ROOT}")
