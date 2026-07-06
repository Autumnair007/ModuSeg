"""Feature bank matrix storage helpers."""
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np


FEATURE_BANK_FORMAT_VERSION = 2
FEATURE_BANK_NPZ_KEYS = {"features", "sample_ids", "format_version"}


def _as_feature_matrix(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2:
        raise ValueError(f"features must be 2D, got shape={features.shape}")
    return np.ascontiguousarray(features)


def _as_sample_ids(sample_ids: Iterable[str], n_rows: int) -> np.ndarray:
    sample_ids = np.asarray([str(x) for x in sample_ids], dtype=str)
    if sample_ids.ndim != 1:
        raise ValueError(f"sample_ids must be 1D, got shape={sample_ids.shape}")
    if sample_ids.shape[0] != n_rows:
        raise ValueError(f"sample_ids length must match feature rows: {sample_ids.shape[0]} != {n_rows}")
    return sample_ids


def save_feature_matrix_npz(path: str | Path, features: np.ndarray, sample_ids: Iterable[str]) -> None:
    """Save one class feature matrix without compression."""
    features = _as_feature_matrix(features)
    sample_ids = _as_sample_ids(sample_ids, features.shape[0])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        features=features,
        sample_ids=sample_ids,
        format_version=np.array(FEATURE_BANK_FORMAT_VERSION, dtype=np.int16),
    )


def load_feature_matrix_npz(path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load one v2 matrix-format feature bank file."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        keys = set(data.files)
        if keys != FEATURE_BANK_NPZ_KEYS:
            raise ValueError(
                f"{path} is not a v2 matrix feature bank. "
                f"Expected keys {sorted(FEATURE_BANK_NPZ_KEYS)}, got {sorted(keys)}. "
                "Please rebuild the feature bank."
            )

        version = int(np.asarray(data["format_version"]).reshape(()))
        if version != FEATURE_BANK_FORMAT_VERSION:
            raise ValueError(
                f"{path} has unsupported format_version={version}; "
                f"expected {FEATURE_BANK_FORMAT_VERSION}. Please rebuild the feature bank."
            )

        features = _as_feature_matrix(data["features"])
        sample_ids = _as_sample_ids(data["sample_ids"], features.shape[0])
    return features, sample_ids


__all__ = [
    "FEATURE_BANK_FORMAT_VERSION",
    "FEATURE_BANK_NPZ_KEYS",
    "save_feature_matrix_npz",
    "load_feature_matrix_npz",
]
