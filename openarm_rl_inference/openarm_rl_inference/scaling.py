from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

_EPS = 1e-8


def standardize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    mean_arr = np.asarray(mean, dtype=np.float32)
    std_arr = np.asarray(std, dtype=np.float32)
    if mean_arr.shape != std_arr.shape:
        raise ValueError(f"Mean/std shape mismatch: {mean_arr.shape} vs {std_arr.shape}")
    if arr.shape[-1] != mean_arr.shape[-1]:
        raise ValueError(
            f"Input trailing dimension {arr.shape[-1]} does not match mean/std {mean_arr.shape[-1]}"
        )
    expand_shape = (1,) * (arr.ndim - 1) + mean_arr.shape
    mean_expanded = mean_arr.reshape(expand_shape)
    std_expanded = np.maximum(std_arr.reshape(expand_shape), _EPS)
    standardized = (arr - mean_expanded) / std_expanded
    return standardized.astype(np.float32)


def load_scaler_from_npz(path: str) -> Tuple[np.ndarray, np.ndarray]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Scaler npz not found at {file_path}")
    data = np.load(file_path)
    if "mean" not in data or "std" not in data:
        raise KeyError("Scaler npz must contain 'mean' and 'std' arrays")
    mean = np.asarray(data["mean"], dtype=np.float32)
    std = np.asarray(data["std"], dtype=np.float32)
    if mean.shape != std.shape:
        raise ValueError(f"Scaler shapes mismatch: mean {mean.shape}, std {std.shape}")
    return mean, std
