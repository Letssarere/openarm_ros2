from __future__ import annotations

import numpy as np


def ema(prev: np.ndarray, current: np.ndarray, alpha: float) -> np.ndarray:
    if alpha <= 0.0:
        return np.asarray(current, dtype=np.float32)
    prev_arr = np.asarray(prev, dtype=np.float32)
    curr_arr = np.asarray(current, dtype=np.float32)
    return (1.0 - alpha) * prev_arr + alpha * curr_arr


def clamp_step(prev: np.ndarray, current: np.ndarray, max_delta: float) -> np.ndarray:
    if max_delta <= 0.0:
        return np.asarray(current, dtype=np.float32)
    prev_arr = np.asarray(prev, dtype=np.float32)
    curr_arr = np.asarray(current, dtype=np.float32)
    delta = np.clip(curr_arr - prev_arr, -max_delta, max_delta)
    return prev_arr + delta


def clip_joint_limits(values: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float32)
    lower_arr = np.asarray(lower, dtype=np.float32)
    upper_arr = np.asarray(upper, dtype=np.float32)
    if vals.shape != lower_arr.shape or vals.shape != upper_arr.shape:
        raise ValueError("Joint limit arrays must share the same shape")
    return np.clip(vals, lower_arr, upper_arr)
