from __future__ import annotations

from typing import Tuple

import numpy as np


def laplacian_family_1d(arr: np.ndarray, *, mode: str = "wrap") -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 1:
        raise ValueError("laplacian_family_1d expects a 1D array")
    if arr.size < 2:
        return np.zeros_like(arr)

    if mode == "wrap":
        left = np.roll(arr, 1)
        right = np.roll(arr, -1)
    elif mode == "reflect":
        left = np.empty_like(arr)
        right = np.empty_like(arr)
        left[1:] = arr[:-1]
        left[0] = arr[1]
        right[:-1] = arr[1:]
        right[-1] = arr[-2]
    elif mode == "clamp":
        left = np.empty_like(arr)
        right = np.empty_like(arr)
        left[1:] = arr[:-1]
        left[0] = arr[0]
        right[:-1] = arr[1:]
        right[-1] = arr[-1]
    else:
        raise ValueError(f"Unknown diffusion boundary mode: {mode}")

    return left + right - 2.0 * arr


def laplacian_orientation_1d(arr: np.ndarray) -> np.ndarray:
    """
    Orientation-channel Laplacian over the four v14 orientation operators.

    This is a group-averaging Laplacian:
        ?_o(x) = mean([x, reverse(x), roll(+1)(x), roll(-1)(x)]) - x

    It preserves total mass because each transform is a permutation of indices.
    """
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 1:
        raise ValueError("laplacian_orientation_1d expects a 1D array")
    if arr.size == 0:
        return np.zeros_like(arr)

    views = [
        arr,
        arr[::-1],
        np.roll(arr, 1),
        np.roll(arr, -1),
    ]
    mean_view = sum(views) / float(len(views))
    return mean_view - arr


def apply_family_diffusion(
    arr: np.ndarray,
    *,
    coeff: float,
    mode: str = "wrap",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(arr, dtype=float)
    coeff = float(coeff)
    if coeff == 0.0:
        zeros = np.zeros_like(arr)
        return arr.copy(), zeros, zeros

    lap = laplacian_family_1d(arr, mode=mode)
    diff_term = coeff * lap
    return arr + diff_term, diff_term, lap


def apply_orientation_diffusion(
    arr: np.ndarray,
    *,
    coeff: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(arr, dtype=float)
    coeff = float(coeff)
    if coeff == 0.0:
        zeros = np.zeros_like(arr)
        return arr.copy(), zeros, zeros

    lap = laplacian_orientation_1d(arr)
    diff_term = coeff * lap
    return arr + diff_term, diff_term, lap


def diffuse_operator_scores(scores: np.ndarray, *, coeff: float) -> np.ndarray:
    """
    Lightweight diffusion over the 4-operator score vector.

    This operates in score-space only. For n=4, we use a complete-graph neighbor mean:
        s_i <- s_i + coeff * (mean(scores[j != i]) - s_i)

    coeff=0.0 preserves input exactly.
    """
    x = np.asarray(scores, dtype=float)
    if x.shape != (4,):
        raise ValueError("diffuse_operator_scores expects shape (4,)")
    c = float(np.clip(float(coeff), 0.0, 1.0))
    if c == 0.0:
        return x.copy()

    total = float(np.sum(x))
    out = np.empty_like(x)
    for i in range(4):
        neighbor_mean = (total - float(x[i])) / 3.0
        out[i] = float(x[i]) + c * (neighbor_mean - float(x[i]))
    return out
