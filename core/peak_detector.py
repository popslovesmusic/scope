
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .polarity_ops import assign_polarity, detect_zero_crossings


@dataclass(frozen=True)
class Peak:
    family: int
    amplitude: float


@dataclass(frozen=True)
class Band:
    center_family: int
    left_family: int
    right_family: int
    width: int
    peak_amplitude: float


def _is_local_max(amplitude: np.ndarray, i: int) -> bool:
    n = int(amplitude.size)
    left = amplitude[(i - 1) % n]
    mid = amplitude[i]
    right = amplitude[(i + 1) % n]
    return (mid >= left) and (mid > right)


def detect_peaks_and_bands(
    amplitude: np.ndarray,
    *,
    min_height: float = 0.05,
    min_distance: int = 1,
    merge_radius: int = 1,
    band_rel_threshold: float = 0.5,
    wraparound_lattice: bool = True,
) -> Tuple[List[Peak], List[Band]]:
    amp = np.asarray(amplitude, dtype=float)
    n = int(amp.size)
    if n == 0:
        return [], []

    min_height = float(min_height)
    min_distance = int(min_distance)
    merge_radius = int(merge_radius)
    band_rel_threshold = float(band_rel_threshold)

    candidates: List[int] = [i for i in range(n) if (amp[i] >= min_height and _is_local_max(amp, i))]

    # Enforce min_distance (greedy by amplitude).
    candidates.sort(key=lambda i: float(amp[i]), reverse=True)
    chosen: List[int] = []
    for i in candidates:
        if all(min((i - j) % n, (j - i) % n) >= min_distance for j in chosen):
            chosen.append(i)

    # Merge within merge_radius (keep the strongest peak).
    chosen.sort(key=lambda i: i)
    merged: List[int] = []
    for i in chosen:
        if not merged:
            merged.append(i)
            continue
        prev = merged[-1]
        dist = min((i - prev) % n, (prev - i) % n)
        if dist <= merge_radius:
            merged[-1] = prev if amp[prev] >= amp[i] else i
        else:
            merged.append(i)

    peaks = [Peak(family=int(i), amplitude=float(amp[i])) for i in merged]

    bands: List[Band] = []
    for p in peaks:
        thresh = p.amplitude * band_rel_threshold
        left = int(p.family)
        right = int(p.family)

        if wraparound_lattice:
            # Expand on a cyclic lattice.
            steps = 0
            while steps < n:
                nxt = (left - 1) % n
                if nxt == right:
                    break
                if amp[nxt] < thresh:
                    break
                left = nxt
                steps += 1

            steps = 0
            while steps < n:
                nxt = (right + 1) % n
                if nxt == left:
                    break
                if amp[nxt] < thresh:
                    break
                right = nxt
                steps += 1
        else:
            # Expand on a non-wrapping lattice.
            while left - 1 >= 0 and amp[left - 1] >= thresh:
                left -= 1
            while right + 1 < n and amp[right + 1] >= thresh:
                right += 1

        # Normalize wrap bands into [left, right] when left <= right; otherwise represent as spanning ends.
        if wraparound_lattice and left > right:
            width = (n - left) + (right + 1)
        else:
            width = right - left + 1
        bands.append(
            Band(
                center_family=p.family,
                left_family=int(left),
                right_family=int(right),
                width=int(width),
                peak_amplitude=float(p.amplitude),
            )
        )

    return peaks, bands


def label_peaks_by_polarity(peaks: List[Peak], polarity: np.ndarray) -> List[dict]:
    pol = np.asarray(polarity, dtype=int)
    out = []
    for p in peaks:
        out.append({"family": int(p.family), "amplitude": float(p.amplitude), "polarity": int(pol[int(p.family)])})
    return out


def polarity_and_zero_crossings_from_signed_source(
    signed_source: np.ndarray,
    *,
    polarity_threshold: float = 0.01,
    zero_crossing_threshold: float = 0.02,
    wraparound_lattice: bool = True,
) -> Tuple[np.ndarray, List[dict]]:
    pol = assign_polarity(signed_source, polarity_threshold=polarity_threshold)
    zc = detect_zero_crossings(
        signed_source,
        pol,
        zero_crossing_threshold=zero_crossing_threshold,
        wraparound_lattice=wraparound_lattice,
    )
    return pol, zc
