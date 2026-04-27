from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np


def assign_polarity(
    signed_source: np.ndarray,
    *,
    polarity_threshold: float = 0.01,
) -> np.ndarray:
    x = np.asarray(signed_source, dtype=float)
    thr = float(polarity_threshold)
    pol = np.zeros_like(x, dtype=int)
    pol[x > thr] = 1
    pol[x < -thr] = -1
    return pol


@dataclass(frozen=True)
class ZeroCrossing:
    left_index: int
    right_index: int
    crossing_position: float
    crossing_strength: float
    left_polarity: int
    right_polarity: int


def detect_zero_crossings(
    signed_source: np.ndarray,
    polarity: np.ndarray,
    *,
    zero_crossing_threshold: float = 0.02,
    wraparound_lattice: bool = True,
) -> List[Dict[str, Any]]:
    x = np.asarray(signed_source, dtype=float)
    pol = np.asarray(polarity, dtype=int)
    if x.shape != pol.shape:
        raise ValueError("signed_source/polarity shape mismatch")
    n = int(x.size)
    thr = float(zero_crossing_threshold)
    out: List[Dict[str, Any]] = []
    if n < 2:
        return out

    pairs = [(i, i + 1) for i in range(n - 1)]
    if wraparound_lattice:
        pairs.append((n - 1, 0))

    for li, ri in pairs:
        lp = int(pol[li])
        rp = int(pol[ri])
        if lp == 0 or rp == 0 or lp == rp:
            continue
        if (abs(float(x[li])) < thr) or (abs(float(x[ri])) < thr):
            continue

        pos = float(li) + 0.5
        if wraparound_lattice and li == n - 1 and ri == 0:
            pos = float(n) - 0.5

        strength = float(min(abs(float(x[li])), abs(float(x[ri]))))
        out.append(
            {
                "left_index": int(li),
                "right_index": int(ri),
                "crossing_position": pos,
                "crossing_strength": strength,
                "left_polarity": lp,
                "right_polarity": rp,
            }
        )

    out.sort(key=lambda d: float(d["crossing_strength"]), reverse=True)
    return out


def extract_polarity_runs(
    polarity: np.ndarray,
    amplitude: np.ndarray,
    *,
    wraparound_lattice: bool = True,
) -> List[Dict[str, Any]]:
    pol = np.asarray(polarity, dtype=int)
    amp = np.asarray(amplitude, dtype=float)
    if pol.shape != amp.shape:
        raise ValueError("polarity/amplitude shape mismatch")
    n = int(pol.size)
    if n == 0:
        return []

    runs: List[Dict[str, Any]] = []
    i = 0
    while i < n:
        if int(pol[i]) == 0:
            i += 1
            continue
        sign = int(pol[i])
        start = i
        j = i
        while j < n and int(pol[j]) == sign:
            j += 1
        end = j - 1
        idx = list(range(start, end + 1))
        runs.append(
            {
                "sign": int(sign),
                "start": int(start),
                "end": int(end),
                "width": int(end - start + 1),
                "support_mass": float(np.sum(amp[idx])),
            }
        )
        i = j

    if wraparound_lattice and len(runs) >= 2 and int(pol[0]) != 0 and int(pol[0]) == int(pol[-1]):
        first = runs[0]
        last = runs[-1]
        if int(first["sign"]) == int(last["sign"]) and int(first["start"]) == 0 and int(last["end"]) == n - 1:
            merged_width = int(first["width"]) + int(last["width"])
            merged_mass = float(first["support_mass"]) + float(last["support_mass"])
            runs = runs[1:-1]
            runs.insert(
                0,
                {
                    "sign": int(first["sign"]),
                    "start": int(last["start"]),
                    "end": int(first["end"]),
                    "width": int(merged_width),
                    "support_mass": float(merged_mass),
                },
            )

    return runs


def compute_mode_index(polarity: np.ndarray, *, wraparound_lattice: bool = True) -> np.ndarray:
    pol = np.asarray(polarity, dtype=int)
    n = int(pol.size)
    if n == 0:
        return pol.copy()

    mode = np.zeros(n, dtype=int)
    run_id = 0
    i = 0
    while i < n:
        if pol[i] == 0:
            i += 1
            continue
        run_id += 1
        sign = int(pol[i])
        j = i
        while j < n and int(pol[j]) == sign:
            mode[j] = sign * run_id
            j += 1
        i = j

    if wraparound_lattice and n > 1 and pol[0] != 0 and pol[0] == pol[-1]:
        # Merge first and last run labels into the first label.
        first_label = int(mode[0])
        last_label = int(mode[-1])
        if first_label != 0 and last_label != 0 and first_label != last_label:
            mode[mode == last_label] = first_label

    return mode


def dominant_polarity(polarity: np.ndarray) -> int:
    pol = np.asarray(polarity, dtype=int)
    pos = int(np.sum(pol == 1))
    neg = int(np.sum(pol == -1))
    if pos == 0 and neg == 0:
        return 0
    return 1 if pos >= neg else -1


def polarity_shift_metric(old: np.ndarray, new: np.ndarray) -> float:
    o = np.asarray(old, dtype=int)
    n = np.asarray(new, dtype=int)
    if o.shape != n.shape:
        raise ValueError("old/new polarity shape mismatch")
    # 0..1: average per-slot polarity change magnitude.
    return float(np.mean(np.abs(n - o)) / 2.0) if o.size else 0.0


def summarize_dominant_component(
    *,
    amplitude: np.ndarray,
    phase: np.ndarray,
    signed_source: np.ndarray,
    polarity: np.ndarray,
    mode_index: np.ndarray,
    zero_crossings: List[Dict[str, Any]],
) -> Dict[str, Any]:
    amp = np.asarray(amplitude, dtype=float)
    ph = np.asarray(phase, dtype=float)
    s = np.asarray(signed_source, dtype=float)
    pol = np.asarray(polarity, dtype=int)
    mode = np.asarray(mode_index, dtype=int)
    if not (amp.shape == ph.shape == s.shape == pol.shape == mode.shape):
        raise ValueError("summary inputs shape mismatch")
    if amp.size == 0:
        return {}

    idx = int(np.argmax(np.abs(s)))
    nearest = None
    if zero_crossings:
        n = int(amp.size)
        best_d = 1e9
        for z in zero_crossings:
            pos = float(z.get("crossing_position", 0.0))
            # crossing_position can be n-0.5 for wrap crossing
            pos_mod = pos % float(n)
            d = abs(pos_mod - float(idx))
            d = min(d, float(n) - d)
            if d < best_d:
                best_d = d
                nearest = z

    return {
        "family": idx,
        "dominant_amplitude": float(amp[idx]),
        "dominant_phase": float(ph[idx]),
        "dominant_polarity": int(pol[idx]),
        "dominant_mode": int(mode[idx]),
        "dominant_zero_crossing": nearest,
        "dominant_signed_strength": float(s[idx]),
    }
