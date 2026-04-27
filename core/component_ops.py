from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from .polarity_ops import extract_polarity_runs


def extract_signed_components(
    *,
    signed_field: np.ndarray,
    amplitude: np.ndarray,
    polarity: np.ndarray,
    mode_index: np.ndarray,
    zero_crossings: List[Dict[str, Any]],
    wraparound_lattice: bool = True,
    component_min_support_mass: float = 0.05,
    component_min_width: int = 1,
) -> List[Dict[str, Any]]:
    sf = np.asarray(signed_field, dtype=float)
    amp = np.asarray(amplitude, dtype=float)
    pol = np.asarray(polarity, dtype=int)
    mode = np.asarray(mode_index, dtype=int)
    if not (sf.shape == amp.shape == pol.shape == mode.shape):
        raise ValueError("component extraction inputs shape mismatch")

    n = int(sf.size)
    runs = extract_polarity_runs(pol, amp, wraparound_lattice=wraparound_lattice)
    comps: List[Dict[str, Any]] = []
    cid = 0

    for r in runs:
        width = int(r["width"])
        if width < int(component_min_width):
            continue
        support_mass = float(r["support_mass"])
        if support_mass < float(component_min_support_mass):
            continue

        sign = int(r["sign"])
        start = int(r["start"])
        end = int(r["end"])
        if start <= end:
            idx = np.arange(start, end + 1, dtype=int)
        else:
            idx = np.concatenate([np.arange(start, n, dtype=int), np.arange(0, end + 1, dtype=int)])

        peak_local = int(idx[int(np.argmax(amp[idx]))])
        peak_amp = float(amp[peak_local])
        mean_signed = float(np.mean(sf[idx])) if idx.size else 0.0

        # Center-of-mass over the cyclic lattice (simple approximation).
        weights = amp[idx] + 1e-9
        angles = (2.0 * np.pi * idx.astype(float)) / max(1.0, float(n))
        cx = float(np.sum(weights * np.cos(angles)))
        sx = float(np.sum(weights * np.sin(angles)))
        center_angle = float(np.arctan2(sx, cx))
        if center_angle < 0:
            center_angle += 2.0 * np.pi
        center_family = (center_angle / (2.0 * np.pi)) * float(n)

        mode_vals = mode[idx]
        nonzero_modes = mode_vals[mode_vals != 0]
        mode_label = int(nonzero_modes[0]) if nonzero_modes.size else 0

        adj_zc: List[Dict[str, Any]] = []
        for z in zero_crossings or []:
            li = int(z.get("left_index", -1))
            ri = int(z.get("right_index", -1))
            if li in idx or ri in idx:
                adj_zc.append(z)

        comps.append(
            {
                "component_id": cid,
                "stable_id": None,
                "sign": sign,
                "start_family": start,
                "end_family": end,
                "width": int(width),
                "center_family": float(center_family),
                "support_mass": float(support_mass),
                "peak_family": int(peak_local),
                "peak_amplitude": float(peak_amp),
                "mean_signed_strength": float(mean_signed),
                "adjacent_zero_crossings": adj_zc,
                "mode_label": mode_label,
            }
        )
        cid += 1

    comps.sort(key=lambda c: float(c["support_mass"]), reverse=True)
    return comps


def build_component_masks(components: List[Dict[str, Any]], *, size: int) -> Dict[int, np.ndarray]:
    n = int(size)
    out: Dict[int, np.ndarray] = {}
    for c in components or []:
        sid = c.get("stable_id", c.get("component_id", None))
        if sid is None:
            continue
        mask = np.zeros(n, dtype=float)
        start = int(c["start_family"])
        end = int(c["end_family"])
        if start <= end:
            mask[start : end + 1] = 1.0
        else:
            mask[start:] = 1.0
            mask[: end + 1] = 1.0
        out[int(sid)] = mask
    return out


def _component_indices(comp: Dict[str, Any], size: int) -> np.ndarray:
    n = int(size)
    start = int(comp["start_family"])
    end = int(comp["end_family"])
    if start <= end:
        return np.arange(start, end + 1, dtype=int)
    return np.concatenate([np.arange(start, n, dtype=int), np.arange(0, end + 1, dtype=int)])


def _overlap_fraction(a: Dict[str, Any], b: Dict[str, Any], size: int) -> float:
    ai = set(_component_indices(a, size).tolist())
    bi = set(_component_indices(b, size).tolist())
    if not ai or not bi:
        return 0.0
    inter = len(ai.intersection(bi))
    union = len(ai.union(bi))
    return float(inter) / float(union) if union else 0.0


def match_components_across_phases(
    *,
    previous_components: List[Dict[str, Any]],
    current_components: List[Dict[str, Any]],
    size: int,
    overlap_weight: float = 0.5,
    center_weight: float = 0.3,
    sign_weight: float = 0.2,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    n = int(size)
    prev = [dict(c) for c in (previous_components or [])]
    cur = [dict(c) for c in (current_components or [])]

    used_prev: set[int] = set()
    used_cur: set[int] = set()

    # Allocate new stable IDs beyond the previous max.
    prev_ids = [c.get("stable_id") for c in prev if c.get("stable_id") is not None]
    next_id = (max(int(i) for i in prev_ids) + 1) if prev_ids else 0

    matches: List[Tuple[float, int, int]] = []
    for pi, p in enumerate(prev):
        for ci, c in enumerate(cur):
            if int(p.get("sign", 0)) != int(c.get("sign", 0)):
                sign_sim = 0.0
            else:
                sign_sim = 1.0

            ov = _overlap_fraction(p, c, n)
            dp = abs(float(p.get("center_family", 0.0)) - float(c.get("center_family", 0.0)))
            dp = min(dp, float(n) - dp)
            center_sim = float(np.exp(-dp / max(1.0, float(n) / 6.0)))

            score = float(overlap_weight) * ov + float(center_weight) * center_sim + float(sign_weight) * sign_sim
            matches.append((score, pi, ci))

    matches.sort(key=lambda t: t[0], reverse=True)

    for score, pi, ci in matches:
        if pi in used_prev or ci in used_cur:
            continue
        if score <= 0.0:
            continue
        stable_id = prev[pi].get("stable_id")
        if stable_id is None:
            stable_id = next_id
            next_id += 1
        cur[ci]["stable_id"] = int(stable_id)
        used_prev.add(pi)
        used_cur.add(ci)

    # Assign new IDs to unmatched current components.
    for c in cur:
        if c.get("stable_id") is None:
            c["stable_id"] = int(next_id)
            next_id += 1

    # Diagnostics: split/merge approximations.
    split_count = 0
    merge_count = 0
    if prev and cur:
        # Build overlap graph.
        overlaps: Dict[int, List[int]] = {}
        rev_overlaps: Dict[int, List[int]] = {}
        for pi, p in enumerate(prev):
            for ci, c in enumerate(cur):
                if int(p.get("sign", 0)) != int(c.get("sign", 0)):
                    continue
                if _overlap_fraction(p, c, n) > 0.0:
                    overlaps.setdefault(pi, []).append(ci)
                    rev_overlaps.setdefault(ci, []).append(pi)
        split_count = sum(1 for _, cs in overlaps.items() if len(cs) > 1)
        merge_count = sum(1 for _, ps in rev_overlaps.items() if len(ps) > 1)

    # Identity persistence: fraction of current components that re-used a previous stable ID.
    prev_id_set = {int(c["stable_id"]) for c in prev if c.get("stable_id") is not None}
    reused = sum(1 for c in cur if int(c.get("stable_id", -1)) in prev_id_set)
    persistence = float(reused) / float(len(cur)) if cur else 0.0

    meta = {
        "component_split_count": int(split_count),
        "component_merge_count": int(merge_count),
        "component_identity_persistence": float(persistence),
    }
    return cur, meta


def summarize_component_set(components: List[Dict[str, Any]]) -> Dict[str, Any]:
    comps = components or []
    if not comps:
        return {"count": 0}
    widths = [int(c.get("width", 0)) for c in comps]
    masses = [float(c.get("support_mass", 0.0)) for c in comps]
    dom = max(comps, key=lambda c: float(c.get("support_mass", 0.0)))
    return {
        "count": int(len(comps)),
        "largest_width": int(max(widths) if widths else 0),
        "total_support_mass": float(sum(masses)),
        "dominant_component_id": dom.get("stable_id", dom.get("component_id")),
        "dominant_sign": int(dom.get("sign", 0)),
        "dominant_support_mass": float(dom.get("support_mass", 0.0)),
    }


def component_caution_scores(
    *,
    components: List[Dict[str, Any]],
    caution_field: np.ndarray,
    size: int,
    mode: str = "mean",
    blended_alpha: float = 0.5,
    recovery_scores: Dict[int, float] | None = None,
    recovery_weight: float = 0.0,
) -> Dict[int, Dict[str, float]]:
    n = int(size)
    c = np.asarray(caution_field, dtype=float)
    if c.shape != (n,):
        raise ValueError("caution_field shape mismatch")

    mode = str(mode or "mean").strip().lower()
    alpha = float(blended_alpha)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    recovery_weight = float(np.clip(float(recovery_weight), 0.0, 1.0))

    scores: Dict[int, Dict[str, float]] = {}
    for comp in components or []:
        sid = comp.get("stable_id", comp.get("component_id", None))
        if sid is None:
            continue
        start = int(comp.get("start_family", 0))
        end = int(comp.get("end_family", 0))
        if start <= end:
            idx = np.arange(start, end + 1, dtype=int)
        else:
            idx = np.concatenate([np.arange(start, n, dtype=int), np.arange(0, end + 1, dtype=int)])
        if idx.size:
            mean_v = float(np.mean(c[idx]))
            peak_v = float(np.max(c[idx]))
        else:
            mean_v = 0.0
            peak_v = 0.0

        if mode == "peak":
            score_v = peak_v
        elif mode == "blended":
            score_v = (1.0 - alpha) * mean_v + alpha * peak_v
        else:
            score_v = mean_v

        rec_v = 0.0 if not recovery_scores else float(recovery_scores.get(int(sid), 0.0))
        rec_v = float(np.clip(rec_v, 0.0, 1.0))
        net_v = float(np.clip(float(score_v) - recovery_weight * rec_v, 0.0, 1.0))

        scores[int(sid)] = {"mean": float(mean_v), "peak": float(peak_v), "score": float(score_v), "recovery": float(rec_v), "net": float(net_v)}
    return scores


def component_recovery_scores(
    *,
    components: List[Dict[str, Any]],
    caution_field: np.ndarray,
    size: int,
    mode: str = "mean",
) -> Dict[int, float]:
    """
    Recovery-favorable evidence aggregated per component.

    Default behavior: components with lower caution get higher recovery scores.
    """
    mode = str(mode or "mean").strip().lower()
    scores = component_caution_scores(components=components, caution_field=caution_field, size=size, mode=("peak" if mode == "peak" else "mean"))
    out: Dict[int, float] = {}
    for sid, d in scores.items():
        base = float(d.get("peak" if mode == "peak" else "mean", 0.0))
        out[int(sid)] = float(np.clip(1.0 - base, 0.0, 1.0))
    return out
