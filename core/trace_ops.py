from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np


def _bounded(x: float) -> float:
    return float(np.clip(float(x), 0.0, 1.0))


def normalize_caution_inputs(*, salience: float, similarity: float) -> Tuple[float, float]:
    """
    Bound and normalize the salience/similarity inputs used for caution composition.

    Both are treated as [0, 1] contributions.
    """
    return _bounded(salience), _bounded(similarity)


def compute_caution_release(
    *,
    salience_max: float,
    similarity_max: float,
    release_rate: float = 0.0,
) -> float:
    """
    Compute a small decay/release term for caution.

    Intuition: when recent traces are highly salient but the current structure is weakly similar to them,
    the system should begin releasing caution instead of only sustaining it.
    """
    r = max(0.0, float(release_rate))
    if r == 0.0:
        return 0.0
    sal, sim = normalize_caution_inputs(salience=float(salience_max), similarity=float(similarity_max))
    return _bounded(r * sal * (1.0 - sim))


def compute_recovery_signal(
    *,
    similarity_max: float,
    salience_max: float,
    corridor_block_count: float,
    boundary_penalty_energy: float,
    spread_measure: float = 0.0,
) -> float:
    """
    Derive a bounded recovery scalar from:
    - low similarity to recent high-salience traces (divergence),
    - low instability/penalty and few blocks (safe continuation quality).
    """
    sim = _bounded(similarity_max)
    sal = _bounded(salience_max)
    blocks = max(0.0, float(corridor_block_count))
    boundary = max(0.0, float(boundary_penalty_energy))
    spread = max(0.0, float(spread_measure))

    divergence = _bounded((1.0 - sim) * sal)
    s_blocks = float(np.exp(-0.25 * blocks))
    s_boundary = float(np.exp(-boundary))
    s_spread = float(np.exp(-0.10 * spread))
    continuation_quality = _bounded(s_blocks * s_boundary * s_spread)
    return _bounded(divergence * continuation_quality)


def compute_recontextualization_score(*, similarity_max: float, salience_max: float) -> float:
    """
    Measure whether current structure sufficiently differs from prior high-salience patterns.
    High when: prior salience is high and similarity is low.
    """
    sim = _bounded(similarity_max)
    sal = _bounded(salience_max)
    return _bounded(sal * (1.0 - sim))


def compute_trace_salience(
    phase_history: List[Dict[str, Any]],
    *,
    window: int = 8,
) -> List[float]:
    """
    Salience heuristics over recent trace entries.
    Higher when: boundary penalties are high, entry is blocked often, cancellation exposure is high,
    or dynamics are unstable (polarity shifts, high diffusion/spread).
    """
    hist = phase_history or []
    if window <= 0 or not hist:
        return []
    recent = hist[-int(window) :]
    sal: List[float] = []
    for t in recent:
        boundary = float(t.get("boundary_penalty_energy", 0.0))
        blocks = float(t.get("corridor_block_count", len(t.get("corridor_blocks", []) or [])))
        cancel = float(t.get("cancellation_penalty_energy", 0.0))
        pol_shift = float(t.get("polarity_shift", 0.0))
        spread = float(t.get("spread_measure", 0.0))

        s_boundary = 1.0 - float(np.exp(-boundary))
        s_blocks = 1.0 - float(np.exp(-0.25 * blocks))
        s_cancel = 1.0 - float(np.exp(-cancel))
        s_pol = _bounded(pol_shift)
        s_spread = 1.0 - float(np.exp(-0.1 * spread))

        # Weighted sum, clipped.
        s = 0.30 * s_boundary + 0.20 * s_blocks + 0.25 * s_cancel + 0.15 * s_pol + 0.10 * s_spread
        sal.append(_bounded(s))
    return sal


def compute_trace_similarity(current: Dict[str, Any], prior: Dict[str, Any], *, size: int = 12) -> float:
    """
    Coarse structural similarity between current and prior topology/component summaries.
    """
    c_sign = int(current.get("active_component_sign", 0))
    p_sign = int(prior.get("active_component_sign", 0))
    sign_sim = 1.0 if (c_sign != 0 and c_sign == p_sign) else (0.25 if (c_sign == 0 or p_sign == 0) else 0.0)

    c_w = float(current.get("active_component_width", 0.0))
    p_w = float(prior.get("active_component_width", 0.0))
    width_sim = float(np.exp(-abs(c_w - p_w) / max(1.0, float(size) / 6.0)))

    c_runs = float(current.get("signed_run_count", 0.0))
    p_runs = float(prior.get("signed_run_count", 0.0))
    run_sim = float(np.exp(-abs(c_runs - p_runs) / 2.0))

    c_lrw = float(current.get("largest_signed_run_width", 0.0))
    p_lrw = float(prior.get("largest_signed_run_width", 0.0))
    lrw_sim = float(np.exp(-abs(c_lrw - p_lrw) / max(1.0, float(size) / 6.0)))

    op_sim = 1.0 if str(current.get("selected_operator", "")) == str(prior.get("selected_operator", "")) else 0.6

    sim = 0.30 * sign_sim + 0.25 * width_sim + 0.20 * run_sim + 0.15 * lrw_sim + 0.10 * op_sim
    return _bounded(sim)


def compute_caution_field(
    *,
    size: int,
    components: List[Dict[str, Any]],
    component_masks: Dict[int, np.ndarray],
    current_structure: Dict[str, Any],
    trace_history: List[Dict[str, Any]],
    salience: List[float],
    salience_weight: float = 0.4,
    similarity_weight: float = 0.6,
    caution_release_rate: float = 0.0,
    recovery_scalar: float = 0.0,
    recontextualization_score: float = 0.0,
    recovery_rate: float = 0.0,
    recontextualization_weight: float = 0.0,
) -> Tuple[float, np.ndarray, float, np.ndarray, float, float, np.ndarray, float, float]:
    """
    Build caution_field by projecting salience-weighted similarity onto current component masks.

    Returns:
      raw_caution_scalar, raw_caution_field,
      bounded_caution_scalar, bounded_caution_field,
      caution_release_scalar,
      caution_after_recovery_scalar, caution_after_recovery_field,
      salience_max, similarity_max
    """
    n = int(size)
    raw_field = np.zeros(n, dtype=float)
    if not trace_history or not salience or not components:
        zeros = np.zeros(n, dtype=float)
        return 0.0, zeros, 0.0, zeros, 0.0, 0.0, zeros, 0.0, 0.0

    recent = trace_history[-len(salience) :]
    sal_max = float(max(salience)) if salience else 0.0
    sim_max = 0.0

    current = dict(current_structure or {})
    if not current.get("active_component_id", None) and components:
        dom = max(components, key=lambda c: float(c.get("support_mass", 0.0)))
        current.update(
            {
                "active_component_id": dom.get("stable_id", dom.get("component_id")),
                "active_component_sign": dom.get("sign", 0),
                "active_component_width": dom.get("width", 0),
                "signed_run_count": current.get("signed_run_count", 0),
                "largest_signed_run_width": current.get("largest_signed_run_width", 0),
            }
        )

    for t, s in zip(recent, salience):
        sim = compute_trace_similarity(current, t, size=n)
        sim_max = max(sim_max, float(sim))
        s_norm, sim_norm = normalize_caution_inputs(salience=float(s), similarity=float(sim))
        weight_raw = float(salience_weight) * float(s_norm) + float(similarity_weight) * float(sim_norm)
        if weight_raw <= 0.0:
            continue
        # Localize to the current active component mask if available; else spread to all component masks.
        active_id = current.get("active_component_id", None)
        if active_id is not None and int(active_id) in component_masks:
            raw_field += weight_raw * component_masks[int(active_id)]
        else:
            for m in component_masks.values():
                raw_field += weight_raw * m

    raw_mx = float(np.max(raw_field)) if raw_field.size else 0.0
    if raw_mx > 0:
        bounded_field = np.clip(raw_field / raw_mx, 0.0, 1.0)
    else:
        bounded_field = np.zeros(n, dtype=float)

    raw_caution_scalar = float(np.mean(raw_field)) if raw_field.size else 0.0
    bounded_caution_scalar = float(np.mean(bounded_field)) if bounded_field.size else 0.0

    caution_release_scalar = compute_caution_release(
        salience_max=sal_max,
        similarity_max=sim_max,
        release_rate=float(caution_release_rate),
    )

    # Phase7: recovery/recontextualization only adjust the bounded (applied) caution, never raw memory.
    rec = _bounded(recovery_scalar)
    rex = _bounded(recontextualization_score)
    rr = max(0.0, float(recovery_rate))
    rw = max(0.0, float(recontextualization_weight))

    scaled = np.clip(bounded_field * (1.0 - rr * rec) * (1.0 - rw * rex), 0.0, 1.0)
    after = np.clip(scaled - float(caution_release_scalar), 0.0, 1.0)
    caution_after_recovery_scalar = float(np.mean(after)) if after.size else 0.0

    return (
        float(raw_caution_scalar),
        raw_field,
        _bounded(bounded_caution_scalar),
        bounded_field,
        float(caution_release_scalar),
        _bounded(caution_after_recovery_scalar),
        after,
        _bounded(sal_max),
        _bounded(sim_max),
    )
