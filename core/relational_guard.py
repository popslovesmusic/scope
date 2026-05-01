from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


def _as_vector(value) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr.reshape(-1)


def _safe_cosine(a: np.ndarray, b: np.ndarray) -> float:
    an = float(np.linalg.norm(a))
    bn = float(np.linalg.norm(b))
    if an <= 1e-12 or bn <= 1e-12:
        return 0.0
    return float(np.clip(float(np.dot(a, b)) / (an * bn), -1.0, 1.0))


def ternary_vector(value, *, threshold: float = 1e-6) -> np.ndarray:
    """Map analog values into a signed ternary vector: -1, 0, +1."""
    v = _as_vector(value)
    th = abs(float(threshold))
    return np.where(v > th, 1, np.where(v < -th, -1, 0)).astype(int)


@dataclass(frozen=True)
class RelationalGuardResult:
    alignment_score: float
    opposition_score: float
    mismatch_score: float
    presence_score: float
    dominance_score: float
    cosine: float
    collapse_risk: float
    recommended_action: str
    relation_class: str

    def to_dict(self) -> Dict[str, float | str]:
        return {
            "alignment_score": float(self.alignment_score),
            "opposition_score": float(self.opposition_score),
            "mismatch_score": float(self.mismatch_score),
            "presence_score": float(self.presence_score),
            "dominance_score": float(self.dominance_score),
            "cosine": float(self.cosine),
            "collapse_risk": float(self.collapse_risk),
            "recommended_action": str(self.recommended_action),
            "relation_class": str(self.relation_class),
        }


def relational_guard(
    current,
    trace,
    reference=0.0,
    *,
    threshold: float = 1e-6,
    overcoherence_threshold: float = 0.88,
) -> RelationalGuardResult:
    """
    Vector-first relational switch/guard.

    The guard compares two vectors after local reference conditioning. It emits
    vector-derived summary scores rather than treating the relation as one
    scalar equality.
    """
    a = _as_vector(current)
    b = _as_vector(trace)
    if a.shape != b.shape:
        raise ValueError("current/trace shape mismatch")

    g = _as_vector(reference)
    if g.size == 1:
        g = np.full_like(a, float(g[0]))
    elif g.shape != a.shape:
        raise ValueError("reference shape mismatch")

    a_p = a - g
    b_p = b - g
    a_t = ternary_vector(a_p, threshold=threshold)
    b_t = ternary_vector(b_p, threshold=threshold)

    active = (a_t != 0) | (b_t != 0)
    active_count = int(np.count_nonzero(active))
    if active_count == 0:
        return RelationalGuardResult(
            alignment_score=0.0,
            opposition_score=0.0,
            mismatch_score=0.0,
            presence_score=0.0,
            dominance_score=0.0,
            cosine=0.0,
            collapse_risk=0.0,
            recommended_action="suppress",
            relation_class="null",
        )

    same = active & (a_t == b_t) & (a_t != 0)
    opposed = active & (a_t == -b_t) & (a_t != 0) & (b_t != 0)
    a_only = active & (a_t != 0) & (b_t == 0)
    b_only = active & (a_t == 0) & (b_t != 0)

    alignment_score = float(np.count_nonzero(same)) / float(active_count)
    opposition_score = float(np.count_nonzero(opposed)) / float(active_count)
    mismatch_score = float(np.count_nonzero(a_t != b_t)) / float(active_count)
    presence_score = float(active_count) / float(a_t.size)

    mag_total = np.abs(a_p) + np.abs(b_p)
    mag_active = mag_total > max(abs(float(threshold)), 1e-12)
    if np.any(mag_active):
        dominance_score = float(np.mean(np.abs(np.abs(a_p[mag_active]) - np.abs(b_p[mag_active])) / (mag_total[mag_active] + 1e-12)))
    else:
        dominance_score = 0.0

    cosine = _safe_cosine(a_p, b_p)

    if alignment_score >= 0.70 and cosine >= 0.50:
        relation_class = "reinforce"
    elif opposition_score >= 0.50 or cosine <= -0.50:
        relation_class = "cancel_or_tension"
    elif np.count_nonzero(a_only) / float(active_count) >= 0.60:
        relation_class = "A_only"
    elif np.count_nonzero(b_only) / float(active_count) >= 0.60:
        relation_class = "B_only"
    else:
        relation_class = "mixed"

    overcoherence = (
        max(0.0, cosine)
        * alignment_score
        * presence_score
        * (1.0 - min(1.0, dominance_score))
    )
    collapse_risk = float(np.clip(overcoherence, 0.0, 1.0))

    if collapse_risk >= float(overcoherence_threshold):
        action = "reinforce_with_resistance"
    elif relation_class == "cancel_or_tension":
        action = "route_to_recovery"
    elif relation_class == "mixed":
        action = "hold_or_probe"
    elif relation_class in {"A_only", "B_only"}:
        action = "limited_propagation"
    elif relation_class == "reinforce":
        action = "allow"
    else:
        action = "suppress"

    return RelationalGuardResult(
        alignment_score=alignment_score,
        opposition_score=opposition_score,
        mismatch_score=mismatch_score,
        presence_score=presence_score,
        dominance_score=dominance_score,
        cosine=cosine,
        collapse_risk=collapse_risk,
        recommended_action=action,
        relation_class=relation_class,
    )
