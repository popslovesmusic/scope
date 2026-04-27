from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _first_present(mapping: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def _clip(value: float, lo: float, hi: float) -> float:
    try:
        v = float(value)
    except Exception:
        v = 0.0
    return float(max(lo, min(hi, v)))


@dataclass
class ShadowAdmissibility:
    score: float
    blocked_fraction: float
    block_count: int
    hit_count: int
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": float(self.score),
            "blocked_fraction": float(self.blocked_fraction),
            "block_count": int(self.block_count),
            "hit_count": int(self.hit_count),
            "reason": str(self.reason),
        }


def _counts_from_trace_item(item: Dict[str, Any]) -> Tuple[int, int]:
    it = _as_dict(item)
    hits = _first_present(it, "corridor_hits", default=None)
    blocks = _first_present(it, "corridor_blocks", default=None)
    if isinstance(hits, list):
        hit_count = int(len(hits))
    else:
        hit_count = int(_first_present(it, "corridor_hit_count", default=0) or 0)
    if isinstance(blocks, list):
        block_count = int(len(blocks))
    else:
        block_count = int(_first_present(it, "corridor_block_count", default=0) or 0)
    return int(max(0, hit_count)), int(max(0, block_count))


def score_shadow_admissibility_for_phase(
    *,
    trace_item: Dict[str, Any],
    blocked_fraction_soft_floor: float = 0.0,
) -> ShadowAdmissibility:
    """
    Shadow-mode admissibility: a bounded score derived from corridor hits/blocks.
    Does not enforce blocking; only records validity.
    """
    hit_count, block_count = _counts_from_trace_item(trace_item)
    denom = float(max(1, hit_count + block_count))
    blocked_fraction = float(block_count) / denom
    blocked_fraction = float(max(float(blocked_fraction_soft_floor), blocked_fraction))
    score = float(_clip(1.0 - blocked_fraction, 0.0, 1.0))

    reason = ""
    if block_count > 0 and hit_count == 0:
        reason = "all_blocked"
    elif blocked_fraction >= 0.50:
        reason = "mostly_blocked"
    elif blocked_fraction >= 0.20:
        reason = "partially_blocked"

    return ShadowAdmissibility(
        score=score,
        blocked_fraction=float(_clip(blocked_fraction, 0.0, 1.0)),
        block_count=int(block_count),
        hit_count=int(hit_count),
        reason=reason,
    )


def summarize_shadow_admissibility(
    *,
    trace: Sequence[Dict[str, Any]],
    min_phases: int = 1,
) -> Dict[str, Any]:
    items = [t for t in _as_list(list(trace)) if isinstance(t, dict)]
    if len(items) < int(min_phases):
        return {
            "phase_scores": [],
            "final_score": 0.0,
            "min_score": 0.0,
            "mean_score": 0.0,
        }

    phase_scores: List[float] = []
    phase_details: List[Dict[str, Any]] = []
    for t in items:
        s = score_shadow_admissibility_for_phase(trace_item=t)
        phase_scores.append(float(s.score))
        phase_details.append(dict({"phase": _first_present(_as_dict(t), "phase", default=None)}, **s.to_dict()))

    final_score = float(phase_scores[-1]) if phase_scores else 0.0
    min_score = float(min(phase_scores)) if phase_scores else 0.0
    mean_score = float(sum(phase_scores) / float(max(1, len(phase_scores)))) if phase_scores else 0.0
    return {
        "phase_scores": phase_scores,
        "phase_details": phase_details,
        "final_score": float(_clip(final_score, 0.0, 1.0)),
        "min_score": float(_clip(min_score, 0.0, 1.0)),
        "mean_score": float(_clip(mean_score, 0.0, 1.0)),
    }

