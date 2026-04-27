from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _first_present(mapping: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def _clip01(value: float) -> float:
    try:
        v = float(value)
    except Exception:
        v = 0.0
    return float(max(0.0, min(1.0, v)))


@dataclass
class ProjectionPhaseSummary:
    phase: int
    operator: str
    shift: float
    blocked: int
    hits: int
    caution: float
    recovery: float
    hold: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phase": int(self.phase),
            "operator": str(self.operator),
            "shift": float(self.shift),
            "blocked": int(self.blocked),
            "hits": int(self.hits),
            "caution": float(self.caution),
            "recovery": float(self.recovery),
            "hold": bool(self.hold),
        }


@dataclass
class ProjectionPath:
    turn_id: int
    phases: List[ProjectionPhaseSummary]
    selected_class: Any
    practical_confidence: float
    structural_confidence: float
    misleading_positive: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn_id": int(self.turn_id),
            "phases": [p.to_dict() for p in self.phases],
            "selected_class": self.selected_class,
            "practical_confidence": float(self.practical_confidence),
            "structural_confidence": float(self.structural_confidence),
            "misleading_positive": bool(self.misleading_positive),
        }


def _counts(item: Dict[str, Any]) -> tuple[int, int]:
    hits = _first_present(item, "corridor_hits", default=None)
    blocks = _first_present(item, "corridor_blocks", default=None)
    if isinstance(hits, list):
        hit_count = int(len(hits))
    else:
        hit_count = int(_first_present(item, "corridor_hit_count", default=0) or 0)
    if isinstance(blocks, list):
        block_count = int(len(blocks))
    else:
        block_count = int(_first_present(item, "corridor_block_count", default=0) or 0)
    return int(max(0, hit_count)), int(max(0, block_count))


def build_projection_path(
    *,
    runtime_output: Dict[str, Any],
    practical_confidence: float,
    structural_confidence: float,
    misleading_positive: bool = False,
) -> Dict[str, Any]:
    """
    Projection-path recording for inspection: an easily readable per-phase summary
    derived from existing trace fields.
    """
    out = _as_dict(runtime_output)
    trace = [t for t in _as_list(out.get("trace", [])) if isinstance(t, dict)]

    phases: List[ProjectionPhaseSummary] = []
    for t in trace:
        it = _as_dict(t)
        hit_count, block_count = _counts(it)
        phases.append(
            ProjectionPhaseSummary(
                phase=int(_first_present(it, "phase", default=0) or 0),
                operator=str(_first_present(it, "selected_operator", "op", default="n/a")),
                shift=float(_first_present(it, "shift", "decision_shift", default=0.0) or 0.0),
                blocked=int(block_count),
                hits=int(hit_count),
                caution=float(_clip01(_first_present(it, "caution_after_recovery", "caution_scalar", "caution", default=0.0) or 0.0)),
                recovery=float(_clip01(_first_present(it, "recovery_scalar", "recovery", "rec", default=0.0) or 0.0)),
                hold=bool(_first_present(it, "hold_state", "hold", default=False)),
            )
        )

    output_block = _as_dict(out.get("output", {}))
    selected_class = _first_present(output_block, "selected_class", default=None)

    path = ProjectionPath(
        turn_id=int(out.get("turn_id", 0) or 0),
        phases=phases,
        selected_class=selected_class,
        practical_confidence=float(_clip01(practical_confidence)),
        structural_confidence=float(_clip01(structural_confidence)),
        misleading_positive=bool(misleading_positive),
    )
    return path.to_dict()

