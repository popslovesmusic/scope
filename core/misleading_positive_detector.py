from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


def _clip(value: float, lo: float, hi: float) -> float:
    try:
        v = float(value)
    except Exception:
        v = 0.0
    return float(max(lo, min(hi, v)))


@dataclass
class MisleadingPositive:
    flagged: bool
    practical_confidence: float
    structural_confidence: float
    gap: float
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "flagged": bool(self.flagged),
            "practical_confidence": float(self.practical_confidence),
            "structural_confidence": float(self.structural_confidence),
            "gap": float(self.gap),
            "reason": str(self.reason),
        }


def detect_misleading_positive(
    *,
    practical_confidence: float,
    structural_confidence: float,
    practical_high: float = 0.60,
    structural_low: float = 0.45,
    gap_min: float = 0.20,
) -> MisleadingPositive:
    """
    Flags "misleading positive" when practical confidence is high but structural confidence is low.
    """
    p = float(_clip(practical_confidence, 0.0, 1.0))
    s = float(_clip(structural_confidence, 0.0, 1.0))
    gap = float(_clip(p - s, -1.0, 1.0))

    flagged = bool(p >= practical_high and s <= structural_low and gap >= gap_min)
    reason = ""
    if flagged:
        reason = f"high_practical_low_structural(p>={practical_high:.2f}, s<={structural_low:.2f}, gap>={gap_min:.2f})"

    return MisleadingPositive(
        flagged=flagged,
        practical_confidence=p,
        structural_confidence=s,
        gap=gap,
        reason=reason,
    )

