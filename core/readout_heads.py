
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np


def binary_readout(
    *,
    amplitude: np.ndarray,
    corridor_window: np.ndarray,
    class_heads: List[Dict[str, Any]],
) -> Tuple[List[float], int, float]:
    amp = np.asarray(amplitude, dtype=float)
    win = np.asarray(corridor_window, dtype=float)
    if amp.shape != win.shape:
        raise ValueError("amplitude/corridor_window shape mismatch")

    class_scores: List[float] = []
    for head in class_heads:
        fam = head.get("readout_family", [])
        score = float(np.sum(amp[fam] * win[fam])) if len(fam) else 0.0
        class_scores.append(score)

    if not class_scores:
        return [], 0, 0.0

    selected = int(np.argmax(class_scores))
    total = float(np.sum(class_scores)) + 1e-9
    # Simple, bounded confidence: margin over total energy.
    ordered = sorted(class_scores, reverse=True)
    margin = float(ordered[0] - (ordered[1] if len(ordered) > 1 else 0.0))
    confidence = float(np.clip(margin / total, 0.0, 1.0))
    return class_scores, selected, confidence
