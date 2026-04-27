
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class OrientationOperator:
    name: str
    read: Callable[[Array], Array]
    write: Callable[[Array], Array]  # inverse of read for stage-1
    phase_shift: float
    symmetry_class: str


def _op_pp(x: Array) -> Array:
    return np.asarray(x, dtype=float)


def _op_mm(x: Array) -> Array:
    return np.asarray(x, dtype=float)[::-1]


def _op_pm(x: Array) -> Array:
    return np.roll(np.asarray(x, dtype=float), 1)


def _op_mp(x: Array) -> Array:
    return np.roll(np.asarray(x, dtype=float), -1)


OPERATORS: List[OrientationOperator] = [
    OrientationOperator(name="++", read=_op_pp, write=_op_pp, phase_shift=0.0, symmetry_class="paired"),
    OrientationOperator(name="--", read=_op_mm, write=_op_mm, phase_shift=np.pi, symmetry_class="paired"),
    OrientationOperator(name="+-", read=_op_pm, write=_op_mp, phase_shift=0.5 * np.pi, symmetry_class="paired"),
    OrientationOperator(name="-+", read=_op_mp, write=_op_pm, phase_shift=-0.5 * np.pi, symmetry_class="paired"),
]

OPERATORS_BY_NAME: Dict[str, OrientationOperator] = {op.name: op for op in OPERATORS}


def apply_operator_to_masked_region(field: Array, operator: OrientationOperator, mask: Array) -> Array:
    x = np.asarray(field, dtype=float)
    m = np.asarray(mask, dtype=float)
    if x.shape != m.shape:
        raise ValueError("field/mask shape mismatch")
    transformed = operator.read(x)
    out = x.copy()
    sel = m > 0.5
    out[sel] = transformed[sel]
    return out
