from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _first(mapping: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    for k in keys:
        if k in mapping and mapping[k] is not None:
            return mapping[k]
    return default


def _clip(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return float(lo)
    if x > hi:
        return float(hi)
    return float(x)


def _clip01(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return 0.0
    return _clip(v, 0.0, 1.0)


def _mean(xs: Sequence[float]) -> float:
    if not xs:
        return 0.0
    return float(sum(xs)) / float(len(xs))


def _std(xs: Sequence[float]) -> float:
    if len(xs) <= 1:
        return 0.0
    m = _mean([float(x) for x in xs])
    v = _mean([(float(x) - m) ** 2 for x in xs])
    return float(v**0.5)


def classify_intent(prompt_text: str) -> str:
    """
    Deterministic intent classifier for residue labeling.
    """
    t = (prompt_text or "").strip().lower()
    if not t:
        return "unknown"
    if t in {"hello", "hi", "hey", "yo"}:
        return "greeting"
    if re.search(r"\bhow\s+are\s+you\b", t) or re.search(r"\bwhat('?s| is)\s+up\b", t):
        return "greeting"
    if re.match(r"^(explain|describe|give|show|teach)\b", t):
        return "explanation_request"
    if re.match(r"^(run|set|turn|toggle|enable|disable)\b", t):
        return "instruction"
    if t.endswith("?") or re.match(r"^(how|why|what|when|where)\b", t):
        return "question"
    if t in {"thanks", "thank you", "thx"}:
        return "meta"
    if re.search(r"\b(system status|status|engine|sbllm|trace|debug|memory)\b", t):
        return "meta"
    return "unknown"


def detect_structured_input(prompt_text: str, *, mode: str = "auto") -> bool:
    """
    Lightweight structured-input detector for stricter qualification + longer persistence.

    mode: off | on | auto
    """
    m = (mode or "auto").strip().lower()
    if m in {"off", "false", "0"}:
        return False
    if m in {"on", "true", "1"}:
        return True

    text = (prompt_text or "").strip()
    if not text:
        return False
    if text.startswith(("{", "[")) and len(text) > 20:
        return True
    if re.search(r"^\s*(osl|tanka)\s*:", text, flags=re.IGNORECASE | re.MULTILINE):
        return True

    # Heuristic: many short lines looks like a structured artifact (poem/form/template).
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) >= 4 and all(len(ln) <= 80 for ln in lines):
        return True
    return False


def build_state_summary(*, operator: str, caution: float, recovery: float, hold: bool) -> str:
    op = str(operator)
    alignment = {
        "++": "aligned",
        "+-": "mixed",
        "--": "slightly opposed",
        "-+": "unsettled",
    }.get(op, "unclear")

    if hold:
        stability = "holding steady"
    elif caution < 0.3:
        stability = "stable"
    elif caution <= 0.6:
        stability = "slightly cautious"
    else:
        stability = "guarded"

    if recovery >= 0.5:
        adapt = "recovering"
    elif recovery >= 0.1:
        adapt = "adapting"
    else:
        adapt = ""

    return f"{stability} and {alignment}, {adapt}".rstrip(", ").strip()


@dataclass
class TurnResidue:
    """
    Raw residue object (always logged), later tagged as qualified/committed.
    """

    turn_id: int
    prompt_text: str
    intent_category: str
    reply_mode: str
    selected_class: int
    confidence: float
    active_component_id: Any
    operator_path: List[str]
    dominant_operator: str
    phase_pattern: List[str]
    caution_profile: Dict[str, float]
    recovery_profile: Dict[str, float]
    caution_series: List[float]
    recovery_series: List[float]
    hold_count: int
    state_summary: str
    component_count: int
    carry_weight: float
    ratchet: bool = False
    stability_score: float = 0.0
    is_qualified: bool = False
    is_committed: bool = False
    commit_reason: str = ""
    reject_reason: str = ""
    persistence_duration: int = 0
    structured_input: bool = False
    operator_histogram: Dict[str, int] = field(default_factory=dict)
    switch_fraction: float = 0.0
    caution_terminal: float = 0.0
    hold_terminal: bool = False
    recovery_terminal: float = 0.0
    admissibility_score: float = 1.0
    structural_confidence: float = 0.0
    practical_confidence: float = 0.0
    misleading_positive: bool = False
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn_id": int(self.turn_id),
            "prompt_text": str(self.prompt_text),
            "intent_category": str(self.intent_category),
            "intent_class": str(self.intent_category),
            "reply_mode": str(self.reply_mode),
            "selected_class": int(self.selected_class),
            "confidence": float(self.confidence),
            "active_component_id": self.active_component_id,
            "operator_path": list(self.operator_path),
            "dominant_operator": str(self.dominant_operator),
            "phase_pattern": list(self.phase_pattern),
            "caution": float(self.caution_profile.get("end", 0.0)),
            "recovery": float(self.recovery_profile.get("end", 0.0)),
            "operator_histogram": dict(self.operator_histogram),
            "switch_fraction": float(self.switch_fraction),
            "caution_terminal": float(self.caution_terminal),
            "hold_terminal": bool(self.hold_terminal),
            "recovery_terminal": float(self.recovery_terminal),
            "admissibility_score": float(self.admissibility_score),
            "structural_confidence": float(self.structural_confidence),
            "practical_confidence": float(self.practical_confidence),
            "misleading_positive": bool(self.misleading_positive),
            "caution_profile": dict(self.caution_profile),
            "recovery_profile": dict(self.recovery_profile),
            "hold_count": int(self.hold_count),
            "ratchet": bool(self.ratchet),
            "stability_score": float(self.stability_score),
            "is_qualified": bool(self.is_qualified),
            "is_committed": bool(self.is_committed),
            "commit_reason": str(self.commit_reason),
            "reject_reason": str(self.reject_reason),
            "persistence_duration": int(self.persistence_duration),
            "structured_input": bool(self.structured_input),
            "state_summary": str(self.state_summary),
            "component_count": int(self.component_count),
            "carry_weight": float(self.carry_weight),
            "created_at": str(self.created_at),
        }


@dataclass
class CommittedResidue:
    """
    Low-bandwidth committed continuation object (memory only stores these).
    """

    key: str
    dominant_operator: str
    phase_pattern: List[str]
    persistence_duration: int
    last_turn_id: int
    avg_stability_score: float
    caution_end: float
    caution_peak: float
    recovery_end: float
    recovery_peak: float
    structured_input: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": str(self.key),
            "dominant_operator": str(self.dominant_operator),
            "phase_pattern": list(self.phase_pattern),
            "persistence_duration": int(self.persistence_duration),
            "last_turn_id": int(self.last_turn_id),
            "avg_stability_score": float(self.avg_stability_score),
            "caution_end": float(self.caution_end),
            "caution_peak": float(self.caution_peak),
            "recovery_end": float(self.recovery_end),
            "recovery_peak": float(self.recovery_peak),
            "structured_input": bool(self.structured_input),
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "CommittedResidue":
        return CommittedResidue(
            key=str(d.get("key", "")),
            dominant_operator=str(d.get("dominant_operator", "n/a")),
            phase_pattern=list(d.get("phase_pattern", []) or []),
            persistence_duration=int(d.get("persistence_duration", 0)),
            last_turn_id=int(d.get("last_turn_id", 0)),
            avg_stability_score=float(d.get("avg_stability_score", 0.0)),
            caution_end=float(d.get("caution_end", 0.0)),
            caution_peak=float(d.get("caution_peak", 0.0)),
            recovery_end=float(d.get("recovery_end", 0.0)),
            recovery_peak=float(d.get("recovery_peak", 0.0)),
            structured_input=bool(d.get("structured_input", False)),
        )


@dataclass
class CommittedResidueRecord:
    """
    Append-only committed residue record for auditing persistence decisions (not trace).
    """

    turn_id: int
    commit_reasons: List[str]
    operator_bias_delta: Dict[str, float]
    caution_baseline_delta: float
    persistence_duration: int
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn_id": int(self.turn_id),
            "commit_reasons": [str(x) for x in (self.commit_reasons or [])],
            "operator_bias_delta": {str(k): float(v) for k, v in (self.operator_bias_delta or {}).items()},
            "caution_baseline_delta": float(self.caution_baseline_delta),
            "persistence_duration": int(self.persistence_duration),
            "created_at": str(self.created_at),

        }


@dataclass
class PersistentMemoryState:
    """
    Memory is continuation bias. It stores only committed residues + derived priors.
    """

    turn_counter: int = 0
    committed: List[CommittedResidue] = field(default_factory=list)

    # Derived priors (computed from committed residues).
    operator_bias: Dict[str, float] = field(default_factory=lambda: {"++": 0.0, "--": 0.0, "+-": 0.0, "-+": 0.0})
    caution_baseline_shift: float = 0.0
    recovery_baseline_shift: float = 0.0

    # Diagnostics / accounting.
    qualified_residue_count: int = 0
    committed_residue_count: int = 0
    rejected_residue_count: int = 0
    average_stability_score: float = 0.0
    last_commit: bool = False
    last_commit_reason: str = ""
    last_reject_reason: str = ""
    last_state_summary: str = ""
    last_active_component_id: Any = None
    
    # Patch 20: Persistent traversal success
    successful_traversals: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn_counter": int(self.turn_counter),
            "committed": [c.to_dict() for c in self.committed],
            "operator_bias": dict(self.operator_bias),
            "caution_baseline_shift": float(self.caution_baseline_shift),
            "recovery_baseline_shift": float(self.recovery_baseline_shift),
            "qualified_residue_count": int(self.qualified_residue_count),
            "committed_residue_count": int(self.committed_residue_count),
            "rejected_residue_count": int(self.rejected_residue_count),
            "average_stability_score": float(self.average_stability_score),
            "last_commit": bool(self.last_commit),
            "last_commit_reason": str(self.last_commit_reason),
            "last_reject_reason": str(self.last_reject_reason),
            "last_state_summary": str(self.last_state_summary),
            "last_active_component_id": self.last_active_component_id,
            "successful_traversals": int(self.successful_traversals),
            "saved_at": datetime.utcnow().isoformat() + "Z",
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "PersistentMemoryState":
        s = PersistentMemoryState()
        s.turn_counter = int(d.get("turn_counter", 0))

        committed_raw = d.get("committed", None)
        if isinstance(committed_raw, list):
            s.committed = [CommittedResidue.from_dict(_as_dict(x)) for x in committed_raw if isinstance(x, dict)]

        # Backward compatibility: if old memory files had only operator_bias/caution shift, keep them.
        if isinstance(d.get("operator_bias", None), dict) and not s.committed:
            s.operator_bias = dict(d.get("operator_bias", s.operator_bias) or s.operator_bias)
        if "caution_baseline_shift" in d:
            s.caution_baseline_shift = float(d.get("caution_baseline_shift", 0.0))
        if "recovery_baseline_shift" in d:
            s.recovery_baseline_shift = float(d.get("recovery_baseline_shift", 0.0))

        s.qualified_residue_count = int(d.get("qualified_residue_count", 0))
        s.committed_residue_count = int(d.get("committed_residue_count", 0))
        s.rejected_residue_count = int(d.get("rejected_residue_count", 0))
        s.average_stability_score = float(d.get("average_stability_score", 0.0))
        s.last_commit = bool(d.get("last_commit", False))
        s.last_commit_reason = str(d.get("last_commit_reason", ""))
        s.last_reject_reason = str(d.get("last_reject_reason", ""))
        s.last_state_summary = str(d.get("last_state_summary", ""))
        s.last_active_component_id = d.get("last_active_component_id", None)
        s.successful_traversals = int(d.get("successful_traversals", 0))
        return s


def load_memory_state(path: str) -> PersistentMemoryState:
    p = Path(path)
    if not p.exists():
        return PersistentMemoryState()
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
        return PersistentMemoryState.from_dict(_as_dict(payload))
    except Exception:
        return PersistentMemoryState()


def save_memory_state(path: str, state: PersistentMemoryState) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_turn_residue(path: str, residue: TurnResidue) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(residue.to_dict(), sort_keys=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def append_committed_residue_record(path: str, record: CommittedResidueRecord) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record.to_dict(), sort_keys=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _series_from_trace(trace: List[Dict[str, Any]]) -> Tuple[List[float], List[float], List[str], int]:
    if not trace:
        return [], [], [], 0
    cautions = [_clip01(_first(t, "caution", "caution_scalar", "caution_after_recovery", default=0.0)) for t in trace]
    recoveries = [_clip01(_first(t, "recovery", "recovery_scalar", "rec", default=0.0)) for t in trace]
    ops = [str(_first(t, "selected_operator", "op", default="n/a")) for t in trace]
    holds = [bool(_first(t, "hold_state", "hold", default=False)) for t in trace]
    hold_count = int(sum(1 for h in holds if h))
    return cautions, recoveries, ops, hold_count


def _operator_histogram(ops: Sequence[str]) -> Dict[str, int]:
    out: Dict[str, int] = {"++": 0, "--": 0, "+-": 0, "-+": 0}
    for op in ops or []:
        k = str(op)
        out[k] = int(out.get(k, 0)) + 1
    return out


def _switch_fraction(ops: Sequence[str]) -> float:
    xs = [str(x) for x in (ops or [])]
    if len(xs) < 2:
        return 0.0
    switches = sum(1 for i in range(1, len(xs)) if xs[i] != xs[i - 1])
    return float(switches) / float(max(1, len(xs) - 1))


def _profiles(series: List[float]) -> Dict[str, float]:
    if not series:
        return {"start": 0.0, "peak": 0.0, "end": 0.0}
    return {"start": float(series[0]), "peak": float(max(series)), "end": float(series[-1])}


def compute_carry_weight(
    *,
    caution_peak: float,
    recovery_peak: float,
    hold_count: int,
    confidence: float,
    intent_category: str,
    component_persistence_strength: float = 0.0,
    admissibility_score: float = 1.0,
) -> float:
    """
    Persistence relevance (not truth scoring). Output clamped to [0,1].
    """
    c = _clip01(caution_peak)
    r = _clip01(recovery_peak)
    h = _clip(float(hold_count) / 3.0, 0.0, 1.0)
    conf = _clip01(confidence)
    cps = _clip01(component_persistence_strength)

    intent_importance = {
        "greeting": 0.10,
        "meta": 0.15,
        "unknown": 0.25,
        "instruction": 0.35,
        "question": 0.50,
        "explanation_request": 0.55,
    }.get(intent_category, 0.25)

    w = 0.30 * c + 0.20 * r + 0.20 * h + 0.15 * cps + 0.10 * conf + 0.05 * intent_importance
    adm = _clip01(admissibility_score)
    # Mild modulation: if admissibility is low, reduce how strongly the turn should persist.
    w *= float(0.65 + 0.35 * float(adm))
    return _clip(float(w), 0.0, 1.0)


def build_turn_residue(
    *,
    runtime_output: Dict[str, Any],
    prompt_text: str,
    intent_category: str,
    reply_mode: str,
    turn_id: int,
    structured_input: bool = False,
) -> TurnResidue:
    state = _as_dict(_first(runtime_output, "state", default={}))
    signature = _as_dict(_first(state, "signature", default={}))
    orientation = _as_dict(_first(state, "orientation", default={}))
    out = _as_dict(_first(runtime_output, "output", default={}))
    trace = [t for t in _as_list(_first(runtime_output, "trace", default=[])) if isinstance(t, dict)]
    shadow = _as_dict(_first(runtime_output, "admissibility_shadow", default={}))
    mp = _as_dict(_first(runtime_output, "misleading_positive", default={}))

    cautions, recoveries, operator_path, hold_count = _series_from_trace(trace)
    caution_profile = _profiles(cautions)
    recovery_profile = _profiles(recoveries)
    operator_hist = _operator_histogram(operator_path)
    switch_frac = _switch_fraction(operator_path)

    dominant_operator = operator_path[-1] if operator_path else str(_first(orientation, "active_operator", default="n/a"))
    selected_class = int(_first(out, "selected_class", default=0) or 0)
    confidence = float(_first(out, "confidence", default=0.0) or 0.0)

    operator = str(_first(orientation, "active_operator", default=dominant_operator))
    caution_end = float(_clip01(_first(signature, "caution_scalar", default=caution_profile["end"])))
    recovery_end = float(_clip01(_first(signature, "recovery_scalar", default=recovery_profile["end"])))
    hold = bool(_first(signature, "hold_state", default=False))
    state_summary = build_state_summary(operator=operator, caution=caution_end, recovery=recovery_end, hold=hold)

    component_count = len(_as_list(_first(signature, "components", default=[])) or [])
    active_component_id = _first(signature, "active_component_id", default=None)
    component_persistence_strength = 0.0
    if trace:
        component_persistence_strength = float(_clip01(_first(trace[-1], "component_identity_persistence", default=0.0)))

    practical_conf = float(_first(out, "practical_confidence", default=_first(out, "confidence", default=0.0)) or 0.0)
    structural_conf = float(_first(out, "structural_confidence", default=0.0) or 0.0)
    adm_score = float(_first(shadow, "final_score", default=1.0) or 1.0)
    mp_flag = bool(_first(mp, "flagged", default=False))

    carry_weight = compute_carry_weight(
        caution_peak=float(caution_profile["peak"]),
        recovery_peak=float(recovery_profile["peak"]),
        hold_count=int(hold_count),
        confidence=float(practical_conf),
        intent_category=str(intent_category),
        component_persistence_strength=float(component_persistence_strength),
        admissibility_score=float(adm_score),
    )

    return TurnResidue(

        turn_id=int(turn_id),
        prompt_text=str(prompt_text),
        intent_category=str(intent_category),
        reply_mode=str(reply_mode),
        selected_class=int(selected_class),
        confidence=float(confidence),
        active_component_id=active_component_id,
        operator_path=list(operator_path),
        dominant_operator=str(dominant_operator),
        phase_pattern=list(operator_path),
        caution_profile=dict(caution_profile),
        recovery_profile=dict(recovery_profile),
        caution_series=[float(x) for x in cautions],
        recovery_series=[float(x) for x in recoveries],
        hold_count=int(hold_count),
        state_summary=str(state_summary),
        component_count=int(component_count),
        carry_weight=float(carry_weight),
        structured_input=bool(structured_input),
        operator_histogram=dict(operator_hist),
        switch_fraction=float(switch_frac),
        caution_terminal=float(caution_profile.get("end", 0.0)),
        hold_terminal=bool(hold),
        recovery_terminal=float(recovery_profile.get("end", 0.0)),
        admissibility_score=float(_clip01(adm_score)),
        structural_confidence=float(_clip01(structural_conf)),
        practical_confidence=float(_clip01(practical_conf)),
        misleading_positive=bool(mp_flag),
    )


def evaluate_stability(
    residue: TurnResidue,
    *,
    structured_input: bool = False,
    epsilon: float = 0.10,
    recovery_threshold: float = 0.02,
    max_switch_freq: float = 0.50,
    min_score: float = 0.55,
    min_admissibility: float = 0.0,
) -> Tuple[float, bool, bool, str]:
    """
    Returns: (stability_score [0..1], is_qualified, ratchet, reason)
    """
    def _drop_warmup_zeroes(series: Sequence[float]) -> List[float]:
        xs = [float(x) for x in (series or [])]
        if len(xs) >= 3 and float(xs[0]) == 0.0 and max(xs[1:]) > 0.0:
            return xs[1:]
        return xs

    ops = residue.operator_path or residue.phase_pattern or [residue.dominant_operator]
    cautions = _drop_warmup_zeroes(residue.caution_series or [float(residue.caution_profile.get("end", 0.0))])
    recoveries = _drop_warmup_zeroes(residue.recovery_series or [float(residue.recovery_profile.get("end", 0.0))])

    switches = sum(1 for i in range(1, len(ops)) if ops[i] != ops[i - 1])
    switch_freq = float(switches) / float(max(1, len(ops) - 1))
    counts: Dict[str, int] = {}
    for op in ops:
        counts[str(op)] = counts.get(str(op), 0) + 1
    dominant_frac = float(max(counts.values())) / float(max(1, len(ops)))
    closure = 1.0 if (len(ops) >= 2 and ops[0] == ops[-1]) else 0.0
    coherence = _clip(0.6 * dominant_frac + 0.4 * closure, 0.0, 1.0)

    tail_delta = 0.0
    if len(cautions) >= 2:
        tail_delta = float(cautions[-1]) - float(cautions[-2])
    caution_converged = abs(float(tail_delta)) < float(epsilon)
    turbulence = _std([float(x) for x in cautions])
    low_turb = _clip(1.0 - _clip(turbulence / 0.20, 0.0, 1.0), 0.0, 1.0)

    recovery_peak = float(max(recoveries)) if recoveries else float(residue.recovery_profile.get("peak", 0.0))
    recovery_present = float(recovery_peak) > float(recovery_threshold)
    rec_score = _clip(float(recovery_peak) / 0.20, 0.0, 1.0)

    adm_score = float(_clip01(getattr(residue, "admissibility_score", 1.0)))

    # Ratchet: stable non-hold completion where caution is settling and recovery exists.
    # Permit early-phase operator exploration as long as the overall path is coherent (closure or dominant operator).
    ratchet = bool(residue.hold_count == 0 and recovery_present and caution_converged and float(coherence) >= 0.60)

    conv_score = _clip(1.0 - _clip(abs(float(tail_delta)) / float(max(1e-6, epsilon)), 0.0, 1.0), 0.0, 1.0)
    ratchet_score = 1.0 if ratchet else 0.0

    stability_score = _clip(
        0.33 * coherence + 0.24 * low_turb + 0.20 * conv_score + 0.10 * rec_score + 0.10 * ratchet_score + 0.03 * adm_score,
        0.0,
        1.0,
    )

    # Structured mode is stricter.
    if structured_input:
        min_score = max(float(min_score), 0.70)
        max_switch_freq = min(float(max_switch_freq), 0.35)

    if structured_input:
        if switch_freq > float(max_switch_freq):
            return float(stability_score), False, ratchet, "rapid_operator_flipping"
    else:
        if switch_freq > float(max_switch_freq) and float(closure) == 0.0 and float(dominant_frac) < 0.50:
            return float(stability_score), False, ratchet, "rapid_operator_flipping"
    if not recovery_present:
        return float(stability_score), False, ratchet, "low_recovery"
    if not caution_converged:
        return float(stability_score), False, ratchet, "caution_not_converged"
    if not ratchet:
        return float(stability_score), False, ratchet, "ratchet_required"
    if float(adm_score) < float(min_admissibility):
        return float(stability_score), False, ratchet, "low_admissibility"
    if float(stability_score) < float(min_score):
        return float(stability_score), False, ratchet, "stability_score_below_threshold"
    return float(stability_score), True, ratchet, "qualified_stable_ratchet"


def qualify_residue(
    residue: TurnResidue,
    *,
    structured_input: bool,
    epsilon: float,
    recovery_threshold: float,
    max_switch_freq: float,
    min_score: float,
    min_admissibility: float = 0.0,
) -> TurnResidue:
    score, qualified, ratchet, reason = evaluate_stability(
        residue,
        structured_input=structured_input,
        epsilon=epsilon,
        recovery_threshold=recovery_threshold,
        max_switch_freq=max_switch_freq,
        min_score=min_score,
        min_admissibility=min_admissibility,
    )
    residue.structured_input = bool(structured_input)
    residue.stability_score = float(score)
    residue.is_qualified = bool(qualified)
    residue.ratchet = bool(ratchet)
    if qualified:
        residue.commit_reason = str(reason)
        residue.reject_reason = ""
    else:
        residue.commit_reason = ""
        residue.reject_reason = str(reason)
    return residue


def _residue_key(dominant_operator: str, phase_pattern: Sequence[str]) -> str:
    pat = "-".join([str(x) for x in phase_pattern]) if phase_pattern else str(dominant_operator)
    return f"{dominant_operator}:{pat}"


def _decay_committed(committed: List[CommittedResidue]) -> List[CommittedResidue]:
    out: List[CommittedResidue] = []
    for c in committed:
        c.persistence_duration = int(c.persistence_duration) - 1
        if int(c.persistence_duration) > 0:
            out.append(c)
    return out


def _recompute_biases(state: PersistentMemoryState) -> None:
    # Operator bias from committed durations.
    sums = {"++": 0.0, "--": 0.0, "+-": 0.0, "-+": 0.0}
    for c in state.committed:
        op = str(c.dominant_operator)
        if op in sums:
            sums[op] += float(max(0, int(c.persistence_duration)))
    total = sum(sums.values())
    if total > 0.0:
        state.operator_bias = {k: float(v / total) for k, v in sums.items()}
    else:
        state.operator_bias = {k: 0.0 for k in sums.keys()}

    # Caution baseline shift from committed residues (mean of targets).
    targets: List[float] = []
    for c in state.committed:
        target = (0.6 * float(c.caution_end) + 0.4 * float(c.caution_peak)) - 0.30
        targets.append(float(_clip(target, -1.0, 1.0)))
    state.caution_baseline_shift = float(_clip(_mean(targets), -0.25, 0.25)) if targets else 0.0


def apply_commit_gate_and_persistence(
    *,
    state: PersistentMemoryState,
    residue: TurnResidue,
    base_duration: int,
    reinforce: int,
    max_duration: int,
) -> Tuple[PersistentMemoryState, TurnResidue]:
    """
    - Always decays existing committed residues.
    - Commits only if residue.is_qualified == True.
    """
    state.last_commit = False
    state.last_commit_reason = ""
    state.last_reject_reason = ""

    state.turn_counter = int(state.turn_counter) + 1
    state.committed = _decay_committed(state.committed)

    if residue.is_qualified:
        residue.is_committed = True
        state.qualified_residue_count = int(state.qualified_residue_count) + 1

        key = _residue_key(residue.dominant_operator, residue.phase_pattern)
        found = None
        for c in state.committed:
            if c.key == key:
                found = c
                break

        initial = int(base_duration)
        if residue.structured_input:
            initial = max(initial, int(base_duration) + 2)

        if found is None:
            c = CommittedResidue(
                key=key,
                dominant_operator=str(residue.dominant_operator),
                phase_pattern=list(residue.phase_pattern),
                persistence_duration=int(_clip(initial, 1, int(max_duration))),
                last_turn_id=int(residue.turn_id),
                avg_stability_score=float(residue.stability_score),
                caution_end=float(residue.caution_profile.get("end", 0.0)),
                caution_peak=float(residue.caution_profile.get("peak", 0.0)),
                recovery_end=float(residue.recovery_profile.get("end", 0.0)),
                recovery_peak=float(residue.recovery_profile.get("peak", 0.0)),
                structured_input=bool(residue.structured_input),
            )
            state.committed.append(c)
        else:
            inc = int(reinforce)
            if residue.structured_input:
                inc = max(inc, int(reinforce) + 1)
            found.persistence_duration = int(_clip(int(found.persistence_duration) + int(inc), 1, int(max_duration)))
            found.last_turn_id = int(residue.turn_id)
            found.avg_stability_score = float(0.7 * float(found.avg_stability_score) + 0.3 * float(residue.stability_score))
            found.caution_end = float(residue.caution_profile.get("end", found.caution_end))

            found.caution_peak = float(max(found.caution_peak, float(residue.caution_profile.get("peak", found.caution_peak))))
            found.recovery_end = float(residue.recovery_profile.get("end", found.recovery_end))
            found.recovery_peak = float(max(found.recovery_peak, float(residue.recovery_profile.get("peak", found.recovery_peak))))
            found.structured_input = bool(found.structured_input or residue.structured_input)

        residue.persistence_duration = int(initial if found is None else found.persistence_duration)

        state.committed_residue_count = int(len(state.committed))
        state.last_commit = True
        state.last_commit_reason = str(residue.commit_reason or "committed")
    else:
        residue.is_committed = False
        residue.persistence_duration = 0
        state.rejected_residue_count = int(state.rejected_residue_count) + 1
        state.last_reject_reason = str(residue.reject_reason or "rejected")

    # Update diagnostics and derived priors.
    if state.committed:
        state.average_stability_score = float(_mean([float(c.avg_stability_score) for c in state.committed]))
    else:
        state.average_stability_score = 0.0
    _recompute_biases(state)

    state.last_state_summary = str(residue.state_summary)
    state.last_active_component_id = residue.active_component_id
    return state, residue


def memory_panel_text(state: PersistentMemoryState) -> str:
    ob = state.operator_bias
    items = ", ".join([f"{k}:{float(ob.get(k, 0.0)):.3f}" for k in ["++", "--", "+-", "-+"]])
    return "\n".join(
        [
            "Memory:",
            f"turn_counter={int(state.turn_counter)}",
            f"committed_residue_count={int(state.committed_residue_count)} qualified_residue_count={int(state.qualified_residue_count)} rejected_residue_count={int(state.rejected_residue_count)}",
            f"average_stability_score={float(state.average_stability_score):.3f}",
            f"operator_bias={{" + items + "}}",
            f"caution_baseline_shift={float(state.caution_baseline_shift):.3f}",
            f"last_commit={bool(state.last_commit)} commit_reason={state.last_commit_reason}",
            f"last_reject_reason={state.last_reject_reason}",
            f"last_state_summary={state.last_state_summary}",
            f"last_active_component_id={state.last_active_component_id}",
        ]
    )
