from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


TRACE_PROPERTY_SCHEMA_V1: Dict[str, Any] = {
    "id": "v14_trace_property_schema_v1",
    "branch": "v14",
    "design_rule": "Properties are inferred from trace structure and continuation dynamics, not from an explicit moral scoring module.",
    "properties": [
        {"name": "caution_rise"},
        {"name": "corridor_narrowing"},
        {"name": "operator_instability"},
        {"name": "hold_onset"},
        {"name": "recovery_difficulty"},
        {"name": "residue_rejection"},
        {"name": "intent_mode_mismatch"},
    ],
}


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


def _switch_fraction(path: Sequence[str]) -> float:
    if not path or len(path) < 2:
        return 0.0
    switches = sum(1 for i in range(1, len(path)) if str(path[i]) != str(path[i - 1]))
    return float(switches) / float(max(1, len(path) - 1))


def _corridor_width_proxy(item: Dict[str, Any], signature_size: int) -> Tuple[int, int]:
    blocks = _first_present(item, "corridor_blocks", default=None)
    if isinstance(blocks, list):
        blocked = int(len(blocks))
    else:
        blocked = int(_first_present(item, "corridor_block_count", default=0) or 0)
    blocked = int(max(0, min(int(signature_size), blocked)))
    return int(signature_size - blocked), blocked


def _infer_reply_mode_from_text(reply: str) -> str:
    text = (reply or "").strip().lower()
    if not text:
        return "unknown"
    if "how can i help" in text or text in {"hi", "hello", "hey"}:
        return "conversational"
    if "system status" in text or text.startswith("status:"):
        return "meta"
    if "\n" in text or "here's" in text or "let me" in text or ":" in text:
        return "structured"
    return "conversational" if len(text.split()) <= 18 else "structured"


def _classify_intent_simple(prompt: str) -> str:
    t = (prompt or "").strip().lower()
    if not t:
        return "unknown"
    if t in {"hello", "hi", "hey", "yo"}:
        return "greeting"
    if "how are you" in t:
        return "greeting"
    if t in {"thanks", "thank you", "thx"}:
        return "meta"
    if t.startswith(("explain", "describe", "show", "teach", "give")):
        return "explanation_request"
    if t.endswith("?") or t.startswith(("how", "why", "what", "when", "where")):
        return "question"
    if t.startswith(("run", "set", "toggle", "enable", "disable")):
        return "instruction"
    return "unknown"


@dataclass
class HighlightRecord:
    turn_id: int
    phase_index: int
    property_name: str
    triggered: bool
    supporting_fields: Dict[str, Any]
    summary_text: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn_id": int(self.turn_id),
            "phase_index": int(self.phase_index),
            "property_name": str(self.property_name),
            "triggered": bool(self.triggered),
            "supporting_fields": dict(self.supporting_fields),
            "summary_text": str(self.summary_text),
        }


def infer_trace_properties(
    *,
    runtime_output: Dict[str, Any],
    config: Dict[str, Any],
    memory_last: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Returns highlight records (dicts) matching the schema contract.

    Notes:
    - Uses trace + config for core properties.
    - Optionally uses memory_last for residue-related properties (does not treat memory as trace).
    """
    output = _as_dict(runtime_output)
    trace = [t for t in _as_list(output.get("trace", [])) if isinstance(t, dict)]
    cfg = _as_dict(config)
    cfg_used = _as_dict(output.get("config_used", {}))
    if cfg_used:
        # Only override keys we actually consume in this module.
        for k in ("signature_size", "caution_threshold"):
            if k in cfg_used:
                cfg[k] = cfg_used[k]

    ml = _as_dict(memory_last or {})
    ml2 = _as_dict(output.get("memory_summary", {}))
    if ml2:
        # Prefer per-turn memory summary embedded in the runtime output.
        ml = dict(ml, **ml2)

    turn_id = int(_first_present(output, "turn_id", default=0) or 0)
    if turn_id <= 0:
        turn_id = int(ml.get("residue_turn_id", 0) or 0)
    if turn_id <= 0:
        turn_id = max(1, int(_first_present(_as_dict(output.get("state", {})).get("reasoning", {}), "phase_index", default=0) or 0))

    signature_size = int(cfg.get("signature_size", 12))
    caution_threshold = float(cfg.get("caution_threshold", 0.8))

    records: List[HighlightRecord] = []

    # ----------------
    # caution_rise
    # ----------------
    delta_min = 0.08
    caution_series = [float(_first_present(t, "caution_after_recovery", "caution_scalar", "caution", default=0.0) or 0.0) for t in trace]
    raw_series = [float(_first_present(t, "raw_caution_scalar", default=0.0) or 0.0) for t in trace]
    trig_phase = -1
    trig_reason = ""
    for i in range(len(caution_series)):
        prev = float(caution_series[i - 1]) if i > 0 else float(caution_series[i])
        cur = float(caution_series[i])
        delta = float(cur - prev)
        # Ignore the common "warmup" jump from exactly 0.0 to first non-zero caution.
        if i == 1 and prev == 0.0 and len(caution_series) >= 3 and max(caution_series[1:]) > 0.0:
            delta = 0.0
        if delta >= float(delta_min):
            trig_phase = i
            trig_reason = f"delta={delta:+.3f} >= {delta_min:.3f}"
            break
        if cur >= float(caution_threshold):
            trig_phase = i
            trig_reason = f"value={cur:.3f} >= threshold={caution_threshold:.3f}"
            break
    records.append(
        HighlightRecord(
            turn_id=turn_id,
            phase_index=(int(_first_present(trace[trig_phase], "phase", default=trig_phase)) if (trig_phase >= 0 and trig_phase < len(trace)) else int(len(trace) - 1)),
            property_name="caution_rise",
            triggered=bool(trig_phase >= 0),
            supporting_fields={
                "caution_threshold": float(caution_threshold),
                "delta_min": float(delta_min),
                "raw_caution_scalar_last": float(raw_series[-1]) if raw_series else 0.0,
                "caution_after_recovery_last": float(caution_series[-1]) if caution_series else 0.0,
                "reason": trig_reason,
            },
            summary_text=("Continuation encountered elevated resistance." if trig_phase >= 0 else "No significant caution rise detected."),
        )
    )

    # ----------------
    # corridor_narrowing (proxy from blocks)
    # ----------------
    drop_fraction_min = 0.20
    widths: List[int] = []
    blocked_counts: List[int] = []
    for t in trace:
        w, b = _corridor_width_proxy(t, signature_size)
        widths.append(int(w))
        blocked_counts.append(int(b))
    trig_phase = -1
    trig_reason = ""
    for i in range(1, len(widths)):
        prev = int(widths[i - 1])
        cur = int(widths[i])
        if prev <= 0:
            continue
        frac = float(prev - cur) / float(prev)
        if frac >= float(drop_fraction_min):
            trig_phase = i
            trig_reason = f"relative_drop={frac:.3f} >= {drop_fraction_min:.3f}"
            break
    records.append(
        HighlightRecord(
            turn_id=turn_id,
            phase_index=(int(_first_present(trace[trig_phase], "phase", default=trig_phase)) if (trig_phase >= 0 and trig_phase < len(trace)) else int(len(trace) - 1)),
            property_name="corridor_narrowing",
            triggered=bool(trig_phase >= 0),
            supporting_fields={
                "signature_size": int(signature_size),
                "drop_fraction_min": float(drop_fraction_min),
                "width_proxy_last": int(widths[-1]) if widths else int(signature_size),
                "blocked_proxy_last": int(blocked_counts[-1]) if blocked_counts else 0,
                "reason": trig_reason,
                "proxy_note": "width_proxy = signature_size - corridor_block_count (per-phase)",
            },
            summary_text=("Admissible continuation range contracted." if trig_phase >= 0 else "No corridor narrowing detected (proxy)."),
        )
    )

    # ----------------
    # operator_instability
    # ----------------
    ops = [str(_first_present(t, "selected_operator", "op", default="n/a")) for t in trace]
    switch_frac = float(_switch_fraction(ops))
    switch_min = 0.45
    closure = bool(len(ops) >= 2 and ops[0] == ops[-1])
    counts: Dict[str, int] = {}
    for op in ops:
        counts[str(op)] = counts.get(str(op), 0) + 1
    dominant_frac = float(max(counts.values())) / float(max(1, len(ops))) if counts else 0.0
    last_scores = {}
    if trace:
        last = _as_dict(trace[-1])
        last_scores = _as_dict(_first_present(last, "diffused_operator_scores", "operator_scores", default={}))
    # Treat a closed, coherent cycle as "stable enough" even if it switches early.
    triggered_instability = bool(switch_frac >= float(switch_min) and (not closure) and float(dominant_frac) < 0.60)
    records.append(
        HighlightRecord(
            turn_id=turn_id,
            phase_index=int(len(trace) - 1),
            property_name="operator_instability",
            triggered=triggered_instability,
            supporting_fields={

                "operator_path": ops,
                "switch_fraction": float(switch_frac),
                "switch_fraction_min": float(switch_min),
                "closure": bool(closure),
                "dominant_fraction": float(dominant_frac),
                "operator_scores_used_last": last_scores,
            },
            summary_text=("Continuation could not hold a stable orientational path." if triggered_instability else "Operator path is stable enough."),
        )
    )

    # ----------------
    # hold_onset
    # ----------------
    onset = -1
    for i, t in enumerate(trace):
        if bool(_first_present(t, "hold_state", "hold", default=False)):
            onset = i
            break
    onset_phase = int(_first_present(trace[onset], "phase", default=onset)) if onset >= 0 else int(len(trace) - 1)
    onset_reason = str(_first_present(trace[onset], "hold_reason", default="") or "") if onset >= 0 else ""
    records.append(
        HighlightRecord(
            turn_id=turn_id,
            phase_index=int(onset_phase),
            property_name="hold_onset",
            triggered=bool(onset >= 0),
            supporting_fields={
                "hold_state": bool(onset >= 0),
                "hold_reason": onset_reason,
            },
            summary_text=("Continuation entered a non-advance state." if onset >= 0 else "No hold detected."),
        )
    )

    # ----------------
    # recovery_difficulty
    # ----------------
    rec_below = 0.03
    rel_min = 2
    trig_phase = -1
    for i, t in enumerate(trace):
        rec = float(_first_present(t, "recovery_scalar", "recovery", "rec", default=0.0) or 0.0)
        counter = int(_first_present(t, "hold_release_counter", default=0) or 0)
        if rec < float(rec_below) and counter >= int(rel_min):
            trig_phase = i
            break
    records.append(
        HighlightRecord(
            turn_id=turn_id,
            phase_index=(int(_first_present(trace[trig_phase], "phase", default=trig_phase)) if (trig_phase >= 0 and trig_phase < len(trace)) else int(len(trace) - 1)),
            property_name="recovery_difficulty",
            triggered=bool(trig_phase >= 0),
            supporting_fields={
                "recovery_below": float(rec_below),
                "release_counter_min": int(rel_min),
                "recovery_last": float(_first_present(trace[-1], "recovery_scalar", "recovery", "rec", default=0.0) or 0.0) if trace else 0.0,
                "hold_release_counter_last": int(_first_present(trace[-1], "hold_release_counter", default=0) or 0) if trace else 0,
                "hold_release_reason_last": str(_first_present(trace[-1], "hold_release_reason", default="") or "") if trace else "",
            },
            summary_text=("Return to admissible continuation was difficult." if trig_phase >= 0 else "No recovery difficulty detected."),
        )
    )

    # ----------------
    # residue_rejection (memory-only signal; still derived from continuation stability rules)
    # ----------------
    commit_decision = "unknown"
    if bool(ml.get("residue_appended", False)):
        commit_decision = "commit" if bool(ml.get("residue_is_committed", False)) else "reject"
    records.append(
        HighlightRecord(
            turn_id=turn_id,
            phase_index=int(len(trace) - 1),
            property_name="residue_rejection",
            triggered=bool(commit_decision == "reject"),
            supporting_fields={
                "commit_decision": commit_decision,
                "qualified": bool(ml.get("residue_is_qualified", False)),
                "reject_reason": str(ml.get("residue_reject_reason", "") or ""),
                "commit_reason": str(ml.get("residue_commit_reason", "") or ""),
            },
            summary_text=("Turn structure was not accepted for persistence." if commit_decision == "reject" else "Residue accepted (or memory disabled)."),
        )
    )

    # ----------------
    # intent_mode_mismatch (heuristic)
    # ----------------
    prompt = str(output.get("prompt", "") or "")
    intent = _classify_intent_simple(prompt)
    reply_text = str(_first_present(output, "reply", default="") or "")
    actual_mode = _infer_reply_mode_from_text(reply_text)
    expected = {
        "greeting": "conversational",
        "question": "structured",
        "explanation_request": "structured",
        "instruction": "structured",
        "meta": "meta",
        "unknown": "structured",
    }.get(intent, "structured")
    mismatch = bool(actual_mode != expected and actual_mode != "unknown")
    records.append(
        HighlightRecord(
            turn_id=turn_id,
            phase_index=int(len(trace) - 1),
            property_name="intent_mode_mismatch",
            triggered=mismatch,
            supporting_fields={
                "intent_class": intent,
                "expected_reply_mode": expected,
                "actual_reply_mode": actual_mode,
                "commit_decision": commit_decision,
            },
            summary_text=("The continuation mode drifted from the inferred task mode." if mismatch else "Reply mode aligns with inferred intent (heuristic)."),
        )
    )

    return [r.to_dict() for r in records]


def format_trace_property_highlights(
    *,
    records: Sequence[Dict[str, Any]],
    mode: str = "compact",
) -> str:
    """
    Renderer helper: compact shows only triggered properties; full shows all + supporting fields.
    """
    recs = [r for r in records if isinstance(r, dict)]
    if not recs:
        return "(no properties)"

    if mode not in {"compact", "full"}:
        mode = "compact"

    if mode == "compact":
        triggered = [r for r in recs if bool(r.get("triggered", False))]
        if not triggered:
            return "Trace properties: (none triggered)"
        parts = []
        for r in triggered:
            parts.append(f"{r.get('property_name')}@phase{r.get('phase_index')}")
        return "Trace properties: " + ", ".join(parts)

    lines: List[str] = ["Trace properties:"]
    for r in recs:
        name = r.get("property_name")
        phase = r.get("phase_index")
        trig = r.get("triggered")
        summary = r.get("summary_text", "")
        lines.append(f"- {name} phase={phase} triggered={trig} :: {summary}")
        sf = r.get("supporting_fields", {})
        if isinstance(sf, dict) and sf:
            for k in sorted(sf.keys()):
                lines.append(f"  {k}={sf.get(k)}")
    return "\n".join(lines)


TRACE_HIGHLIGHT_EXTRACTOR_V1: Dict[str, Any] = {
    "id": "v14_trace_highlight_extractor_v1",
    "extraction_mode": "on_demand",
    "windowing": {"default_recent_turns": 3, "max_turns": 12},
    "selection": {
        "include_only_triggered_properties": True,
        "max_highlights_per_turn": 8,
        "sort_by": [
            "hold_onset",
            "residue_rejection",
            "intent_mode_mismatch",
            "caution_rise",
            "operator_instability",
            "recovery_difficulty",
            "corridor_narrowing",
        ],
    },
    "rendering": {
        "include_phase_indices": True,
        "include_supporting_values": True,
        "include_plain_summary": True,
    },
    "storage": {"persist_highlights": False, "highlight_cache_turns": 16},
}


def _sort_key(property_name: str, order: Sequence[str]) -> Tuple[int, str]:
    try:
        idx = list(order).index(str(property_name))
    except ValueError:
        idx = 10_000
    return int(idx), str(property_name)


def extract_triggered_highlights(
    *,
    runtime_output: Dict[str, Any],
    config: Dict[str, Any],
    memory_last: Optional[Dict[str, Any]] = None,
    max_per_turn: int = 8,
    sort_by: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    all_records = infer_trace_properties(runtime_output=runtime_output, config=config, memory_last=memory_last)
    triggered = [r for r in all_records if bool(r.get("triggered", False))]
    order = list(sort_by or TRACE_HIGHLIGHT_EXTRACTOR_V1["selection"]["sort_by"])
    triggered.sort(key=lambda r: _sort_key(str(r.get("property_name", "")), order))
    return triggered[: int(max(0, max_per_turn))]


def format_highlight_stream(
    *,
    highlights_by_turn: Sequence[Tuple[int, List[Dict[str, Any]]]],
    include_supporting_values: bool = True,
) -> str:
    """
    Compact, inspectable highlight stream. Does not dump raw trace.
    """
    items = list(highlights_by_turn)
    if not items:
        return "Trace highlights: (no turns)"

    any_triggered = any(bool(hs) for _, hs in items)
    if not any_triggered:
        return "Trace highlights: (none triggered)"

    lines: List[str] = ["Trace highlights:"]
    for turn_id, hs in items:
        if not hs:
            continue
        for h in hs:
            name = str(h.get("property_name", ""))
            phase = h.get("phase_index", "?")
            summary = str(h.get("summary_text", "") or "")
            if include_supporting_values:
                sf = h.get("supporting_fields", {})
                sf_txt = ""
                if isinstance(sf, dict) and sf:
                    keep = {}
                    for k in ("reason", "commit_decision", "reject_reason", "intent_class", "expected_reply_mode", "actual_reply_mode"):
                        if k in sf and sf.get(k) not in (None, "", []):
                            keep[k] = sf.get(k)
                    if keep:
                        sf_txt = " " + " ".join([f"{k}={keep[k]}" for k in sorted(keep.keys())])
                lines.append(f"- turn={int(turn_id)} phase={phase} {name}: {summary}{sf_txt}".rstrip())
            else:
                lines.append(f"- turn={int(turn_id)} phase={phase} {name}: {summary}".rstrip())
    return "\n".join(lines)
