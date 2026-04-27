from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .trace_properties import extract_triggered_highlights


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _first_present(mapping: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def _switch_fraction(path: Sequence[str]) -> float:
    xs = [str(x) for x in (path or [])]
    if len(xs) < 2:
        return 0.0
    switches = sum(1 for i in range(1, len(xs)) if xs[i] != xs[i - 1])
    return float(switches) / float(max(1, len(xs) - 1))


def _corridor_width_fraction_series(trace: Sequence[Dict[str, Any]], signature_size: int) -> List[float]:
    out: List[float] = []
    size = int(max(1, signature_size))
    for t in trace:
        blocks = _first_present(t, "corridor_blocks", default=None)
        if isinstance(blocks, list):
            blocked = int(len(blocks))
        else:
            blocked = int(_first_present(t, "corridor_block_count", default=0) or 0)
        blocked = int(max(0, min(size, blocked)))
        out.append(float(size - blocked) / float(size))
    return out


def build_trace_record(*, runtime_output: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    out = _as_dict(runtime_output)
    cfg = _as_dict(config)
    cfg_used = _as_dict(out.get("config_used", {}))
    if cfg_used:
        for k in ("signature_size", "caution_threshold"):
            if k in cfg_used:
                cfg[k] = cfg_used[k]

    signature_size = int(cfg.get("signature_size", 12))
    caution_threshold = float(cfg.get("caution_threshold", 0.8))

    trace = [t for t in _as_list(out.get("trace", [])) if isinstance(t, dict)]
    operator_path = [str(_first_present(t, "selected_operator", "op", default="n/a")) for t in trace]
    caution_series = [float(_first_present(t, "caution_after_recovery", "caution_scalar", "caution", default=0.0) or 0.0) for t in trace]
    raw_caution_series = [float(_first_present(t, "raw_caution_scalar", default=0.0) or 0.0) for t in trace]
    recovery_series = [float(_first_present(t, "recovery_scalar", "recovery", "rec", default=0.0) or 0.0) for t in trace]
    hold_series = [bool(_first_present(t, "hold_state", "hold", default=False)) for t in trace]

    hold_events = []
    for t in trace:
        if bool(_first_present(t, "hold_triggered", "hold_state", default=False)):
            hold_events.append(
                {
                    "phase_index": int(_first_present(t, "phase", default=0) or 0),
                    "hold_reason": str(_first_present(t, "hold_reason", default="") or ""),
                }
            )

    output_block = _as_dict(out.get("output", {}))
    selected_class = _first_present(output_block, "selected_class", default=None)
    confidence = _first_present(output_block, "confidence", default=None)

    return {
        "turn_id": out.get("turn_id", None),
        "phase_count": int(len(trace)),
        "operator_path": operator_path,
        "switch_fraction": float(_switch_fraction(operator_path)),
        "caution_series": caution_series,
        "raw_caution_terminal": float(raw_caution_series[-1]) if raw_caution_series else 0.0,
        "caution_after_recovery": float(caution_series[-1]) if caution_series else 0.0,
        "caution_threshold": float(caution_threshold),
        "hold_events": hold_events,
        "hold_terminal": bool(hold_series[-1]) if hold_series else False,
        "recovery_series": recovery_series,
        "recovery_terminal": float(recovery_series[-1]) if recovery_series else 0.0,
        "corridor_width_series": _corridor_width_fraction_series(trace, signature_size),
        "selected_class": selected_class,
        "confidence": confidence,
    }


def build_highlight_record(*, runtime_output: Dict[str, Any], config: Dict[str, Any], memory_last: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    out = _as_dict(runtime_output)
    highlights = extract_triggered_highlights(runtime_output=out, config=config, memory_last=memory_last)
    triggered = []
    for h in highlights:
        if not isinstance(h, dict):
            continue
        triggered.append(
            {
                "property_name": str(h.get("property_name", "")),
                "phase_indices": [int(h.get("phase_index", 0) or 0)],
                "supporting_fields": _as_dict(h.get("supporting_fields", {})),
                "summary_text": str(h.get("summary_text", "")),
            }
        )
    if not triggered:
        review_summary = "No significant pressure signatures detected. Continuation remained admissible and stable."
    else:
        names = ", ".join([t["property_name"] for t in triggered if t.get("property_name")])
        review_summary = f"Triggered: {names}. Review before committing persistence."
    return {
        "turn_id": out.get("turn_id", None),
        "triggered_properties": triggered,
        "review_summary": review_summary,
    }


def build_raw_residue_record(*, runtime_output: Dict[str, Any]) -> Dict[str, Any]:
    out = _as_dict(runtime_output)
    raw = _as_dict(out.get("raw_residue_record", {}))
    if not raw:
        ms = _as_dict(out.get("memory_summary", {}))
        return {
            "turn_id": out.get("turn_id", None),
            "intent_class": str(ms.get("intent_class", "") or ""),
            "reply_mode": str(ms.get("reply_mode", "") or ""),
            "qualified": bool(ms.get("residue_is_qualified", False)),
            "reject_reasons": ([str(ms.get("residue_reject_reason"))] if ms.get("residue_reject_reason") else []),
        }
    # Normalize keys to match the separation schema.
    qualified = bool(_first_present(raw, "is_qualified", default=False))
    commit_reason = str(_first_present(raw, "commit_reason", default="") or "")
    reject_reason = str(_first_present(raw, "reject_reason", default="") or "")
    return {
        "turn_id": raw.get("turn_id", out.get("turn_id", None)),
        "intent_class": str(_first_present(raw, "intent_class", "intent_category", default="") or ""),
        "reply_mode": str(_first_present(raw, "reply_mode", default="") or ""),
        "operator_histogram": _as_dict(raw.get("operator_histogram", {})),
        "switch_fraction": float(raw.get("switch_fraction", 0.0) or 0.0),
        "caution_terminal": float(raw.get("caution_terminal", raw.get("caution", 0.0)) or 0.0),
        "hold_terminal": bool(raw.get("hold_terminal", False)),
        "recovery_terminal": float(raw.get("recovery_terminal", raw.get("recovery", 0.0)) or 0.0),
        "qualified": qualified,
        "qualification_reasons": ([commit_reason] if (qualified and commit_reason) else []),
        "reject_reasons": ([reject_reason] if ((not qualified) and reject_reason) else []),
    }


def build_committed_residue_record(*, runtime_output: Dict[str, Any]) -> Dict[str, Any]:
    out = _as_dict(runtime_output)
    committed = _as_dict(out.get("committed_residue_record", {}))
    ms = _as_dict(out.get("memory_summary", {}))
    if committed:
        return dict(committed)
    # If we don't have a committed record, emit a reject shape.
    commit_decision = str(ms.get("commit_decision", "reject") or "reject")
    reject_reason = str(ms.get("residue_reject_reason", "") or "")
    return {
        "turn_id": out.get("turn_id", None),
        "commit_decision": commit_decision,
        "commit_reasons": [],
        "reject_reasons": ([reject_reason] if reject_reason else []),
        "operator_bias_delta": {},
        "caution_baseline_delta": 0.0,
        "persistence_duration": 0,
    }


def build_turn_record(
    *,
    runtime_output: Dict[str, Any],
    config: Dict[str, Any],
    memory_last: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    out = _as_dict(runtime_output)
    prompt = str(out.get("prompt", "") or "")
    user_input_class = str(_first_present(out, "intent_class", default="") or "")
    response_outcome = str(out.get("response_mode_override", "") or "answered")
    if response_outcome == "review_or_refuse":
        response_outcome = "refused"
    elif response_outcome in {"review", "repair_to_intent"}:
        response_outcome = "review"
    else:
        response_outcome = "answered"

    reply_mode = str(out.get("response_mode_override", "") or "")
    if not reply_mode:
        reply_mode = "explanatory"

    return {
        "turn_id": out.get("turn_id", None),
        "prompt": prompt,
        "user_input_class": user_input_class,
        "reply_mode": reply_mode,
        "response_outcome": response_outcome,
        "trace_record": build_trace_record(runtime_output=out, config=config),
        "highlight_record": build_highlight_record(runtime_output=out, config=config, memory_last=memory_last),
        "raw_residue_record": build_raw_residue_record(runtime_output=out),
        "committed_residue_record": build_committed_residue_record(runtime_output=out),
    }


def build_turn_record_pack(
    *,
    runtime_records: Sequence[Dict[str, Any]],
    config: Dict[str, Any],
    memory_last: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    recs = [r for r in runtime_records if isinstance(r, dict)]
    out = []
    for r in recs:
        out.append(build_turn_record(runtime_output=r, config=config, memory_last=memory_last))
    return {
        "id": "v14_turn_record_pack_v1",
        "branch": "v14",
        "kind": "pack",
        "records": out,
    }


def format_turn_record_json(turn_record: Dict[str, Any]) -> str:
    return json.dumps(turn_record, indent=2, sort_keys=True)
