from core.memory_layer import PersistentMemoryState, apply_commit_gate_and_persistence, build_turn_residue
from core.turn_record_pack import build_turn_record
from core.trace_properties import extract_triggered_highlights


def _stable_commit_runtime_output() -> dict:
    # Stable operator, low/flat caution, no hold; commit info is supplied explicitly.
    trace = [
        {"phase": 0, "selected_operator": "++", "caution_after_recovery": 0.20, "raw_caution_scalar": 0.20, "recovery": 0.02, "hold_state": False},
        {"phase": 1, "selected_operator": "++", "caution_after_recovery": 0.21, "raw_caution_scalar": 0.21, "recovery": 0.03, "hold_state": False},
        {"phase": 2, "selected_operator": "++", "caution_after_recovery": 0.20, "raw_caution_scalar": 0.20, "recovery": 0.03, "hold_state": False},
        {"phase": 3, "selected_operator": "++", "caution_after_recovery": 0.20, "raw_caution_scalar": 0.20, "recovery": 0.03, "hold_state": False},
    ]
    return {
        "turn_id": 1,
        "prompt": "what is gravity?",
        "intent_class": "question",
        "reply": "Here's a structured take: Gravity is the attractive force between masses.",
        "output": {"selected_class": 0, "confidence": 0.8},
        "trace": trace,
        "config_used": {"signature_size": 12, "caution_threshold": 0.8},
        "raw_residue_record": {
            "turn_id": 1,
            "intent_class": "question",
            "reply_mode": "explanatory",
            "operator_histogram": {"++": 4, "--": 0, "+-": 0, "-+": 0},
            "switch_fraction": 0.0,
            "caution_terminal": 0.20,
            "hold_terminal": False,
            "recovery_terminal": 0.03,
            "is_qualified": True,
            "commit_reason": "qualified",
            "reject_reason": "",
        },
        "committed_residue_record": {
            "turn_id": 1,
            "commit_decision": "commit",
            "commit_reasons": ["qualified"],
            "operator_bias_delta": {"++": 0.03},
            "caution_baseline_delta": -0.01,
            "persistence_duration": 4,
        },
    }


def _review_reject_runtime_output() -> dict:
    # High switching + rising caution => highlights trigger; persistence rejected.
    trace = [
        {"phase": 0, "selected_operator": "++", "caution_after_recovery": 0.30, "raw_caution_scalar": 0.30, "recovery": 0.02, "hold_state": False},
        {"phase": 1, "selected_operator": "+-", "caution_after_recovery": 0.42, "raw_caution_scalar": 0.42, "recovery": 0.02, "hold_state": False},
        {"phase": 2, "selected_operator": "++", "caution_after_recovery": 0.53, "raw_caution_scalar": 0.53, "recovery": 0.01, "hold_state": False},
        {"phase": 3, "selected_operator": "-+", "caution_after_recovery": 0.64, "raw_caution_scalar": 0.64, "recovery": 0.01, "hold_state": False},
    ]
    return {
        "turn_id": 2,
        "prompt": "explain this weird thing",
        "intent_class": "explanation_request",
        "reply": "Review: unstable path; not committing persistence.",
        "response_mode_override": "review",
        "output": {"selected_class": 0, "confidence": 0.4},
        "trace": trace,
        "config_used": {"signature_size": 12, "caution_threshold": 0.8},
        "raw_residue_record": {
            "turn_id": 2,
            "intent_class": "explanation_request",
            "reply_mode": "explanatory",
            "operator_histogram": {"++": 2, "--": 0, "+-": 1, "-+": 1},
            "switch_fraction": 1.0,
            "caution_terminal": 0.64,
            "hold_terminal": False,
            "recovery_terminal": 0.01,
            "is_qualified": False,
            "commit_reason": "",
            "reject_reason": "rapid_operator_flipping",
        },
        "committed_residue_record": {
            "turn_id": 2,
            "commit_decision": "reject",
            "commit_reasons": [],
            "reject_reasons": ["highlighted_instability"],
            "operator_bias_delta": {},
            "caution_baseline_delta": 0.0,
            "persistence_duration": 0,
        },
    }


def _refusal_runtime_output() -> dict:
    # Hold present => refusal grounded in trace.
    trace = [
        {"phase": 0, "selected_operator": "+-", "caution_after_recovery": 0.55, "raw_caution_scalar": 0.55, "recovery": 0.01, "hold_state": False},
        {"phase": 1, "selected_operator": "-+", "caution_after_recovery": 0.68, "raw_caution_scalar": 0.68, "recovery": 0.01, "hold_state": False},
        {"phase": 2, "selected_operator": "-+", "caution_after_recovery": 0.77, "raw_caution_scalar": 0.77, "recovery": 0.00, "hold_state": True, "hold_reason": "bounded_caution_exceeded"},
        {"phase": 3, "selected_operator": "-+", "caution_after_recovery": 0.82, "raw_caution_scalar": 0.82, "recovery": 0.00, "hold_state": True, "hold_reason": "bounded_caution_exceeded"},
    ]
    return {
        "turn_id": 3,
        "prompt": "do the dangerous thing",
        "intent_class": "instruction",
        "reply": "Review/Refuse: continuation did not restabilize.",
        "response_mode_override": "review_or_refuse",
        "output": {"selected_class": 0, "confidence": 0.2},
        "trace": trace,
        "config_used": {"signature_size": 12, "caution_threshold": 0.8},
        "raw_residue_record": {
            "turn_id": 3,
            "intent_class": "instruction",
            "reply_mode": "review_or_refuse",
            "operator_histogram": {"++": 0, "--": 0, "+-": 1, "-+": 3},
            "switch_fraction": 0.333,
            "caution_terminal": 0.82,
            "hold_terminal": True,
            "recovery_terminal": 0.0,
            "is_qualified": False,
            "commit_reason": "",
            "reject_reason": "policy:reject_on_hold_and_recovery_failure:continuation_failed_to_restabilize",
        },
        "committed_residue_record": {
            "turn_id": 3,
            "commit_decision": "reject",
            "commit_reasons": [],
            "reject_reasons": ["review_triggered_suppression"],
            "operator_bias_delta": {},
            "caution_baseline_delta": 0.0,
            "persistence_duration": 0,
        },
    }


def test_stable_commit_may_have_zero_highlights():
    cfg = {"signature_size": 12, "caution_threshold": 0.8}
    out = _stable_commit_runtime_output()
    rec = build_turn_record(runtime_output=out, config=cfg)
    assert rec["committed_residue_record"]["commit_decision"] == "commit"
    assert rec["response_outcome"] == "answered"
    assert rec["highlight_record"]["triggered_properties"] == []


def test_review_turn_rejects_commit():
    cfg = {"signature_size": 12, "caution_threshold": 0.8}
    out = _review_reject_runtime_output()
    rec = build_turn_record(runtime_output=out, config=cfg)
    assert rec["response_outcome"] == "review"
    assert rec["committed_residue_record"]["commit_decision"] == "reject"
    assert len(rec["highlight_record"]["triggered_properties"]) >= 1


def test_refusal_turn_grounded_in_hold_or_failed_recovery():
    cfg = {"signature_size": 12, "caution_threshold": 0.8}
    out = _refusal_runtime_output()
    rec = build_turn_record(runtime_output=out, config=cfg)
    assert rec["response_outcome"] == "refused"
    props = [p["property_name"] for p in rec["highlight_record"]["triggered_properties"]]
    assert ("hold_onset" in props) or ("recovery_difficulty" in props)


def test_trace_never_equals_memory():
    from core.memory_layer import qualify_residue

    runtime_output = {
        "state": {"signature": {"caution_scalar": 0.45, "recovery_scalar": 0.08, "hold_state": False}, "orientation": {"active_operator": "++"}},
        "output": {"selected_class": 1, "confidence": 0.8},
        "trace": [
            {"phase": 0, "selected_operator": "++", "caution_after_recovery": 0.40, "raw_caution_scalar": 0.50, "recovery": 0.05, "hold_state": False},
            {"phase": 1, "selected_operator": "++", "caution_after_recovery": 0.45, "raw_caution_scalar": 0.55, "recovery": 0.08, "hold_state": False},
        ],
    }
    residue = build_turn_residue(runtime_output=runtime_output, prompt_text="hello", intent_category="greeting", reply_mode="local", turn_id=1)
    residue = qualify_residue(
        residue,
        structured_input=False,
        epsilon=0.10,
        recovery_threshold=0.02,
        max_switch_freq=0.50,
        min_score=0.55,
    )
    s = PersistentMemoryState()
    s, residue2 = apply_commit_gate_and_persistence(state=s, residue=residue, base_duration=3, reinforce=2, max_duration=12)
    assert residue2.is_committed is True
    payload = s.to_dict()
    assert "committed" in payload
    assert "trace" not in payload
    assert "turn_residue" not in payload
    assert "caution_series" not in payload
    assert "operator_path" not in payload


def test_highlight_is_derived_from_trace():
    cfg = {"signature_size": 12, "caution_threshold": 0.8}
    out = _review_reject_runtime_output()
    rec = build_turn_record(runtime_output=out, config=cfg)
    derived = extract_triggered_highlights(runtime_output=out, config=cfg, memory_last=None)
    derived_names = {d["property_name"] for d in derived}
    record_names = {p["property_name"] for p in rec["highlight_record"]["triggered_properties"]}
    assert record_names.issubset(derived_names)
