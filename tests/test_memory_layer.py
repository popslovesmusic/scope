from core.memory_layer import (
    PersistentMemoryState,
    apply_commit_gate_and_persistence,
    build_turn_residue,
    classify_intent,
    load_memory_state,
    qualify_residue,
    save_memory_state,
)


def test_intent_classifier_basic():
    assert classify_intent("hello") == "greeting"
    assert classify_intent("how are you?") == "greeting"
    assert classify_intent("explain photosynthesis") == "explanation_request"
    assert classify_intent("what is gravity?") == "question"
    assert classify_intent("thanks") == "meta"


def test_residue_qualifies_and_commits_when_stable():
    runtime_output = {
        "state": {
            "signature": {
                "caution_scalar": 0.45,
                "recovery_scalar": 0.08,
                "hold_state": False,
                "active_component_id": 0,
                "components": [{}, {}],
            },
            "orientation": {"active_operator": "++"},
        },
        "output": {"selected_class": 1, "confidence": 0.8},
        "trace": [
            {"phase": 0, "selected_operator": "++", "caution_scalar": 0.40, "recovery": 0.05, "hold_state": False},
            {"phase": 1, "selected_operator": "++", "caution_scalar": 0.45, "recovery": 0.08, "hold_state": False},
        ],
    }

    residue = build_turn_residue(
        runtime_output=runtime_output,
        prompt_text="hello",
        intent_category="greeting",
        reply_mode="local",
        turn_id=1,
        structured_input=False,
    )
    d = residue.to_dict()
    assert "operator_histogram" in d
    assert "switch_fraction" in d
    assert "caution_terminal" in d
    assert "hold_terminal" in d
    assert "recovery_terminal" in d
    residue = qualify_residue(
        residue,
        structured_input=False,
        epsilon=0.10,
        recovery_threshold=0.02,
        max_switch_freq=0.50,
        min_score=0.65,
    )
    assert residue.is_qualified is True
    assert residue.ratchet is True

    s = PersistentMemoryState()
    s, residue2 = apply_commit_gate_and_persistence(state=s, residue=residue, base_duration=3, reinforce=2, max_duration=12)
    assert residue2.is_committed is True
    assert s.committed_residue_count == 1
    assert sum(float(v) for v in s.operator_bias.values()) == 1.0


def test_unstable_residue_is_rejected_and_does_not_commit():
    runtime_output = {
        "state": {"signature": {"caution_scalar": 0.8, "recovery_scalar": 0.0, "hold_state": False}, "orientation": {"active_operator": "+-"}},
        "output": {"selected_class": 0, "confidence": 0.4},
        "trace": [
            {"phase": 0, "selected_operator": "++", "caution_scalar": 0.2, "recovery": 0.0, "hold_state": False},
            {"phase": 1, "selected_operator": "+-", "caution_scalar": 0.9, "recovery": 0.0, "hold_state": False},
            {"phase": 2, "selected_operator": "--", "caution_scalar": 0.1, "recovery": 0.0, "hold_state": False},
        ],
    }
    residue = build_turn_residue(
        runtime_output=runtime_output,
        prompt_text="noisy",
        intent_category="unknown",
        reply_mode="local",
        turn_id=1,
        structured_input=False,
    )
    assert residue.switch_fraction >= 0.0
    residue = qualify_residue(
        residue,
        structured_input=False,
        epsilon=0.10,
        recovery_threshold=0.02,
        max_switch_freq=0.50,
        min_score=0.65,
    )
    assert residue.is_qualified is False
    s = PersistentMemoryState()
    s, residue2 = apply_commit_gate_and_persistence(state=s, residue=residue, base_duration=3, reinforce=2, max_duration=12)
    assert residue2.is_committed is False
    assert s.committed_residue_count == 0


def test_greeting_like_phase_cycle_can_qualify_despite_switching():
    runtime_output = {
        "state": {"signature": {"caution_scalar": 0.38, "recovery_scalar": 0.08, "hold_state": False}, "orientation": {"active_operator": "--"}},
        "output": {"selected_class": 0, "confidence": 0.28},
        "trace": [
            {"phase": 0, "selected_operator": "--", "caution": 0.0, "recovery": 0.0, "hold_state": False},
            {"phase": 1, "selected_operator": "+-", "caution": 0.292, "recovery": 0.059, "hold_state": False},
            {"phase": 2, "selected_operator": "++", "caution": 0.295, "recovery": 0.091, "hold_state": False},
            {"phase": 3, "selected_operator": "--", "caution": 0.383, "recovery": 0.080, "hold_state": False},
        ],
    }
    residue = build_turn_residue(
        runtime_output=runtime_output,
        prompt_text="hello",
        intent_category="greeting",
        reply_mode="local",
        turn_id=1,
        structured_input=False,
    )
    residue = qualify_residue(
        residue,
        structured_input=False,
        epsilon=0.10,
        recovery_threshold=0.02,
        max_switch_freq=0.50,
        min_score=0.55,
    )
    assert residue.is_qualified is True
    assert residue.ratchet is True


def test_memory_state_save_load_roundtrip(tmp_path):
    p = tmp_path / "memory_state.json"
    s = PersistentMemoryState()
    s.turn_counter = 7
    save_memory_state(str(p), s)
    loaded = load_memory_state(str(p))
    assert loaded.turn_counter == 7
