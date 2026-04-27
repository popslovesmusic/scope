from core.admissibility_gate import score_shadow_admissibility_for_phase, summarize_shadow_admissibility
from core.misleading_positive_detector import detect_misleading_positive
from core.projection_path import build_projection_path


def test_shadow_admissibility_scoring_uses_blocks_and_hits():
    item = {"corridor_hits": [1, 2, 3], "corridor_blocks": [4]}
    s = score_shadow_admissibility_for_phase(trace_item=item)
    assert 0.0 <= s.score <= 1.0
    assert s.block_count == 1
    assert s.hit_count == 3
    assert s.blocked_fraction == 0.25


def test_shadow_admissibility_summary_has_phase_scores():
    trace = [
        {"phase": 0, "corridor_hits": [1, 2], "corridor_blocks": []},
        {"phase": 1, "corridor_hits": [1], "corridor_blocks": [2, 3]},
    ]
    summ = summarize_shadow_admissibility(trace=trace)
    assert len(summ["phase_scores"]) == 2
    assert 0.0 <= summ["final_score"] <= 1.0
    assert 0.0 <= summ["min_score"] <= 1.0
    assert 0.0 <= summ["mean_score"] <= 1.0


def test_misleading_positive_detector_flags_gap_case():
    mp = detect_misleading_positive(practical_confidence=0.9, structural_confidence=0.2, practical_high=0.6, structural_low=0.45, gap_min=0.2)
    assert mp.flagged is True
    assert mp.gap > 0.2


def test_projection_path_is_derived_from_trace():
    runtime_output = {
        "turn_id": 1,
        "output": {"selected_class": 0, "confidence": 0.8},
        "trace": [
            {"phase": 0, "selected_operator": "++", "shift": 0.0, "corridor_hits": [1], "corridor_blocks": [], "caution_after_recovery": 0.1, "recovery": 0.1, "hold_state": False},
            {"phase": 1, "selected_operator": "--", "shift": 0.1, "corridor_hits": [], "corridor_blocks": [2], "caution_after_recovery": 0.2, "recovery": 0.0, "hold_state": False},
        ],
    }
    pp = build_projection_path(runtime_output=runtime_output, practical_confidence=0.8, structural_confidence=0.6, misleading_positive=False)
    assert pp["turn_id"] == 1
    assert len(pp["phases"]) == 2
    assert pp["phases"][0]["operator"] == "++"

