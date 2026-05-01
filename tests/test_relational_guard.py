import numpy as np

from core.relational_guard import relational_guard, ternary_vector
from native_platform.residue_phase_continuation import ResiduePhaseContinuation
from native_platform.phase_space import normalize


def test_ternary_vector_maps_thresholded_signs():
    out = ternary_vector(np.array([-0.2, -0.001, 0.0, 0.001, 0.2]), threshold=0.01)
    assert out.tolist() == [-1, 0, 0, 0, 1]


def test_relational_guard_classifies_vector_reinforcement_and_risk():
    a = np.ones(8)
    b = np.ones(8)
    result = relational_guard(a, b, 0.0)

    assert result.relation_class == "reinforce"
    assert result.recommended_action == "reinforce_with_resistance"
    assert result.alignment_score == 1.0
    assert result.collapse_risk > 0.99


def test_relational_guard_classifies_opposition_as_recovery_route():
    a = np.ones(8)
    b = -np.ones(8)
    result = relational_guard(a, b, 0.0)

    assert result.relation_class == "cancel_or_tension"
    assert result.recommended_action == "route_to_recovery"
    assert result.opposition_score == 1.0


def test_survivability_records_relational_guard_without_early_hold():
    cont = ResiduePhaseContinuation()
    trace_vec = normalize(np.ones(8))
    cont.history = [np.zeros(8)]
    cont.trace_segments = [trace_vec.copy() for _ in range(16)]

    decision, failed = cont.evaluate_survivability(
        phi_candidate=trace_vec,
        mismatch=0.0,
        op_star="identity",
        signal_x=1.0,
    )

    assert decision == "reinforce"
    assert "relational_overcoherence" not in failed
    assert cont.last_relational_guard["recommended_action"] == "reinforce_with_resistance"


def test_survivability_uses_sustained_relational_overcoherence_as_soft_hold():
    cont = ResiduePhaseContinuation()
    trace_vec = normalize(np.ones(8))
    cont.history = [np.zeros(8)]
    cont.trace_segments = [trace_vec.copy() for _ in range(24)]
    cont.mismatch_history = [0.01, 0.0101, 0.0099]

    decision, failed = cont.evaluate_survivability(
        phi_candidate=trace_vec,
        mismatch=0.01,
        op_star="identity",
        signal_x=1.0,
    )

    assert decision == "hold"
    assert "relational_overcoherence" in failed
    assert cont.last_relational_guard["recommended_action"] == "reinforce_with_resistance"
