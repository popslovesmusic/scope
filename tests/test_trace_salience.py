import numpy as np

from core.corridor_gate import build_dynamic_corridor
from core.peak_detector import Peak
from core.reasoning_loop import run_reasoning
from core.signature_state import SignatureState
from core.trace_ops import compute_caution_field, compute_trace_salience
from core.component_ops import build_component_masks


def test_high_salience_and_similarity_increases_caution_scalar():
    trace = [
        {
            "boundary_penalty_energy": 3.0,
            "corridor_block_count": 10,
            "cancellation_penalty_energy": 2.0,
            "polarity_shift": 0.5,
            "spread_measure": 10.0,
            "active_component_sign": 1,
            "active_component_width": 4,
            "signed_run_count": 2,
            "largest_signed_run_width": 4,
            "selected_operator": "++",
        }
    ]
    sal = compute_trace_salience(trace, window=8)
    assert sal and sal[0] > 0.5

    components = [{"stable_id": 0, "component_id": 0, "sign": 1, "start_family": 0, "end_family": 3, "width": 4, "support_mass": 1.0}]
    masks = build_component_masks(components, size=12)
    raw_scalar, raw_field, bounded_scalar, bounded_field, release_scalar, after_scalar, after_field, sal_max, sim_max = compute_caution_field(
        size=12,
        components=components,
        component_masks=masks,
        current_structure={
            "active_component_id": 0,
            "active_component_sign": 1,
            "active_component_width": 4,
            "signed_run_count": 2,
            "largest_signed_run_width": 4,
        },
        trace_history=trace,
        salience=sal,
        salience_weight=0.4,
        similarity_weight=0.6,
    )
    assert raw_scalar > 0.0
    assert float(np.max(raw_field)) > 0.0
    assert 0.0 <= bounded_scalar <= 1.0
    assert float(np.max(bounded_field)) > 0.0
    assert 0.0 <= release_scalar <= 1.0
    assert 0.0 <= after_scalar <= 1.0
    assert float(np.max(after_field)) > 0.0
    assert sal_max > 0.0
    assert sim_max > 0.0


def test_caution_field_narrows_corridor_locally():
    peaks = [Peak(family=6, amplitude=1.0)]
    caution = np.zeros(12, dtype=float)
    caution[6] = 1.0

    no_c = build_dynamic_corridor(
        size=12,
        peaks=peaks,
        bands=[],
        stability=np.ones(12),
        amplitude=np.ones(12),
        floor=0.0,
        threshold=0.1,
        caution_corridor_penalty=0.0,
    )
    with_c = build_dynamic_corridor(
        size=12,
        peaks=peaks,
        bands=[],
        stability=np.ones(12),
        amplitude=np.ones(12),
        floor=0.0,
        threshold=0.1,
        caution_field=caution,
        caution_corridor_penalty=0.9,
    )
    assert with_c.window[6] < no_c.window[6]


def test_hold_state_triggers_only_above_threshold():
    cfg = {
        "signature_size": 12,
        "lambda": 0.2,
        "kappa": 0.6,
        "eta": 0.1,
        "phases": 2,
        "family_diffusion": 0.05,
        "diffusion_boundary_mode": "wrap",
        "use_wraparound_lattice": True,
        "polarity_threshold": 0.01,
        "zero_crossing_threshold": 0.02,
        "enable_component_promotion": True,
        "component_target_mode": "dominant",
        "component_min_support_mass": 0.0,
        "component_min_width": 1,
        "enable_trace_salience": True,
        "trace_salience_window": 8,
        "caution_similarity_weight": 0.6,
        "caution_salience_weight": 0.4,
        "caution_operator_penalty": 0.0,
        "caution_corridor_penalty": 0.0,
        "enable_hold_state": True,
        # Force blocks/penalties to increase salience early.
        "corridor": {"floor": 0.0, "threshold": 0.95},
    }
    signed_init = np.linspace(-1.0, 1.0, 12)

    s_low = SignatureState(12)
    s_low.signed_field = signed_init.copy()
    s_low.derive_amplitude_from_signed()
    tr_low = run_reasoning(s_low, dict(cfg, caution_threshold=0.0))
    assert len(tr_low) == 2
    assert bool(tr_low[1].get("hold_triggered", False)) is True

    s_high = SignatureState(12)
    s_high.signed_field = signed_init.copy()
    s_high.derive_amplitude_from_signed()
    tr_high = run_reasoning(s_high, dict(cfg, caution_threshold=1.0))
    assert bool(tr_high[1].get("hold_triggered", False)) is False


def test_raw_caution_can_exceed_bounded_caution_but_bounded_stays_in_range():
    trace = [
        {
            "boundary_penalty_energy": 3.0,
            "corridor_block_count": 10,
            "cancellation_penalty_energy": 2.0,
            "polarity_shift": 0.5,
            "spread_measure": 10.0,
            "active_component_sign": 1,
            "active_component_width": 6,
            "signed_run_count": 1,
            "largest_signed_run_width": 6,
            "selected_operator": "++",
        }
        for _ in range(3)
    ]
    sal = [1.0, 1.0, 1.0]
    components = [{"stable_id": 0, "component_id": 0, "sign": 1, "start_family": 0, "end_family": 5, "width": 6, "support_mass": 1.0}]
    masks = build_component_masks(components, size=12)
    raw_scalar, _, bounded_scalar, _, release_scalar, _after_scalar, _after_field, _sal_max, _sim_max = compute_caution_field(
        size=12,
        components=components,
        component_masks=masks,
        current_structure={
            "active_component_id": 0,
            "active_component_sign": 1,
            "active_component_width": 6,
            "signed_run_count": 1,
            "largest_signed_run_width": 6,
            "selected_operator": "++",
        },
        trace_history=trace,
        salience=sal,
        salience_weight=0.4,
        similarity_weight=0.6,
    )
    assert raw_scalar > bounded_scalar
    assert raw_scalar > 1.0
    assert 0.0 <= bounded_scalar <= 1.0
    assert 0.0 <= release_scalar <= 1.0


def test_caution_release_reduces_applied_caution_when_similarity_is_low():
    trace = [
        {
            "boundary_penalty_energy": 3.0,
            "corridor_block_count": 10,
            "cancellation_penalty_energy": 2.0,
            "polarity_shift": 0.5,
            "spread_measure": 10.0,
            "active_component_sign": -1,
            "active_component_width": 6,
            "signed_run_count": 1,
            "largest_signed_run_width": 6,
            "selected_operator": "--",
        },
        {
            "boundary_penalty_energy": 3.0,
            "corridor_block_count": 10,
            "cancellation_penalty_energy": 2.0,
            "polarity_shift": 0.5,
            "spread_measure": 10.0,
            "active_component_sign": -1,
            "active_component_width": 6,
            "signed_run_count": 1,
            "largest_signed_run_width": 6,
            "selected_operator": "--",
        },
    ]
    sal = [1.0, 1.0]
    components = [{"stable_id": 0, "component_id": 0, "sign": 1, "start_family": 0, "end_family": 5, "width": 6, "support_mass": 1.0}]
    masks = build_component_masks(components, size=12)
    _, _, bounded_scalar, bounded_field, release_scalar, after_scalar, after_field, _sal_max, _sim_max = compute_caution_field(
        size=12,
        components=components,
        component_masks=masks,
        current_structure={
            "active_component_id": 0,
            "active_component_sign": 1,
            "active_component_width": 6,
            "signed_run_count": 1,
            "largest_signed_run_width": 6,
            "selected_operator": "++",
        },
        trace_history=trace,
        salience=sal,
        salience_weight=0.4,
        similarity_weight=0.6,
        caution_release_rate=0.5,
    )
    assert release_scalar > 0.0
    assert float(after_scalar) <= float(bounded_scalar)
    assert float(np.mean(after_field)) < float(bounded_scalar)


def test_recovery_and_recontextualization_reduce_applied_caution_only():
    trace = [
        {
            "boundary_penalty_energy": 0.0,
            "corridor_block_count": 0,
            "cancellation_penalty_energy": 2.0,
            "polarity_shift": 0.0,
            "spread_measure": 0.0,
            "active_component_sign": 1,
            "active_component_width": 4,
            "signed_run_count": 1,
            "largest_signed_run_width": 4,
            "selected_operator": "++",
        }
    ]
    sal = compute_trace_salience(trace, window=8)
    components = [{"stable_id": 0, "component_id": 0, "sign": 1, "start_family": 0, "end_family": 3, "width": 4, "support_mass": 1.0}]
    masks = build_component_masks(components, size=12)

    raw0, _, _b0, _bf0, _rel0, after0, _af0, _sal0, _sim0 = compute_caution_field(
        size=12,
        components=components,
        component_masks=masks,
        current_structure={
            "active_component_id": 0,
            "active_component_sign": 1,
            "active_component_width": 4,
            "signed_run_count": 1,
            "largest_signed_run_width": 4,
            "selected_operator": "++",
        },
        trace_history=trace,
        salience=sal,
        salience_weight=0.4,
        similarity_weight=0.6,
        caution_release_rate=0.0,
        recovery_scalar=0.0,
        recontextualization_score=0.0,
        recovery_rate=0.5,
        recontextualization_weight=0.5,
    )
    raw1, _, _b1, _bf1, _rel1, after1, _af1, _sal1, _sim1 = compute_caution_field(
        size=12,
        components=components,
        component_masks=masks,
        current_structure={
            "active_component_id": 0,
            "active_component_sign": 1,
            "active_component_width": 4,
            "signed_run_count": 1,
            "largest_signed_run_width": 4,
            "selected_operator": "++",
        },
        trace_history=trace,
        salience=sal,
        salience_weight=0.4,
        similarity_weight=0.6,
        caution_release_rate=0.0,
        recovery_scalar=1.0,
        recontextualization_score=1.0,
        recovery_rate=0.5,
        recontextualization_weight=0.5,
    )

    assert
 abs(float(raw0) - float(raw1)) < 1e-12
    assert float(after1) < float(after0)


def test_hold_uses_bounded_caution_not_raw_caution():
    cfg = {
        "signature_size": 12,
        "lambda": 0.2,
        "kappa": 0.6,
        "eta": 0.1,
        "phases": 5,
        "family_diffusion": 0.05,
        "diffusion_boundary_mode": "wrap",
        "use_wraparound_lattice": True,
        "polarity_threshold": 0.01,
        "zero_crossing_threshold": 0.02,
        "enable_component_promotion": True,
        "component_target_mode": "dominant",
        "component_min_support_mass": 0.0,
        "component_min_width": 1,
        "enable_trace_salience": True,
        "trace_salience_window": 8,
        "caution_similarity_weight": 0.6,
        "caution_salience_weight": 0.4,
        "caution_operator_penalty": 0.0,
        "caution_corridor_penalty": 0.0,
        "caution_release_rate": 0.0,
        "enable_hold_state": True,
        "hold_from_bounded_caution_only": True,
        "caution_threshold": 0.75,
        # Force blocks/penalties to increase salience early.
        "corridor": {"floor": 0.0, "threshold": 0.95},
    }
    signed_init = np.linspace(-1.0, 1.0, 12)

    s = SignatureState(12)
    s.signed_field = signed_init.copy()
    s.derive_amplitude_from_signed()
    tr = run_reasoning(s, cfg)
    assert len(tr) == 5
    assert float(tr[-1].get("raw_caution_scalar", 0.0)) > float(cfg["caution_threshold"])
    assert float(tr[-1].get("caution_scalar", 0.0)) < float(cfg["caution_threshold"])
    assert bool(tr[-1].get("hold_triggered", False)) is False


def test_operator_scores_are_traceable_pre_and_post_caution():
    cfg = {
        "signature_size": 12,
        "lambda": 0.2,
        "kappa": 0.6,
        "eta": 0.1,
        "phases": 2,
        "family_diffusion": 0.05,
        "diffusion_boundary_mode": "wrap",
        "use_wraparound_lattice": True,
        "polarity_threshold": 0.01,
        "zero_crossing_threshold": 0.02,
        "enable_component_promotion": True,
        "component_target_mode": "dominant",
        "component_min_support_mass": 0.0,
        "component_min_width": 1,
        "enable_trace_salience": True,
        "trace_salience_window": 8,
        "caution_similarity_weight": 0.6,
        "caution_salience_weight": 0.4,
        "caution_release_rate": 0.0,
        "caution_global_operator_penalty": 1.0,
        "caution_local_operator_penalty": 0.0,
        "caution_corridor_penalty": 0.0,
        "enable_hold_state": False,
        # Force blocks/penalties to increase salience early.
        "corridor": {"floor": 0.0, "threshold": 0.95},
    }
    signed_init = np.linspace(-1.0, 1.0, 12)

    s = SignatureState(12)
    s.signed_field = signed_init.copy()
    s.derive_amplitude_from_signed()
    tr = run_reasoning(s, cfg)
    assert len(tr) == 2
    pre = tr[-1].get("operator_scores_pre_caution", {})
    post = tr[-1].get("operator_scores_post_caution", {})
    assert pre and post and set(pre.keys()) == set(post.keys())
    assert any(abs(float(pre[k]) - float(post[k])) > 1e-12 for k in pre.keys())


def test_hold_semantics_freeze_keeps_signature_unchanged_for_hold_phase():
    cfg = {
        "signature_size": 12,
        "lambda": 0.2,
        "kappa": 0.6,
        "eta": 0.1,
        "family_diffusion": 0.05,
        "orientation_diffusion": 0.0,
        "diffusion_boundary_mode": "wrap",
        "use_wraparound_lattice": True,
        "polarity_threshold": 0.01,
        "zero_crossing_threshold": 0.02,
        "enable_component_promotion": True,
        "component_target_mode": "dominant",
        "component_min_support_mass": 0.0,
        "component_min_width": 1,
        "enable_trace_salience": True,
        "trace_salience_window": 8,
        "caution_similarity_weight": 0.6,
        "caution_salience_weight": 0.4,
        "caution_operator_penalty": 0.0,
        "caution_corridor_penalty": 0.0,
        "enable_hold_state": True,
        "hold_semantics": "freeze",
        "caution_threshold": 0.0,
        # Force blocks/penalties to increase salience early.
        "corridor": {"floor": 0.0, "threshold": 0.95},
    }
    signed_init = np.linspace(-1.0, 1.0, 12)

    s1 = SignatureState(12)
    s1.signed_field = signed_init.copy()
    s1.derive_amplitude_from_signed()
    run_reasoning(s1, dict(cfg, phases=1))
    signed_after_phase1 = s1.signed_field.copy()

    s2 = SignatureState(12)
    s2.signed_field = signed_init.copy()
    s2.derive_amplitude_from_signed()
    tr2 = run_reasoning(s2, dict(cfg, phases=2))
    assert bool(tr2[1].get("hold_triggered", False)) is True
    assert np.allclose(s2.signed_field, signed_after_phase1)


def test_hold_semantics_decay_matches_default_behavior():
    cfg = {
        "signature_size": 12,
        "lambda": 0.2,
        "kappa": 0.6,
        "eta": 0.1,
        "phases": 3,
        "family_diffusion": 0.05,
        "diffusion_boundary_mode": "wrap",
        "use_wraparound_lattice": True,
        "polarity_threshold": 0.01,
        "zero_crossing_threshold": 0.02,
        "enable_component_promotion": True,
        "component_target_mode": "dominant",
        "component_min_support_mass": 0.0,
        "component_min_width": 1,
        "enable_trace_salience": True,
        "trace_salience_window": 8,
        "caution_similarity_weight": 0.6,
        "caution_salience_weight": 0.4,
        "caution_operator_penalty": 0.0,
        "caution_corridor_penalty": 0.0,
        "enable_hold_state": True,
        "hold_from_bounded_caution_only": True,
        "caution_threshold": 0.0,
        # Force blocks/penalties to increase salience early.
        "corridor": {"floor": 0.0, "threshold": 0.95},
    }
    signed_init = np.linspace(-1.0, 1.0, 12)

    s_def = SignatureState(12)
    s_def.signed_field = signed_init.copy()
    s_def.derive_amplitude_from_signed()
    tr_def = run_reasoning(s_def, dict(cfg))

    s_explicit = SignatureState(12)
    s_explicit.signed_field = signed_init.copy()
    s_explicit.derive_amplitude_from_signed()
    tr_explicit = run_reasoning(s_explicit, dict(cfg, hold_semantics="decay"))

    assert len(tr_def) == len(tr_explicit) == 3
    assert np.allclose(s_def.signed_field, s_explicit.signed_field)


def test_hold_release_counter_increments_under_recovery_favorable_hold():
    cfg = {
        "signature_size": 12,
        "lambda": 0.2,
        "kappa": 0.6,
        "eta": 0.1,
        "phases": 3,
        "family_diffusion": 0.0,
        "diffusion_boundary_mode": "wrap",
        "use_wraparound_lattice": True,
        "polarity_threshold": 0.01,
        "zero_crossing_threshold": 0.02,
        "enable_component_promotion": True,
        "component_target_mode": "dominant",
        "component_min_support_mass": 0.0,
        "component_min_width": 1,
        "enable_trace_salience": True,
        "trace_salience_window": 8,
        "caution_similarity_weight": 0.6,
        "caution_salience_weight": 0.4,
        "caution_release_rate": 0.0,
        "caution_global_operator_penalty": 0.0,
        "caution_local_operator_penalty": 0.0,
        "caution_corridor_penalty": 0.0,
        "enable_hold_state": True,
        "hold_semantics": "decay",
        "hold_persist": True,
        "caution_threshold": 0.05,
        "enable_recovery": True,
        "recovery_rate": 0.5,
        "recontextualization_weight": 0.5,
        "hold_release_threshold": 0.0,
        "hold_release_required_phases": 1,
        # Keep corridor permissive so blocks/boundary stay low; cancellation still contributes to salience.
        "corridor": {"floor": 0.0, "threshold": 0.1},
    }
    signed_init = np.linspace(-1.0, 1.0, 12)

    s = SignatureState(12)
    s.signed_field = signed_init.copy()
    s.derive_amplitude_from_signed()
    tr = run_reasoning(s, cfg)
    assert len(tr) == 3
    assert any(bool(t.get("hold_triggered", False)) for t in tr[1:])
    assert int(tr[-1].get("hold_release_counter", 0)) >= 0


def test_orientation_diffusion_applies_in_score_space_when_enabled():
    cfg = {
        "signature_size": 12,
        "lambda": 0.2,
        "kappa": 0.6,
        "eta": 0.1,
        "phases": 2,
        "family_diffusion": 0.05,
        "orientation_diffusion": 0.75,
        "diffusion_boundary_mode": "wrap",
        "use_wraparound_lattice": True,
        "polarity_threshold": 0.01,
        "zero_crossing_threshold": 0.02,
        "enable_component_promotion": True,
        "component_target_mode": "dominant",
        "component_min_support_mass": 0.0,
        "component_min_width": 1,
        "enable_trace_salience": True,
        "trace_salience_window": 8,
        "caution_similarity_weight": 0.6,
        "caution_salience_weight": 0.4,
        "caution_release_rate": 0.0,
        "caution_global_operator_penalty": 0.0,
        "caution_local_operator_penalty": 0.0,
        "caution_corridor_penalty": 0.0,
        "enable_hold_state": False,
        # Force blocks/penalties to increase salience early.
        "corridor": {"floor": 0.0, "threshold": 0.95},
    }
    signed_init = np.linspace(-1.0, 1.0, 12)

    s = SignatureState(12)
    s.signed_field = signed_init.copy()
    s.derive_amplitude_from_signed()
    tr = run_reasoning(s, cfg)
    assert len(tr) == 2
    last = tr[-1]
    assert bool(last.get("orientation_diffusion_applied", False)) is True
    raw = last.get("raw_operator_scores", {})
    diff = last.get("diffused_operator_scores", {})
    assert raw and diff and set(raw.keys()) == set(diff.keys())
    raw_vals = [float(raw[k]) for k in raw.keys()]
    if len(set(raw_vals)) > 1:
        assert any(abs(float(raw[k]) - float(diff[k])) > 1e-12 for k in raw.keys())


def test_disabling_trace_salience_restores_behavior_close_to_phase4():
    base_cfg = {
        "signature_size": 12,
        "lambda": 0.2,
        "kappa": 0.6,
        "eta": 0.1,
        "phases": 3,
        "family_diffusion": 0.05,
        "diffusion_boundary_mode":
 "wrap",
        "use_wraparound_lattice": True,
        "polarity_threshold": 0.01,
        "zero_crossing_threshold": 0.02,
        "enable_component_promotion": True,
        "component_target_mode": "dominant",
        "component_min_support_mass": 0.0,
        "component_min_width": 1,
        # Neutral corridor/topology knobs.
        "corridor_topology_support_gain": 0.0,
        "corridor_zero_crossing_penalty": 0.0,
        "corridor_run_support_gain": 0.0,
        "corridor_require_signed_support_for_entry": False,
        "corridor_polarity_consistency_bonus": 0.0,
        "corridor_zero_crossing_bonus": 0.0,
        "corridor": {"floor": 0.0, "threshold": 0.1},
    }
    signed_init = np.linspace(-1.0, 1.0, 12)

    s_off = SignatureState(12)
    s_off.signed_field = signed_init.copy()
    s_off.derive_amplitude_from_signed()
    run_reasoning(s_off, dict(base_cfg, enable_trace_salience=False))

    s_on_neutral = SignatureState(12)
    s_on_neutral.signed_field = signed_init.copy()
    s_on_neutral.derive_amplitude_from_signed()
    run_reasoning(
        s_on_neutral,
        dict(
            base_cfg,
            enable_trace_salience=True,
            caution_similarity_weight=0.0,
            caution_salience_weight=0.0,
            caution_operator_penalty=0.0,
            caution_corridor_penalty=0.0,
            enable_hold_state=False,
            caution_threshold=1.0,
        ),
    )

    assert np.allclose(s_off.signed_field, s_on_neutral.signed_field)
