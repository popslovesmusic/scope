import numpy as np

from core.component_ops import (
    build_component_masks,
    component_recovery_scores,
    component_caution_scores,
    extract_signed_components,
    match_components_across_phases,
)
from core.orientation_ops import OPERATORS_BY_NAME, apply_operator_to_masked_region
from core.polarity_ops import assign_polarity, compute_mode_index, detect_zero_crossings
from core.reasoning_loop import run_reasoning
from core.signature_state import SignatureState


def test_extracts_one_component_per_contiguous_run():
    signed = np.array([0.2, 0.3, 0.0, -0.2, -0.3, 0.0, 0.4, 0.5, 0.0, 0.0, 0.0, 0.0])
    amp = np.abs(signed)
    pol = assign_polarity(signed, polarity_threshold=0.01)
    mode = compute_mode_index(pol, wraparound_lattice=True)
    zc = detect_zero_crossings(signed, pol, zero_crossing_threshold=0.02, wraparound_lattice=True)
    comps = extract_signed_components(
        signed_field=signed,
        amplitude=amp,
        polarity=pol,
        mode_index=mode,
        zero_crossings=zc,
        wraparound_lattice=True,
        component_min_support_mass=0.0,
        component_min_width=1,
    )
    # Runs: + (0-1), - (3-4), + (6-7)
    assert len(comps) == 3


def test_wraparound_same_sign_runs_merge_into_one_component():
    signed = np.zeros(12)
    signed[10:] = 0.3
    signed[:2] = 0.25
    amp = np.abs(signed)
    pol = assign_polarity(signed, polarity_threshold=0.01)
    mode = compute_mode_index(pol, wraparound_lattice=True)
    zc = detect_zero_crossings(signed, pol, zero_crossing_threshold=0.02, wraparound_lattice=True)
    comps = extract_signed_components(
        signed_field=signed,
        amplitude=amp,
        polarity=pol,
        mode_index=mode,
        zero_crossings=zc,
        wraparound_lattice=True,
        component_min_support_mass=0.0,
        component_min_width=1,
    )
    # One wraparound positive component.
    assert len(comps) == 1
    assert comps[0]["width"] == 4


def test_component_matching_preserves_identity_under_small_center_drift():
    prev = [
        {
            "stable_id": 7,
            "sign": 1,
            "start_family": 2,
            "end_family": 4,
            "width": 3,
            "center_family": 3.0,
        }
    ]
    cur = [
        {
            "component_id": 0,
            "stable_id": None,
            "sign": 1,
            "start_family": 3,
            "end_family": 5,
            "width": 3,
            "center_family": 4.0,
        }
    ]
    matched, meta = match_components_across_phases(previous_components=prev, current_components=cur, size=12)
    assert matched[0]["stable_id"] == 7
    assert meta["component_identity_persistence"] >= 1.0 / 1.0


def test_masked_operator_application_changes_only_targeted_region():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    mask = np.array([0.0, 1.0, 1.0, 0.0])
    op = OPERATORS_BY_NAME["--"]
    y = apply_operator_to_masked_region(x, op, mask)
    # Outside mask: unchanged.
    assert y[0] == x[0]
    assert y[3] == x[3]
    # Inside mask: should differ for a reversing operator on an asymmetric vector.
    assert not np.allclose(y[1:3], x[1:3])


def test_global_only_mode_matches_disabled_component_promotion_behavior():
    base_cfg = {
        "signature_size": 12,
        "lambda": 0.2,
        "kappa": 0.6,
        "eta": 0.1,
        "phases": 2,
        "peak_threshold": 0.1,
        "family_diffusion": 0.05,
        "diffusion_boundary_mode": "wrap",
        "use_wraparound_lattice": True,
        "polarity_threshold": 0.01,
        "zero_crossing_threshold": 0.02,
        "corridor_topology_support_gain": 0.0,
        "corridor_zero_crossing_penalty": 0.0,
        "corridor_run_support_gain": 0.0,
        "corridor_require_signed_support_for_entry": False,
        "corridor": {"floor": 0.0, "threshold": 0.1},
        "enable_polarized_signature_summary": True,
    }
    signed_init = np.linspace(-1.0, 1.0, 12)

    s1 = SignatureState(12)
    s1.signed_field = signed_init.copy()
    s1.derive_amplitude_from_signed()
    tr1 = run_reasoning(s1, dict(base_cfg, enable_component_promotion=False))

    s2 = SignatureState(12)
    s2.signed_field = signed_init.copy()
    s2.derive_amplitude_from_signed()
    tr2 = run_reasoning(
        s2,
        dict(
            base_cfg,
            enable_component_promotion=True,
            component_target_mode="global_only",
            component_global_blend=1.0,
            component_local_blend=0.0,
        ),
    )

    assert np.allclose(s1.signed_field, s2.signed_field)
    assert len(tr1) == len(tr2) == 2


def test_component_caution_score_mode_mean_works():
    components = [{"stable_id": 0, "component_id": 0, "start_family": 0, "end_family": 2}]
    caution = np.zeros(12, dtype=float)
    caution[1] = 1.0
    scores = component_caution_scores(components=components, caution_field=caution, size=12, mode="mean")
    assert abs(float(scores[0]["mean"]) - (1.0 / 3.0)) < 1e-12
    assert abs(float(scores[0]["score"]) - (1.0 / 3.0)) < 1e-12


def test_component_caution_score_mode_peak_works():
    components = [{"stable_id": 0, "component_id": 0, "start_family": 0, "end_family": 2}]
    caution = np.zeros(12, dtype=float)
    caution[1] = 1.0
    scores = component_caution_scores(components=components, caution_field=caution, size=12, mode="peak")
    assert abs(float(scores[0]["peak"]) - 1.0) < 1e-12
    assert abs(float(scores[0]["score"]) - 1.0) < 1e-12


def test_component_caution_score_mode_blended_works():
    components = [{"stable_id": 0, "component_id": 0, "start_family": 0, "end_family": 2}]
    caution = np.zeros(12, dtype=float)
    caution[1] = 1.0
    scores = component_caution_scores(components=components, caution_field=caution, size=12, mode="blended", blended_alpha=0.5)
    expected = 0.5 * (1.0 / 3.0) + 0.5 * 1.0
    assert abs(float(scores[0]["score"]) - expected) < 1e-12


def test_component_recovery_scores_favor_safe_component():
    components = [
        {"stable_id": 0, "component_id": 0, "start_family": 0, "end_family": 2},
        {"stable_id": 1, "component_id": 1, "start_family": 6, "end_family": 8},
    ]
    caution = np.zeros(12, dtype=float)
    caution[0:3] = 0.1
    caution[6:9] = 0.9
    rec = component_recovery_scores(components=components, caution_field=caution, size=12, mode="mean")
    assert float(rec[0]) > float(rec[1])
