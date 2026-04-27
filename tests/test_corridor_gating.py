
import numpy as np

from core.corridor_gate import apply_gate, build_dynamic_corridor
from core.peak_detector import Peak


def test_corridor_window_is_peak_centered_and_resistances_asymmetric():
    peaks = [Peak(family=5, amplitude=1.0)]
    corridor = build_dynamic_corridor(
        size=12,
        peaks=peaks,
        bands=[],
        stability=np.ones(12),
        base_width=1.0,
        width_gain=2.0,
        stability_gain=1.0,
        floor=0.0,
        threshold=0.1,
        exit_scale=0.5,
    )
    assert corridor.window[5] == max(corridor.window)
    assert np.all(corridor.entry_resistance >= corridor.exit_resistance)


def test_gate_blocks_updates_outside_threshold():
    cur = np.zeros(12)
    proj = np.zeros(12)
    proj[0] = 1.0  # attempt to add energy outside corridor
    corridor = build_dynamic_corridor(
        size=12,
        peaks=[Peak(family=6, amplitude=1.0)],
        bands=[],
        stability=np.ones(12),
        floor=0.0,
        threshold=0.9,  # make corridor very strict
    )
    gated_delta, penalty, hits, blocks = apply_gate(current=cur, projected=proj, corridor=corridor)
    assert gated_delta[0] == 0.0
    assert penalty[0] > 0.0
    assert 0 in blocks


def test_topology_runs_widen_corridor_when_enabled():
    amp = np.full(12, 0.4, dtype=float)
    peaks = [Peak(family=3, amplitude=0.4)]
    pol_run = np.zeros(12, dtype=int)
    pol_run[2:6] = 1

    base = build_dynamic_corridor(
        size=12,
        peaks=peaks,
        bands=[],
        stability=np.ones(12),
        amplitude=amp,
        polarity=np.zeros(12, dtype=int),
        zero_crossings=[],
        floor=0.0,
        threshold=0.1,
        corridor_topology_support_gain=0.0,
        corridor_run_support_gain=0.0,
        corridor_zero_crossing_penalty=0.0,
    )
    topo = build_dynamic_corridor(
        size=12,
        peaks=peaks,
        bands=[],
        stability=np.ones(12),
        amplitude=amp,
        polarity=pol_run,
        zero_crossings=[],
        floor=0.0,
        threshold=0.1,
        corridor_topology_support_gain=0.0,
        corridor_run_support_gain=0.25,
        corridor_zero_crossing_penalty=0.0,
    )

    run_idx = [2, 3, 4, 5]
    assert float(np.mean(topo.window[run_idx])) > float(np.mean(base.window[run_idx]))


def test_zero_crossing_penalty_narrows_corridor_locally():
    amp = np.full(12, 0.6, dtype=float)
    peaks = [Peak(family=5, amplitude=0.6)]
    pol = np.zeros(12, dtype=int)
    pol[:6] = 1
    pol[6:] = -1
    zc = [
        {
            "left_index": 5,
            "right_index": 6,
            "crossing_position": 5.5,
            "crossing_strength": 1.0,
            "left_polarity": 1,
            "right_polarity": -1,
        }
    ]

    no_pen = build_dynamic_corridor(
        size=12,
        peaks=peaks,
        bands=[],
        stability=np.ones(12),
        amplitude=amp,
        polarity=pol,
        zero_crossings=zc,
        floor=0.0,
        threshold=0.1,
        corridor_zero_crossing_penalty=0.0,
    )
    with_pen = build_dynamic_corridor(
        size=12,
        peaks=peaks,
        bands=[],
        stability=np.ones(12),
        amplitude=amp,
        polarity=pol,
        zero_crossings=zc,
        floor=0.0,
        threshold=0.1,
        corridor_zero_crossing_penalty=0.5,
        corridor_crossing_decay=1.5,
    )

    near = [5, 6]
    far = [0, 11]
    assert float(np.mean(with_pen.window[near])) < float(np.mean(no_pen.window[near]))
    assert float(np.mean(with_pen.window[near])) < float(np.mean(with_pen.window[far]))


def test_magnitude_only_is_unchanged_when_topology_gains_zero():
    amp = np.linspace(0.2, 0.8, 12)
    peaks = [Peak(family=2, amplitude=float(amp[2]))]
    pol = np.where(np.arange(12) < 6, 1, -1)
    zc = [
        {
            "left_index": 5,
            "right_index": 6,
            "crossing_position": 5.5,
            "crossing_strength": 0.5,
            "left_polarity": 1,
            "right_polarity": -1,
        }
    ]

    baseline = build_dynamic_corridor(
        size=12,
        peaks=peaks,
        bands=[],
        stability=np.ones(12),
        amplitude=amp,
        polarity=None,
        zero_crossings=None,
        corridor_polarity_consistency_bonus=0.0,
        corridor_zero_crossing_bonus=0.0,
        corridor_topology_support_gain=0.0,
        corridor_zero_crossing_penalty=0.0,
        corridor_run_support_gain=0.0,
        corridor_require_signed_support_for_entry=False,
        floor=0.0,
        threshold=0.1,
    )
    topo_zero = build_dynamic_corridor(
        size=12,
        peaks=peaks,
        bands=[],
        stability=np.ones(12),
        amplitude=amp,
        polarity=pol,
        zero_crossings=zc,
        corridor_polarity_consistency_bonus=0.0,
        corridor_zero_crossing_bonus=0.0,
        corridor_topology_support_gain=0.0,
        corridor_zero_crossing_penalty=0.0,
        corridor_run_support_gain=0.0,
        corridor_require_signed_support_for_entry=False,
        floor=0.0,
        threshold=0.1,
    )
    assert np.allclose(baseline.window, topo_zero.window)


def test_entry_updates_reduced_when_signed_support_required():
    amp = np.ones(12, dtype=float)
    peaks = [Peak(family=0, amplitude=1.0)]

    corridor_open = build_dynamic_corridor(
        size=12,
        peaks=peaks,
        bands=[],
        stability=np.ones(12),
        amplitude=amp,
        polarity=np.zeros(12, dtype=int),
        zero_crossings=[],
        floor=0.0,
        threshold=0.1,
        corridor_require_signed_support_for_entry=False,
    )
    corridor_req = build_dynamic_corridor(
        size=12,
        peaks=peaks,
        bands=[],
        stability=np.ones(12),
        amplitude=amp,
        polarity=np.zeros(12, dtype=int),
        zero_crossings=[],
        floor=0.0,
        threshold=0.1,
        corridor_require_signed_support_for_entry=True,
    )

    cur = np.zeros(12)
    proj = np.zeros(12)
    proj[0] = 1.0
    gated_open, _, _, _ = apply_gate(current=cur, projected=proj, corridor=corridor_open)
    gated_req, _, _, _ = apply_gate(current=cur, projected=proj, corridor=corridor_req)
    assert gated_req[0] < gated_open[0]


def test_non_wrap_lattice_does_not_wrap_corridor_distance():
    peaks = [Peak(family=0, amplitude=1.0)]
    wrap = build_dynamic_corridor(
        size=12,
        peaks=peaks,
        bands=[],
        stability=np.ones(12),
        floor=0.0,
        threshold=0.1,
        base_width=0.5,
        width_gain=0.0,
        stability_gain=0.0,
        wraparound_lattice=True,
    )
    nowrap = build_dynamic_corridor(
        size=12,
        peaks=peaks,
        bands=[],
        stability=np.ones(12),
        floor=0.0,
        threshold=0.1,
        base_width=0.5,
        width_gain=0.0,
        stability_gain=0.0,
        wraparound_lattice=False,
    )
    assert float(wrap.window[11]) > float(nowrap.window[11])


def test_entry_and_exit_penalties_can_be_tuned_independently():
    peaks = [Peak(family=6, amplitude=1.0)]
    caution = np.zeros(12, dtype=float)
    caution[6] = 1.0

    base = build_dynamic_corridor(
        size=12,
        peaks=peaks,
        bands=[],
        stability=np.ones(12),
        amplitude=np.ones(12),
        floor=0.0,
        threshold=0.1,
        caution_field=caution,
        caution_corridor_penalty=0.0,
        caution_entry_penalty=0.8,
        caution_exit_penalty=0.0,
    )
    more_exit = build_dynamic_corridor(
        size=12,
        peaks=peaks,
        bands=[],
        stability=np.ones(12),
        amplitude=np.ones(12),
        floor=0.0,
        threshold=0.1,
        caution_field=caution,
        caution_corridor_penalty=0.0,
        caution_entry_penalty=0.8,
        caution_exit_penalty=0.8,
    )

    assert float(more_exit.window[6]) == float(base.window[6])
    assert float(more_exit.entry_resistance[6]) == float(base.entry_resistance[6])
    assert float(more_exit.exit_resistance[6]) > float(base.exit_resistance[6])


def test_caution_field_values_are_bounded_before_application():
    peaks = [Peak(family=6, amplitude=1.0)]
    caution_hi = np.zeros(12, dtype=float)
    caution_hi[6] = 10.0
    caution_one = np.zeros(12, dtype=float)
    caution_one[6] = 1.0

    hi = build_dynamic_corridor(
        size=12,
        peaks=peaks,
        bands=[],
        stability=np.ones(12),
        amplitude=np.ones(12),
        floor=0.0,
        threshold=0.1,
        caution_field=caution_hi,
        caution_corridor_penalty=0.0,
        caution_entry_penalty=0.8,
        caution_exit_penalty=0.8,
    )
    one = build_dynamic_corridor(
        size=12,
        peaks=peaks,
        bands=[],
        stability=np.ones(12),
        amplitude=np.ones(12),
        floor=0.0,
        threshold=0.1,
        caution_field=caution_one,
        caution_corridor_penalty=0.0,
        caution_entry_penalty=0.8,
        caution_exit_penalty=0.8,
    )

    assert np.allclose(hi.window, one.window)


def test_non_wrap_topology_support_does_not_merge_edge_runs():
    amp = np.ones(12, dtype=float)
    peaks = [Peak(family=0, amplitude=1.0)]
    pol = np.zeros(12, dtype=int)
    pol[10:] = 1
    pol[:2] = 1

    wrap = build_dynamic_corridor(
        size=12,
        peaks=peaks,
        bands=[],
        stability=np.ones(12),
        amplitude=amp,
        polarity=pol,
        zero_crossings=[],
        floor=0.0,
        threshold=0.1,
        corridor_topology_support_gain=0.0,
        corridor_run_support_gain=0.5,
        corridor_zero_crossing_penalty=0.0,
        wraparound_lattice=True,
    )
    nowrap = build_dynamic_corridor(
        size=12,
        peaks=peaks,
        bands=[],
        stability=np.ones(12),
        amplitude=amp,
        polarity=pol,
        zero_crossings=[],
        floor=0.0,
        threshold=0.1,
        corridor_topology_support_gain=0.0,
        corridor_run_support_gain=0.5,

        corridor_zero_crossing_penalty=0.0,
        wraparound_lattice=False,
    )

    edge_idx = [10, 11, 0, 1]
    assert float(np.mean(wrap.window[edge_idx])) != float(np.mean(nowrap.window[edge_idx]))


def test_cancellation_penalty_respects_lattice_mode():
    amp = np.ones(12, dtype=float)
    peaks = [Peak(family=6, amplitude=1.0)]
    pol = np.zeros(12, dtype=int)
    zc = [
        {
            "left_index": 0,
            "right_index": 1,
            "crossing_position": 0.5,
            "crossing_strength": 1.0,
            "left_polarity": 1,
            "right_polarity": -1,
        }
    ]

    wrap = build_dynamic_corridor(
        size=12,
        peaks=peaks,
        bands=[],
        stability=np.ones(12),
        amplitude=amp,
        polarity=pol,
        zero_crossings=zc,
        floor=0.0,
        threshold=0.1,
        corridor_topology_support_gain=0.0,
        corridor_run_support_gain=0.0,
        corridor_zero_crossing_penalty=0.9,
        wraparound_lattice=True,
    )
    nowrap = build_dynamic_corridor(
        size=12,
        peaks=peaks,
        bands=[],
        stability=np.ones(12),
        amplitude=amp,
        polarity=pol,
        zero_crossings=zc,
        floor=0.0,
        threshold=0.1,
        corridor_topology_support_gain=0.0,
        corridor_run_support_gain=0.0,
        corridor_zero_crossing_penalty=0.9,
        wraparound_lattice=False,
    )

    assert float(wrap.window[11]) < float(nowrap.window[11])


def test_recovery_reopens_corridor_locally_but_never_above_base_window():
    peaks = [Peak(family=6, amplitude=1.0)]
    amp = np.ones(12, dtype=float)
    caution = np.zeros(12, dtype=float)
    caution[6] = 1.0
    recovery = np.zeros(12, dtype=float)
    recovery[6] = 1.0

    base = build_dynamic_corridor(
        size=12,
        peaks=peaks,
        bands=[],
        stability=np.ones(12),
        amplitude=amp,
        floor=0.0,
        threshold=0.1,
        caution_corridor_penalty=0.0,
    )
    caut = build_dynamic_corridor(
        size=12,
        peaks=peaks,
        bands=[],
        stability=np.ones(12),
        amplitude=amp,
        floor=0.0,
        threshold=0.1,
        caution_field=caution,
        caution_corridor_penalty=0.0,
        caution_entry_penalty=0.8,
        caution_exit_penalty=0.0,
    )
    rec = build_dynamic_corridor(
        size=12,
        peaks=peaks,
        bands=[],
        stability=np.ones(12),
        amplitude=amp,
        floor=0.0,
        threshold=0.1,
        caution_field=caution,
        recovery_field=recovery,
        caution_corridor_penalty=0.0,
        caution_entry_penalty=0.8,
        caution_exit_penalty=0.0,
        corridor_recovery_gain=0.5,
        max_recovery_fraction_of_base_window=0.5,
    )

    assert float(rec.window[6]) > float(caut.window[6])
    assert np.all(rec.window <= base.window + 1e-12)
