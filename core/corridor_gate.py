
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .peak_detector import Peak, Band
from .polarity_ops import extract_polarity_runs


@dataclass(frozen=True)
class Corridor:
    window: np.ndarray
    entry_resistance: np.ndarray
    exit_resistance: np.ndarray
    threshold: float
    topology_support: np.ndarray
    cancellation_penalty: np.ndarray
    signed_run_count: int
    largest_signed_run_width: int
    topology_bias_applied: bool
    require_signed_support_for_entry: bool
    wraparound_lattice: bool
    caution_window_delta_mean: float
    caution_entry_delta_mean: float
    caution_exit_delta_mean: float
    recovery_support_energy: float
    net_caution_after_recovery: float


def _circular_distance(i: int, j: int, n: int) -> int:
    d = abs(i - j)
    return min(d, n - d)


def _lattice_distance(i: int, j: int, n: int, *, wraparound_lattice: bool) -> int:
    d = abs(int(i) - int(j))
    if wraparound_lattice:
        return min(d, int(n) - d)
    return d


def _mode_run_support(
    *,
    polarity: np.ndarray,
    amplitude: np.ndarray,
    wraparound_lattice: bool = True,
) -> Tuple[np.ndarray, int, int]:
    pol = np.asarray(polarity, dtype=int)
    amp = np.asarray(amplitude, dtype=float)
    if pol.shape != amp.shape:
        raise ValueError("polarity/amplitude shape mismatch")
    n = int(pol.size)
    support = np.zeros(n, dtype=float)
    runs = extract_polarity_runs(pol, amp, wraparound_lattice=wraparound_lattice)
    if not runs:
        return support, 0, 0

    largest_width = max(int(r["width"]) for r in runs)
    for r in runs:
        width = int(r["width"])
        if width <= 0:
            continue
        # Indices covered by the run, including wrap runs.
        start = int(r["start"])
        end = int(r["end"])
        if start <= end:
            idx = list(range(start, end + 1))
        else:
            idx = list(range(start, n)) + list(range(0, end + 1))

        # Reward contiguity: longer runs contribute more to each member slot.
        width_gain = 1.0 + (float(width) - 1.0) / max(1.0, float(n))
        support[idx] += amp[idx] * width_gain

    mx = float(np.max(support)) if support.size else 0.0
    if mx > 0:
        support = np.clip(support / mx, 0.0, 1.0)
    return support, int(len(runs)), int(largest_width)


def _zero_crossing_penalty_field(
    *,
    zero_crossings: List[dict],
    size: int,
    wraparound_lattice: bool = True,
    decay: float = 1.5,
) -> np.ndarray:
    n = int(size)
    penalty = np.zeros(n, dtype=float)
    if not zero_crossings or n <= 0:
        return penalty

    decay = max(0.25, float(decay))
    for z in zero_crossings:
        strength = float(z.get("crossing_strength", 0.0))
        pos = float(z.get("crossing_position", 0.0)) % float(n)
        for i in range(n):
            d = abs(float(i) - pos)
            if bool(wraparound_lattice):
                d = min(d, float(n) - d)
            penalty[i] += strength * float(np.exp(-d / decay))

    mx = float(np.max(penalty)) if penalty.size else 0.0
    if mx > 0:
        penalty = np.clip(penalty / mx, 0.0, 1.0)
    return penalty


def _signed_topology_support(
    *,
    polarity: np.ndarray,
    zero_crossings: List[dict],
    amplitude: np.ndarray,
    size: int,
    wraparound_lattice: bool = True,
    crossing_decay: float = 1.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    amp = np.asarray(amplitude, dtype=float)
    n = int(size)
    if amp.shape != (n,):
        raise ValueError("amplitude shape mismatch")

    if polarity is None:
        pol = np.zeros(n, dtype=int)
    else:
        pol = np.asarray(polarity, dtype=int)
        if pol.shape != (n,):
            pol = np.zeros(n, dtype=int)

    run_support, run_count, largest_width = _mode_run_support(polarity=pol, amplitude=amp, wraparound_lattice=wraparound_lattice)
    crossing_penalty = _zero_crossing_penalty_field(
        zero_crossings=zero_crossings or [], size=n, wraparound_lattice=wraparound_lattice, decay=crossing_decay
    )

    # Topology support is currently run support (Phase 3 v1); keep cancellation separate.
    topology_support = run_support.copy()
    return topology_support, crossing_penalty, run_support, int(run_count), int(largest_width)


def build_dynamic_corridor(
    *,
    size: int,
    peaks: List[Peak],
    bands: List[Band],
    stability: np.ndarray,
    amplitude: np.ndarray = None,
    polarity: np.ndarray = None,
    zero_crossings: List[dict] = None,
    components: List[dict] = None,
    caution_field: np.ndarray = None,
    recovery_field: np.ndarray = None,
    wraparound_lattice: bool = True,
    base_width: float = 1.0,
    width_gain: float = 2.0,
    stability_gain: float = 1.0,
    floor: float = 0.05,
    threshold: float = 0.10,
    exit_scale: float = 0.50,
    corridor_polarity_consistency_bonus: float = 0.0,
    corridor_zero_crossing_bonus: float = 0.0,
    corridor_topology_support_gain: float = 0.0,
    corridor_zero_crossing_penalty: float = 0.0,
    corridor_run_support_gain: float = 0.0,
    corridor_crossing_decay: float = 1.5,
    corridor_require_signed_support_for_entry: bool = False,
    caution_corridor_penalty: float = 0.0,
    caution_entry_penalty: float = None,
    caution_exit_penalty: float = None,
    corridor_recovery_gain: float = 0.0,
    max_recovery_fraction_of_base_window: float = 0.5,
) -> Corridor:
    n = int(size)
    if n <= 0:
        raise ValueError("size must be positive")

    stability = np.asarray(stability, dtype=float)
    if stability.shape != (n,):
        stability = np.ones(n, dtype=float)

    window = np.full(n, float(floor), dtype=float)
    amplitude_arr = None if amplitude is None else np.asarray(amplitude, dtype=float)
    if amplitude_arr is not None and amplitude_arr.shape != (n,):
        amplitude_arr = None

    # Peak-driven corridor (gaussian contributions).
    for p in peaks:
        # If there is an associated band, treat its width/support as widening signal.
        band_width = 0.0
        band_support = 0.0
        for b in bands:
            if int(b.center_family) != int(p.family):
                continue
            band_width = float(b.width)
            if amplitude_arr is not None:
                if int(b.left_family) <= int(b.right_family):
                    band_support = float(np.sum(amplitude_arr[int(b.left_family) : int(b.right_family) + 1]))
                else:
                    if bool(wraparound_lattice):
                        band_support = float(np.sum(amplitude_arr[int(b.left_family) :])) + float(
                            np.sum(amplitude_arr[: int(b.right_family) + 1])
                        )
                    else:
                        lo = min(int(b.left_family), int(b.right_family))
                        hi = max(int(b.left_family), int(b.right_family))
                        band_support = float(np.sum(amplitude_arr[lo : hi + 1]))
            break

        width = (
            float(base_width)
            + float(width_gain) * float(p.amplitude)
            + float(stability_gain) * float(stability[p.family])
            + 0.15 * float(band_width)
            + 0.10 * float(band_support)
        )
        width = max(0.5, width)
        for i in range(n):
            dist = _lattice_distance(i, p.family, n, wraparound_lattice=bool(wraparound_lattice))
            window[i] += float(p.amplitude) * np.exp(-0.5 * (dist / width) ** 2)

    # Band-driven widening (stage-2: scale by width/support).
    for b in bands:
        if int(b.left_family) <= int(b.right_family):
            idx = list(range(int(b.left_family), int(b.right_family) + 1))
        else:
            if bool(wraparound_lattice):
                idx = list(range(int(b.left_family), n)) + list(range(0, int(b.right_family) + 1))
            else:
                lo = min(int(b.left_family), int(b.right_family))
                hi = max(int(b.left_family), int(b.right_family))
                idx = list(range(lo, hi + 1))

        support = 0.0
        if amplitude_arr is not None and idx:
            support = float(np.sum(amplitude_arr[idx]))

        bump = 0.05 * float(b.peak_amplitude) + 0.01 * float(b.width) + 0.02 * float(support)
        for i in idx:
            window[int(i)] += float(bump)

    # Patch2: small structural bonuses (keep them small to avoid destabilizing Stage 1/2 behavior).
    if polarity is not None and float(corridor_polarity_consistency_bonus) != 0.0:
        pol = np.asarray(polarity, dtype=int)
        if pol.shape == (n,):
            if bool(wraparound_lattice):
                left = np.roll(pol, 1)
                right = np.roll(pol, -1)
            else:
                left = np.empty_like(pol)
                right = np.empty_like(pol)
                left[1:] = pol[:-1]
                left[0] = pol[0]
                right[:-1] = pol[1:]
                right[-1] = pol[-1]
            same_left = (pol != 0) & (pol == left)
            same_right = (pol != 0) & (pol == right)
            consistency = 0.5 * same_left.astype(float) + 0.5 * same_right.astype(float)
            window += float(corridor_polarity_consistency_bonus) * consistency

    if zero_crossings and float(corridor_zero_crossing_bonus) != 0.0:
        # Proximity to cancellation axes (zero crossings) as a soft bias.
        prox = np.zeros(n, dtype=float)
        for z in zero_crossings:
            pos = float(z.get("crossing_position", 0.0)) % float(n)
            for i in range(n):
                d = abs(float(i) - pos)
                if bool(wraparound_lattice):
                    d = min(d, float(n) - d)
                prox[i] = max(prox[i], float(np.exp(-d / 1.5)))
        window += float(corridor_zero_crossing_bonus) * prox

    topology_support = np.zeros(n, dtype=float)
    cancellation_penalty = np.zeros(n, dtype=float)
    signed_run_count = 0
    largest_signed_run_width = 0
    topology_bias_applied = (
        float(corridor_topology_support_gain) != 0.0
        or float(corridor_zero_crossing_penalty) != 0.0
        or float(corridor_run_support_gain) != 0.0
        or bool(corridor_require_signed_support_for_entry)
    )

    if (polarity is not None or zero_crossings or components) and (amplitude_arr is not None):
        if components:
            # Prefer explicit components if supplied.
            comp_support = np.zeros(n, dtype=float)
            comp_widths: List[int] = []
            comp_count = 0
            for c in components:
                start = int(c.get("start_family", 0))
                end = int(c.get("end_family", 0))
                if start <= end:
                    idx = list(range(start, end + 1))
                else:
                    if bool(wraparound_lattice):
                        idx = list(range(start, n)) + list(range(0, end + 1))
                    else:
                        idx = list(range(start, n))
                if not idx:
                    continue
                comp_count += 1
                comp_widths.append(int(c.get("width", len(idx))))
                mass = float(c.get("support_mass", 0.0))
                comp_support[idx] += (amplitude_arr[idx] + 1e-9) * max(0.0, mass)
            mx = float(np.max(comp_support)) if comp_support.size else 0.0
            if mx > 0:
                comp_support = np.clip(comp_support / mx, 0.0, 1.0)

            topology_support = comp_support
            cancellation_penalty = _zero_crossing_penalty_field(
                zero_crossings=zero_crossings or [], size=n, wraparound_lattice=wraparound_lattice, decay=float(corridor_crossing_decay)
            )
            run_support = comp_support
            signed_run_count = int(comp_count)
            largest_signed_run_width = int(max(comp_widths) if comp_widths else 0)
        else:
            topology_support, cancellation_penalty, run_support, signed_run_count, largest_signed_run_width = _signed_topology_support(
                polarity=polarity,
                zero_crossings=zero_crossings,
                amplitude=amplitude_arr,
                size=n,
                wraparound_lattice=wraparound_lattice,
                crossing_decay=float(corridor_crossing_decay),
            )

        # Re-anchor window to signed topology.
        window = window + float(corridor_topology_support_gain) * topology_support + float(corridor_run_support_gain) * run_support
        window = window - float(corridor_zero_crossing_penalty) * cancellation_penalty
        window = np.maximum(0.0, window)

    base_window_pre_caution = window.copy()

    # Patch5/5b: caution narrows the corridor before normalization (use bounded caution_field).
    caution_arr = None if caution_field is None else np.asarray(caution_field, dtype=float)
    entry_pen = float(caution_corridor_penalty) if caution_entry_penalty is None else float(caution_entry_penalty)
    exit_pen = float(caution_corridor_penalty) if caution_exit_penalty is None else float(caution_exit_penalty)

    caution_window_delta_mean = 0.0
    caution_entry_delta_mean = 0.0
    caution_exit_delta_mean = 0.0

    if caution_arr is not None and caution_arr.shape == (n,):
        c = np.clip(caution_arr, 0.0, 1.0)
        if entry_pen != 0.0:
            window_before = window.copy()
            window = np.maximum(0.0, window - float(entry_pen) * c)
            caution_window_delta_mean = float(np.mean(window_before - window))

    # Phase7: graded local recovery can reopen the corridor toward its pre-caution base window (never beyond).
    recovery_support_energy = 0.0
    net_caution_after_recovery = float(np.mean(np.clip(caution_arr, 0.0, 1.0))) if (caution_arr is not None and caution_arr.shape == (n,)) else 0.0
    r_arr = None if recovery_field is None else np.asarray(recovery_field, dtype=float)
    if r_arr is not None and r_arr.shape == (n,) and float(corridor_recovery_gain) != 0.0:
        r = np.clip(r_arr, 0.0, 1.0)
        base = np.clip(np.asarray(base_window_pre_caution, dtype=float), 0.0, None)
        cap = float(np.clip(float(max_recovery_fraction_of_base_window), 0.0, 1.0))
        reopen = float(corridor_recovery_gain) * r * base
        reopen = np.minimum(reopen, cap * base)
        recovery_support_energy = float(np.sum(np.abs(reopen)))
        window = np.minimum(base, window + reopen)
        net_caution_after_recovery = float(np.clip(net_caution_after_recovery * (1.0 - float(np.mean(r))), 0.0, 1.0))

    # Normalize into [0, 1] using the pre-caution base window scale when available.
    # This keeps recovery from inflating non-peak slots above their base-normalized values.
    mx_base = float(np.max(base_window_pre_caution)) if base_window_pre_caution.size else 1.0
    mx = mx_base if mx_base > 0 else (float(np.max(window)) if window.size else 1.0)
    if mx > 0:
        window = np.clip(window / mx, 0.0, 1.0)
    else:
        window = np.zeros(n, dtype=float)

    entry = np.clip(1.0 - window, 0.0, 1.0)
    exit_r = np.clip((1.0 - window) * float(exit_scale), 0.0, 1.0)
    if caution_arr is not None and caution_arr.shape == (n,):
        c = np.clip(caution_arr, 0.0, 1.0)
        if entry_pen != 0.0:
            entry_before = entry.copy()
            entry = np.clip(entry + 0.25 * float(entry_pen) * c, 0.0, 1.0)
            caution_entry_delta_mean = float(np.mean(entry - entry_before))
        if exit_pen != 0.0:
            exit_before = exit_r.copy()
            exit_r = np.clip(exit_r + 0.10 * float(exit_pen) * c, 0.0, 1.0)
            caution_exit_delta_mean = float(np.mean(exit_r - exit_before))
    return Corridor(
        window=window,
        entry_resistance=entry,
        exit_resistance=exit_r,
        threshold=float(threshold),
        topology_support=topology_support,
        cancellation_penalty=cancellation_penalty,
        signed_run_count=int(signed_run_count),
        largest_signed_run_width=int(largest_signed_run_width),
        topology_bias_applied=bool(topology_bias_applied),
        require_signed_support_for_entry=bool(corridor_require_signed_support_for_entry),
        wraparound_lattice=bool(wraparound_lattice),
        caution_window_delta_mean=float(caution_window_delta_mean),
        caution_entry_delta_mean=float(caution_entry_delta_mean),
        caution_exit_delta_mean=float(caution_exit_delta_mean),
        recovery_support_energy=float(recovery_support_energy),
        net_caution_after_recovery=float(net_caution_after_recovery),
    )


def apply_gate(
    *,
    current: np.ndarray,
    projected: np.ndarray,
    corridor: Corridor,
) -> Tuple[np.ndarray, np.ndarray, List[int], List[int]]:
    cur = np.asarray(current, dtype=float)
    proj = np.asarray(projected, dtype=float)
    if cur.shape != proj.shape:
        raise ValueError("current/projected shape mismatch")

    delta = proj - cur
    is_entry = delta >= 0.0
    resist = np.where(is_entry, corridor.entry_resistance, corridor.exit_resistance)
    gate_factor = corridor.window * (1.0 - resist)
    # Admissibility threshold: block entry updates outside the corridor.
    gate_factor = np.where((is_entry) & (corridor.window < corridor.threshold), 0.0, gate_factor)

    # Phase-3: optionally require signed-topology support for entry updates.
    if corridor.require_signed_support_for_entry and getattr(corridor, "topology_support", None) is not None:
        support = np.asarray(corridor.topology_support, dtype=float)
        if support.shape == gate_factor.shape:
            entry_factor = 0.2 + 0.8 * np.clip(support, 0.0, 1.0)
            gate_factor = np.where(is_entry, gate_factor * entry_factor, gate_factor)

    gate_factor = np.clip(gate_factor, 0.0, 1.0)
    gated_delta = delta * gate_factor

    boundary_penalty = np.maximum(0.0, delta) * (1.0 - corridor.window)

    hits = np.where((corridor.window >= corridor.threshold) & (np.abs(gated_delta) > 1e-12))[0].tolist()
    blocks = np.where((corridor.window < corridor.threshold) & (delta > 1e-12))[0].tolist()

    return gated_delta, boundary_penalty, [int(i) for i in hits], [int(i) for i in blocks]
