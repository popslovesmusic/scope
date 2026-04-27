
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Tuple

import numpy as np

from .corridor_gate import Corridor, apply_gate, build_dynamic_corridor
from .component_ops import (
    build_component_masks,
    component_recovery_scores,
    component_caution_scores,
    extract_signed_components,
    match_components_across_phases,
    summarize_component_set,
)
from .diffusion_ops import apply_family_diffusion, diffuse_operator_scores
from .orientation_ops import OPERATORS, OrientationOperator
from .peak_detector import Band, Peak, detect_peaks_and_bands
from .polarity_ops import assign_polarity, compute_mode_index, detect_zero_crossings, dominant_polarity, polarity_shift_metric
from .signature_state import SignatureState
from .trace_ops import (
    compute_caution_field,
    compute_recontextualization_score,
    compute_recovery_signal,
    compute_trace_salience,
)


def _config_get(config: Dict[str, Any], key: str, default: Any) -> Any:
    return config[key] if key in config else default


def _project_signature(
    read_signed: np.ndarray,
    read_amplitude: np.ndarray,
    *,
    peaks: List[Peak],
    peak_gain: float,
    neighbor_gain: float,
) -> np.ndarray:
    signed = np.asarray(read_signed, dtype=float)
    amp = np.asarray(read_amplitude, dtype=float)
    proj = signed.copy()
    n = int(proj.size)
    for p in peaks:
        idx = int(p.family)
        sgn = 1.0 if signed[idx] >= 0.0 else -1.0
        strength = float(amp[idx])
        proj[idx] += float(peak_gain) * sgn * strength
        proj[(idx - 1) % n] += float(neighbor_gain) * sgn * strength
        proj[(idx + 1) % n] += float(neighbor_gain) * sgn * strength
    return proj


def _score_operator(
    state: SignatureState,
    config: Dict[str, Any],
    op: OrientationOperator,
    active_mask_native,
    caution_field_native,
    recovery_field_native,
) -> Tuple[float, Dict[str, Any]]:
    signed_read = op.read(state.signed_field)
    amp_read = np.abs(np.asarray(signed_read, dtype=float))

    peak_cfg = _config_get(config, "peak_detection", {})
    peaks, bands = detect_peaks_and_bands(
        amp_read,
        min_height=float(_config_get(peak_cfg, "min_height", _config_get(config, "peak_threshold", 0.10))),
        min_distance=int(_config_get(peak_cfg, "min_distance", 1)),
        merge_radius=int(_config_get(peak_cfg, "merge_radius", 1)),
        band_rel_threshold=float(_config_get(config, "band_rel_threshold", 0.50)),
        wraparound_lattice=bool(_config_get(config, "use_wraparound_lattice", True)),
    )

    wraparound_lattice = bool(_config_get(config, "use_wraparound_lattice", True))
    polarity_threshold = float(_config_get(config, "polarity_threshold", 0.01))
    zero_crossing_threshold = float(_config_get(config, "zero_crossing_threshold", 0.02))
    pol_read = assign_polarity(signed_read, polarity_threshold=polarity_threshold)
    zc_read = detect_zero_crossings(
        signed_read,
        pol_read,
        zero_crossing_threshold=zero_crossing_threshold,
        wraparound_lattice=wraparound_lattice,
    )

    enable_components = bool(_config_get(config, "enable_component_promotion", False))
    comp_min_mass = float(_config_get(config, "component_min_support_mass", 0.05))
    comp_min_width = int(_config_get(config, "component_min_width", 1))
    components_read = None
    if enable_components:
        mode_read = compute_mode_index(pol_read, wraparound_lattice=wraparound_lattice)
        components_read = extract_signed_components(
            signed_field=signed_read,
            amplitude=amp_read,
            polarity=pol_read,
            mode_index=mode_read,
            zero_crossings=zc_read,
            wraparound_lattice=wraparound_lattice,
            component_min_support_mass=comp_min_mass,
            component_min_width=comp_min_width,
        )

    corridor_cfg = _config_get(config, "corridor", {})
    corridor_crossing_decay = float(_config_get(config, "corridor_crossing_decay", 1.5))
    caution_read = None
    if caution_field_native is not None:
        c_native = np.asarray(caution_field_native, dtype=float)
        if c_native.shape == signed_read.shape:
            caution_read = op.read(c_native)
    recovery_read = None
    if recovery_field_native is not None:
        r_native = np.asarray(recovery_field_native, dtype=float)
        if r_native.shape == signed_read.shape:
            recovery_read = op.read(r_native)
    corridor = build_dynamic_corridor(
        size=int(amp_read.size),
        peaks=peaks,
        bands=bands,
        stability=op.read(state.stability),
        amplitude=amp_read,
        polarity=pol_read,
        zero_crossings=zc_read,
        components=components_read,
        caution_field=caution_read,
        wraparound_lattice=wraparound_lattice,
        base_width=float(_config_get(corridor_cfg, "base_width", 1.0)),
        width_gain=float(_config_get(corridor_cfg, "width_gain", 2.0)),
        stability_gain=float(_config_get(corridor_cfg, "stability_gain", 1.0)),
        floor=float(_config_get(corridor_cfg, "floor", 0.05)),
        threshold=float(_config_get(corridor_cfg, "threshold", 0.10)),
        exit_scale=float(_config_get(corridor_cfg, "exit_scale", 0.50)),
        corridor_polarity_consistency_bonus=float(_config_get(config, "corridor_polarity_consistency_bonus", 0.0)),
        corridor_zero_crossing_bonus=float(_config_get(config, "corridor_zero_crossing_bonus", 0.0)),
        corridor_topology_support_gain=float(_config_get(config, "corridor_topology_support_gain", 0.0)),
        corridor_zero_crossing_penalty=float(_config_get(config, "corridor_zero_crossing_penalty", 0.0)),
        corridor_run_support_gain=float(_config_get(config, "corridor_run_support_gain", 0.0)),
        corridor_crossing_decay=corridor_crossing_decay,
        corridor_require_signed_support_for_entry=bool(_config_get(config, "corridor_require_signed_support_for_entry", False)),
        caution_corridor_penalty=float(_config_get(config, "caution_corridor_penalty", 0.0)),
        caution_entry_penalty=_config_get(config, "caution_entry_penalty", None),
        caution_exit_penalty=_config_get(config, "caution_exit_penalty", None),
        recovery_field=recovery_read,
        corridor_recovery_gain=float(_config_get(config, "corridor_recovery_gain", 0.0)),
        max_recovery_fraction_of_base_window=float(_config_get(config, "max_recovery_fraction_of_base_window", 0.5)),
    )

    proj_cfg = _config_get(config, "projection", {})
    projected = _project_signature(
        signed_read,
        amp_read,
        peaks=peaks,
        peak_gain=float(_config_get(proj_cfg, "peak_gain", 0.25)),
        neighbor_gain=float(_config_get(proj_cfg, "neighbor_gain", 0.10)),
    )

    component_global_blend = float(_config_get(config, "component_global_blend", 1.0))
    component_local_blend = float(_config_get(config, "component_local_blend", 0.0))
    local_blend_eff = float(component_local_blend)
    operator_component_score = 0.0
    if enable_components and active_mask_native is not None:
        mask_native = np.asarray(active_mask_native, dtype=float)
        if mask_native.shape == signed_read.shape:
            mask_read = op.read(mask_native)
            if caution_read is not None:
                c = np.clip(np.asarray(caution_read, dtype=float), 0.0, 1.0)
                avg_caution = float(np.mean(c[mask_read > 0.5])) if np.any(mask_read > 0.5) else 0.0
                local_blend_eff = float(local_blend_eff) * max(0.0, 1.0 - avg_caution)
            delta = projected - signed_read
            # Blend: keep full delta on the component, damp elsewhere.
            projected = signed_read + (component_global_blend * delta) + (local_blend_eff * delta * (mask_read > 0.5))

    gated_delta, boundary_penalty, hits, blocks = apply_gate(current=signed_read, projected=projected, corridor=corridor)

    if enable_components and active_mask_native is not None:
        mask_native = np.asarray(active_mask_native, dtype=float)
        if mask_native.shape == signed_read.shape:
            mask_read = op.read(mask_native)
            operator_component_score = float(np.sum(np.abs(gated_delta) * (mask_read > 0.5)))

    penalty_weight = float(_config_get(config, "boundary_penalty_weight", 1.0))
    global_score_pre = float(np.sum(np.abs(gated_delta)) - penalty_weight * np.sum(boundary_penalty))

    # Phase5b: separate global/local caution penalties (fallback to legacy `caution_operator_penalty`).
    caution_pen_legacy = float(_config_get(config, "caution_operator_penalty", 0.0))
    caution_global_operator_penalty = float(_config_get(config, "caution_global_operator_penalty", caution_pen_legacy))
    caution_local_operator_penalty = float(_config_get(config, "caution_local_operator_penalty", caution_pen_legacy))

    # Phase7: reduce caution penalty strength when recovery is high (config-gated).
    enable_recovery = bool(_config_get(config, "enable_recovery", False))
    recovery_rate = float(_config_get(config, "recovery_rate", 0.0))
    rec = float(getattr(state, "recovery_scalar", 0.0))
    penalty_scale = 1.0
    if enable_recovery and float(recovery_rate) != 0.0:
        penalty_scale = float(np.clip(1.0 - float(recovery_rate) * float(np.clip(rec, 0.0, 1.0)), 0.0, 1.0))
    caution_global_operator_penalty *= penalty_scale
    caution_local_operator_penalty *= penalty_scale

    caution_energy = 0.0
    caution_energy_local = 0.0
    global_score_post = float(global_score_pre)
    component_score_pre = float(operator_component_score)
    component_score_post = float(component_score_pre)

    if caution_read is not None and (caution_global_operator_penalty != 0.0 or caution_local_operator_penalty != 0.0):
        c = np.clip(np.asarray(caution_read, dtype=float), 0.0, 1.0)
        caution_energy = float(np.sum(np.abs(gated_delta)
 * c))
        if caution_global_operator_penalty != 0.0:
            global_score_post = float(global_score_post - caution_global_operator_penalty * caution_energy)
        if enable_components and active_mask_native is not None and caution_local_operator_penalty != 0.0:
            mask_native = np.asarray(active_mask_native, dtype=float)
            if mask_native.shape == signed_read.shape:
                mask_read = op.read(mask_native)
                sel = mask_read > 0.5
                caution_energy_local = float(np.sum(np.abs(gated_delta) * c * sel.astype(float)))
                component_score_post = float(component_score_post - caution_local_operator_penalty * caution_energy_local)

    if enable_components and active_mask_native is not None:
        score_pre = float(component_global_blend * global_score_pre + local_blend_eff * component_score_pre)
        score_post = float(component_global_blend * global_score_post + local_blend_eff * component_score_post)
        score = float(score_post)
    else:
        score_pre = float(global_score_pre)
        score_post = float(global_score_post)
        score = float(score_post)

    details: Dict[str, Any] = {
        "operator": op.name,
        "peaks": [asdict(p) for p in peaks],
        "bands": [asdict(b) for b in bands],
        "corridor": {
            "window": corridor.window,
            "entry_resistance": corridor.entry_resistance,
            "exit_resistance": corridor.exit_resistance,
            "threshold": corridor.threshold,
        },
        "gated_delta": gated_delta,
        "boundary_penalty": boundary_penalty,
        "corridor_hits": hits,
        "corridor_blocks": blocks,
        "corridor_block_count": int(len(blocks)),
        "gated_update_energy": float(np.sum(np.abs(gated_delta))),
        "boundary_penalty_energy": float(np.sum(boundary_penalty)),
        "caution_energy": float(caution_energy),
        "caution_energy_local": float(caution_energy_local),
        "score_pre_caution": float(score_pre),
        "score_post_caution": float(score_post),
        "polarity_read": pol_read,
        "zero_crossings_read": zc_read,
        "topology_support_energy": float(np.sum(np.abs(corridor.topology_support))) if corridor.topology_support is not None else 0.0,
        "cancellation_penalty_energy": float(np.sum(np.abs(corridor.cancellation_penalty))) if corridor.cancellation_penalty is not None else 0.0,
        "signed_run_count": int(corridor.signed_run_count),
        "largest_signed_run_width": int(corridor.largest_signed_run_width),
        "corridor_topology_bias_applied": bool(corridor.topology_bias_applied),
        "operator_component_score": float(operator_component_score),
    }
    return score, details


def _select_operator(
    scores: Dict[str, float],
    config: Dict[str, Any],
) -> Tuple[str, float]:
    sym = _config_get(config, "symmetry_handling", {})
    enabled = bool(_config_get(sym, "enabled", True))
    if not enabled:
        ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        best = ordered[0]
        shift = float(best[1] - (ordered[1][1] if len(ordered) > 1 else 0.0))
        return best[0], shift

    family_mode = str(_config_get(sym, "family_mode", "paired")).strip().lower()
    if family_mode not in ("paired", "equivalence_class"):
        ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        best = ordered[0]
        shift = float(best[1] - (ordered[1][1] if len(ordered) > 1 else 0.0))
        return best[0], shift

    pairs = _config_get(sym, "inversion_pairs", [["++", "--"], ["+-", "-+"]])
    magnitude_invariant = bool(_config_get(sym, "magnitude_invariant", True))
    orientation_distinct = bool(_config_get(sym, "orientation_distinct", True))
    group_scores: Dict[str, float] = {}
    for a, b in pairs:
        sa = float(scores.get(a, -1e9))
        sb = float(scores.get(b, -1e9))
        if magnitude_invariant:
            group_scores[f"{a}|{b}"] = max(sa, sb)
        else:
            group_scores[f"{a}|{b}"] = 0.5 * (sa + sb)
    # Choose best group then best operator in that group.
    best_group = max(group_scores.items(), key=lambda kv: kv[1])[0]
    a, b = best_group.split("|", 1)
    if orientation_distinct:
        best_op = a if float(scores.get(a, -1e9)) >= float(scores.get(b, -1e9)) else b
    else:
        best_op = a
    ordered = sorted(group_scores.items(), key=lambda kv: kv[1], reverse=True)
    shift = float(ordered[0][1] - (ordered[1][1] if len(ordered) > 1 else 0.0))
    return best_op, shift


def run_reasoning(state: SignatureState, config: Dict[str, Any]) -> List[Dict[str, Any]]:
    phases = int(_config_get(config, "phases", 4))
    lam = float(_config_get(config, "lambda", 0.2))
    kappa = float(_config_get(config, "kappa", 0.6))
    eta = float(_config_get(config, "eta", 0.1))
    stability_alpha = float(_config_get(config, "stability_alpha", 0.2))
    phase_mix = float(_config_get(config, "phase_mix", 0.25))
    family_diffusion = float(_config_get(config, "family_diffusion", 0.0))
    orientation_diffusion = float(_config_get(config, "orientation_diffusion", 0.0))
    diffusion_boundary_mode = str(_config_get(config, "diffusion_boundary_mode", "wrap"))
    polarity_threshold = float(_config_get(config, "polarity_threshold", 0.01))
    zero_crossing_threshold = float(_config_get(config, "zero_crossing_threshold", 0.02))
    enable_polarized_signature_summary = bool(_config_get(config, "enable_polarized_signature_summary", True))
    wraparound_lattice = bool(_config_get(config, "use_wraparound_lattice", True))
    enable_trace_salience = bool(_config_get(config, "enable_trace_salience", False))
    trace_salience_window = int(_config_get(config, "trace_salience_window", 8))
    caution_similarity_weight = float(_config_get(config, "caution_similarity_weight", 0.6))
    caution_salience_weight = float(_config_get(config, "caution_salience_weight", 0.4))
    caution_threshold = float(_config_get(config, "caution_threshold", 0.7))
    caution_release_rate = float(_config_get(config, "caution_release_rate", 0.0))
    caution_component_score_mode = str(_config_get(config, "caution_component_score_mode", "mean"))
    enable_hold_state = bool(_config_get(config, "enable_hold_state", False))
    hold_semantics = str(_config_get(config, "hold_semantics", "decay")).strip().lower()
    hold_from_bounded_caution_only = bool(_config_get(config, "hold_from_bounded_caution_only", True))
    hold_persist = bool(_config_get(config, "hold_persist", False))

    enable_recovery = bool(_config_get(config, "enable_recovery", False))
    recovery_rate = float(_config_get(config, "recovery_rate", 0.0))
    recontextualization_weight = float(_config_get(config, "recontextualization_weight", 0.0))
    hold_release_threshold = float(_config_get(config, "hold_release_threshold", 0.25))
    hold_release_required_phases = int(_config_get(config, "hold_release_required_phases", 2))
    component_recovery_weight = float(_config_get(config, "component_recovery_weight", 0.0))

    mem_cfg = _config_get(config, "memory_layer", {})
    memory_enabled = bool(_config_get(config, "memory_enabled", _config_get(mem_cfg, "enabled", False)))
    memory_operator_bias_strength = float(
        _config_get(config, "memory_operator_bias_strength", _config_get(mem_cfg, "operator_bias_strength", 0.0))
    )
    memory_operator_bias = _config_get(config, "memory_operator_bias", {})
    memory_caution_baseline_strength = float(
        _config_get(config, "memory_caution_baseline_strength", _config_get(mem_cfg, "caution_baseline_strength", 0.0))
    )
    memory_caution_baseline_shift = float(_config_get(config, "memory_caution_baseline_shift", 0.0))

    sym_cfg = _config_get(config, "symmetry_handling", {})
    sym_enabled = bool(_config_get(sym_cfg, "enabled", True))
    sym_family_mode = str(_config_get(sym_cfg, "family_mode", "paired")).strip().lower()
    sym_magnitude_invariant = bool(_config_get(sym_cfg, "magnitude_invariant", True))
    sym_orientation_distinct = bool(_config_get(sym_cfg, "orientation_distinct", True))
    if not sym_enabled:
        symmetry_mode_used = "disabled"
    elif sym_family_mode in ("paired", "equivalence_class"):
        symmetry_mode_used = f"paired(magnitude_invariant={sym_magnitude_invariant},orientation_distinct={sym_orientation_distinct})"
    else:
        symmetry_mode_used = f"raw(family_mode={sym_family_mode})"

    # Patch3: ensure derived fields are initialized from persistent signed_field.
    state.derive_amplitude_from_signed(0.0, 1.0)
    state.recompute_polarized_summary(
        signed_source=state.signed_field,
        polarity_threshold=polarity_threshold,
        zero_crossing_threshold=zero_crossing_threshold,
        wraparound_lattice=wraparound_lattice,
        enable_summary=enable_polarized_signature_summary,
    )

    trace: List[Dict[str, Any]] = []
    for phase in range(phases):
        enable_components = bool(_config_get(config, "enable_component_promotion", False))
        component_target_mode = str(_config_get(config, "component_target_mode", "dominant"))
        component_min_support_mass = float(_config_get(config, "component_min_support_mass", 0.05))
        component_min_width = int(_config_get(config, "component_min_width", 1))
        overlap_w = float(_config_get(config, "component_match_overlap_weight", 0.5))
        center_w = float(_config_get(config, "component_match_center_weight", 0.3))
        sign_w = float(_config_get(config, "component_match_sign_weight", 0.2))

        active_mask_native = None
        active = None
        component_meta = {"component_identity_persistence": 0.0, "component_split_count": 0, "component_merge_count": 0}
        if enable_components:
            current_components = extract_signed_components(
                signed_field=state.signed_field,

                amplitude=state.amplitude,
                polarity=state.polarity,
                mode_index=state.mode_index,
                zero_crossings=state.zero_crossings,
                wraparound_lattice=wraparound_lattice,
                component_min_support_mass=component_min_support_mass,
                component_min_width=component_min_width,
            )
            matched, component_meta = match_components_across_phases(
                previous_components=state.components,
                current_components=current_components,
                size=state.size,
                overlap_weight=overlap_w,
                center_weight=center_w,
                sign_weight=sign_w,
            )
            state.components = matched
            state.component_history.append(summarize_component_set(state.components))

            # Choose active component.
            if state.components:
                if component_target_mode == "widest":
                    active = max(state.components, key=lambda c: int(c.get("width", 0)))
                elif component_target_mode == "highest_support":
                    active = max(state.components, key=lambda c: float(c.get("support_mass", 0.0)))
                elif component_target_mode == "global_only":
                    active = None
                else:  # dominant
                    active = max(state.components, key=lambda c: float(c.get("support_mass", 0.0)))

            if active is not None:
                state.active_component_id = int(active.get("stable_id", active.get("component_id", 0)))
                masks = build_component_masks(state.components, size=state.size)
                active_mask_native = masks.get(int(state.active_component_id), None)
            else:
                state.active_component_id = None

        salience_max = 0.0
        similarity_max = 0.0
        caution_peak = 0.0
        recovery_scalar = 0.0
        recontextualization_score = 0.0
        caution_after_recovery = 0.0
        prev_hold_state = bool(getattr(state, "hold_state", False))
        hold_release_reason = "none"
        hold_released = False
        if enable_trace_salience:
            state.trace_salience = compute_trace_salience(trace, window=trace_salience_window)

            masks_all = build_component_masks(state.components, size=state.size) if (enable_components and state.components) else {}
            largest_width = max((int(c.get("width", 0)) for c in state.components), default=0)
            current_structure = {
                "active_component_id": state.active_component_id,
                "active_component_sign": (int(active.get("sign", 0)) if (enable_components and active is not None) else 0),
                "active_component_width": (int(active.get("width", 0)) if (enable_components and active is not None) else 0),
                "signed_run_count": int(len(state.components)) if (enable_components and state.components) else 0,
                "largest_signed_run_width": int(largest_width),
            }
            # Pass 1: compute raw/bounded caution + similarity stats (no recovery adjustment yet).
            (
                raw_caution_scalar,
                _raw_caution_field,
                _bounded_caution_scalar,
                bounded_caution_field,
                caution_release_scalar,
                _caution_after_recovery_scalar0,
                _caution_after_recovery_field0,
                salience_max,
                similarity_max,
            ) = compute_caution_field(
                size=state.size,
                components=state.components,
                component_masks=masks_all,
                current_structure=current_structure,
                trace_history=trace,
                salience=state.trace_salience,
                salience_weight=caution_salience_weight,
                similarity_weight=caution_similarity_weight,
                caution_release_rate=caution_release_rate,
                recovery_scalar=0.0,
                recontextualization_score=0.0,
                recovery_rate=0.0,
                recontextualization_weight=0.0,
            )

            last_t = trace[-1] if trace else {}
            recovery_scalar = 0.0
            recontextualization_score = 0.0
            if enable_recovery and trace:
                recovery_scalar = compute_recovery_signal(
                    similarity_max=float(similarity_max),
                    salience_max=float(salience_max),
                    corridor_block_count=float(last_t.get("corridor_block_count", len(last_t.get("corridor_blocks", []) or []))),
                    boundary_penalty_energy=float(last_t.get("boundary_penalty_energy", 0.0)),
                    spread_measure=float(last_t.get("spread_measure", 0.0)),
                )
                recontextualization_score = compute_recontextualization_score(
                    similarity_max=float(similarity_max),
                    salience_max=float(salience_max),
                )

            # Pass 2: apply recovery/recontextualization (bounded only) and compute post-adjustment caution field.
            (
                _raw_caution_scalar2,
                _raw_caution_field2,
                _bounded_caution_scalar2,
                _bounded_caution_field2,
                _caution_release_scalar2,
                caution_after_recovery_scalar,
                caution_after_recovery_field,
                _salience_max2,
                _similarity_max2,
            ) = compute_caution_field(
                size=state.size,
                components=state.components,
                component_masks=masks_all,
                current_structure=current_structure,
                trace_history=trace,
                salience=state.trace_salience,
                salience_weight=caution_salience_weight,
                similarity_weight=caution_similarity_weight,
                caution_release_rate=caution_release_rate,
                recovery_scalar=float(recovery_scalar),
                recontextualization_score=float(recontextualization_score),
                recovery_rate=float(recovery_rate if enable_recovery else 0.0),
                recontextualization_weight=float(recontextualization_weight if enable_recovery else 0.0),
            )

            state.raw_caution_scalar = float(raw_caution_scalar)
            state.caution_release_scalar = float(caution_release_scalar)
            state.recovery_scalar = float(recovery_scalar)
            state.recontextualization_score = float(recontextualization_score)
            state.caution_field = np.asarray(caution_after_recovery_field, dtype=float)
            state.caution_scalar = float(caution_after_recovery_scalar)
            caution_after_recovery = float(caution_after_recovery_scalar)
            caution_peak = float(np.max(state.caution_field)) if state.caution_field.size else 0.0

            # Memory layer: apply a mild additive baseline shift to the bounded caution field/scalar.
            # This is continuation bias, not trace replay.
            if memory_enabled and float(memory_caution_baseline_strength) != 0.0:
                adj = float(np.clip(float(memory_caution_baseline_shift), -0.25, 0.25)) * float(memory_caution_baseline_strength)
                if adj != 0.0:
                    state.caution_field = np.clip(state.caution_field + adj, 0.0, 1.0)
                    state.caution_scalar = float(np.mean(state.caution_field)) if state.caution_field.size else float(np.clip(state.caution_scalar + adj, 0.0, 1.0))
                    caution_after_recovery = float(state.caution_scalar)
                    caution_peak = float(np.max(state.caution_field)) if state.caution_field.size else 0.0

            # Phase7: component-local recovery scores and a recovery field for corridor modulation.
            state.recovery_field = np.zeros_like(state.caution_field, dtype=float)
            recovery_scores = {}
            if enable_recovery and enable_components and state.components:
                recovery_scores = component_recovery_scores(components=state.components, caution_field=state.caution_field, size=state.size, mode="mean")
                rf = np.zeros(state.size, dtype=float)
                for comp in state.components:
                    sid = int(comp.get("stable_id", comp.get("component_id", 0)))
                    m = masks_all.get(sid, None)
                    if m is None:
                        continue
                    rf += float(np.clip(recovery_scores.get(sid, 0.0), 0.0, 1.0)) * np.asarray(m, dtype=float)
                mx = float(np.max(rf)) if rf.size else 0.0
                if mx > 0:
                    rf = np.clip(rf / mx, 0.0, 1.0)
                state.recovery_field = np.clip(float(recovery_scalar) * rf, 0.0, 1.0)

            # Make component targeting caution-aware (if components are enabled and a local target is desired).
            if enable_components and state.components and component_target_mode != "global_only":
                c_scores = component_caution_scores(
                    components=state.components,
                    caution_field=state.caution_field,
                    size=state.size,
                    mode=caution_component_score_mode,
                    recovery_scores=(recovery_scores if enable_recovery else None),
                    recovery_weight=(component_recovery_weight if enable_recovery else 0.0),
                )
                def _adj_score(c):
                    sid = int(c.get("stable_id", c.get("component_id", 0)))
                    score = float(c_scores.get(sid, {}).get("net" if enable_recovery else "score", 0.0))
                    return float(c.get("support_mass", 0.0)) * (1.0 - score)
                active = max(state.components, key=_adj_score)
                state.active_component_id = int(active.get("stable_id", active.get("component_id", 0)))
                active_mask_native = masks_all.get(int(state.active_component_id), None)
        else:
            state.trace_salience = []
            state.raw_caution_scalar = 0.0
            state.caution_scalar = 0.0
            state.caution_release_scalar = 0.0
            state.caution_field = np.zeros_like(state.signed_field, dtype=float)
            state.hold_state = False
            state.recovery_scalar = 0.0
            state.recontextualization_score = 0.0
            state.hold_release_counter = 0
            state.recovery_field = np.zeros_like(state.signed_field, dtype=float)

        dominant_pol_before = dominant_polarity(state.polarity)
        zero_crossings_before = list(state.zero_crossings) if state.zero_crossings else []
        polarity_before = state.polarity.copy()

        op_scores_pre: Dict[str, float] = {}
        op_scores_post: Dict[str, float] = {}
        op_details: Dict[str, Dict[str, Any]] = {}
        recovery_field_native = state.recovery_field if (enable_trace_salience and enable_recovery) else None
        for op in OPERATORS:
            score, details = _score_operator(
                state,
                config,
                op,
                active_mask_native,
                state.caution_field if enable_trace_salience else None,
                recovery_field_native,
            )
            op_scores_post[op.name] = float(score)
            op_scores_pre[op.name] = float(details.get("score_pre_caution", score))
            op_details[op.name] = details

        # Memory layer: add a small operator prior bias (continuation deformation, not trace replay).
        if memory_enabled and float(memory_operator_bias_strength) != 0.0 and isinstance(memory_operator_bias, dict):
            for k in list(op_scores_post.keys()):
                op_scores_post[k] = float(op_scores_post.get(k, 0.0)) + float(memory_operator_bias_strength) * float(memory_operator_bias.get(k, 0.0))

        # Phase6: optional score-space orientation diffusion across the 4 operator scores.
        operator_order = ["++", "--", "+-", "-+"]
        raw_operator_scores = {k: float(op_scores_post.get(k, 0.0)) for k in operator_order}
        diffused_operator_scores = dict(raw_operator_scores)
        orientation_diffusion_applied = bool(float(orientation_diffusion) > 0.0)
        if orientation_diffusion_applied:
            v = np.array([raw_operator_scores[k] for k in operator_order], dtype=float)
            v2 = diffuse_operator_scores(v, coeff=float(orientation_diffusion))
            diffused_operator_scores = {k: float(v2[i]) for i, k in enumerate(operator_order)}

        selected_op, decision_shift = _select_operator(diffused_operator_scores if orientation_diffusion_applied else op_scores_post, config)
        chosen = next(op for op in OPERATORS if op.name == selected_op)
        details = op_details[selected_op]

        # Writeback is the inverse transform of the chosen operator.
        gated_delta_read = details["gated_delta"]
        penalty_read = details["boundary_penalty"]
        gated_delta_write = chosen.write(gated_delta_read)
        penalty_write = chosen.write(penalty_read)

        top_before = state.top_peaks(k=int(_config_get(config, "trace_topk_peaks", 5)))
        hold_triggered = False
        hold_reason = "none"
        kappa_eff = float(kappa)
        eta_eff = float(eta)
        gated_eff = np.asarray(gated_delta_write, dtype=float)
        penalty_eff = np.asarray(penalty_write, dtype=float)
        hold_metric = float(state.caution_scalar) if hold_from_bounded_caution_only else float(np.clip(state.raw_caution_scalar, 0.0, 1.0))
        threshold_crossed = enable_trace_salience and (hold_metric > float(caution_threshold))
        carry_hold = enable_trace_salience and hold_persist and bool(getattr(state, "hold_state", False)) and (hold_metric > 0.5 * float(caution_threshold))

        if enable_trace_salience and enable_recovery and prev_hold_state:
            if float(recovery_scalar) >= float(hold_release_threshold):
                state.hold_release_counter = int(getattr(state, "hold_release_counter", 0)) + 1
            else:
                state.hold_release_counter = 0
        else:
            state.hold_release_counter = 0

        release_ready = bool(prev_hold_state and enable_recovery and int(getattr(state, "hold_release_counter", 0)) >= int(hold_release_required_phases))
        if release_ready:
            carry_hold = False

        if enable_trace_salience and (threshold_crossed or carry_hold):
            kappa_eff = float(kappa) * 0.2
            if enable_hold_state:
                hold_triggered = True
                hold_reason = "carry" if (carry_hold and not threshold_crossed) else "threshold"
                kappa_eff = 0.0
                eta_eff = 0.0
                gated_eff = np.zeros_like(gated_eff)
                penalty_eff = np.zeros_like(penalty_eff)

        if prev_hold_state and not hold_triggered:
            if release_ready:
                hold_released = True
                hold_release_reason = "recovery_release"
            elif not threshold_crossed:
                hold_released = True
                hold_release_reason = "threshold_drop"

        state.hold_state = bool(hold_triggered)
        freeze_hold = bool(hold_triggered and hold_semantics == "freeze")

        diffusion_energy = 0.0
        spread_measure = 0.0
        orientation_diffusion_energy = 0.0
        orientation_spread_measure = 0.0

        if not freeze_hold:
            state.apply_update(gated_eff, lam=lam, kappa=kappa_eff, eta=eta_eff, boundary_penalty=penalty_eff)

            # Patch3: diffuse the persistent signed field, then re-derive amplitude.
            signed_before_diffusion = np.asarray(state.signed_field, dtype=float)
            diffused_signed_f, diff_term_f, lap_f = apply_family_diffusion(
                signed_before_diffusion, coeff=family_diffusion, mode=diffusion_boundary_mode
            )
            diffused_signed = diffused_signed_f
            diffusion_energy = float(np.sum(np.abs(diff_term_f)))
            spread_measure = float(np.sum(np.abs(lap_f)))

            state.signed_field = np.clip(diffused_signed, -1.0, 1.0)
            state.derive_amplitude_from_signed(0.0, 1.0)
        else:
            # Hold semantics "freeze": keep the signature surface unchanged for this phase.
            state.derive_amplitude_from_signed(0.0, 1.0)

        # Patch3: polarity + zero-crossing overlay derived from persistent signed_field.
        signed_source = state.signed_field
        state.recompute_polarized_summary(
            signed_source=signed_source,
            polarity_threshold=polarity_threshold,
            zero_crossing_threshold=zero_crossing_threshold,
            wraparound_lattice=wraparound_lattice,
            enable_summary=enable_polarized_signature_summary,
        )
        dominant_pol_after = dominant_polarity(state.polarity)
        zero_crossings_after = list(state.zero_crossings) if state.zero_crossings else []
        polarity_shift = polarity_shift_metric(polarity_before, state.polarity)

        if not freeze_hold:
            state.apply_phase_shift(chosen.phase_shift, mix=phase_mix)
            state.update_stability(gated_eff, alpha=stability_alpha)

        # Refresh peaks/bands in native (writeback) space for trace readability.
        peak_cfg = _config_get(config, "peak_detection", {})
        peaks_native, _bands_native = detect_peaks_and_bands(
            state.amplitude,
            min_height=float(_config_get(peak_cfg, "min_height", _config_get(config, "peak_threshold", 0.10))),
            min_distance=int(_config_get(peak_cfg, "min_distance", 1)),
            merge_radius=int(_config_get(peak_cfg, "merge_radius", 1)),
            band_rel_threshold=float(_config_get(config, "band_rel_threshold", 0.50)),
            wraparound_lattice=wraparound_lattice,
        )
        state.peaks = [asdict(p) for p in peaks_native]
        state.bands = [asdict(b) for b in _bands_native]
        band_widths_after = [int(b.width) for b in _bands_native]

        trace.append(
            {
                "phase": int(phase),
                "selected_operator": selected_op,
                "top_peaks_before": top_before,
                "gated_update_energy": float(details["gated_update_energy"]),
                "top_peaks_after": state.top_peaks(k=int(_config_get(config, "trace_topk_peaks", 5))),
                "corridor_hits": details["corridor_hits"],
                "corridor_blocks": details["corridor_blocks"],
                "decision_shift": float(decision_shift),
                # Shell-friendly aliases (keep the canonical fields above for backward compatibility).
                "shift": float(decision_shift),
                "operator_scores": {k: float(v) for k, v in op_scores_post.items()},
                "operator_scores_pre_caution": {k: float(v) for k, v in op_scores_pre.items()},
                "operator_scores_post_caution": {k: float(v) for k, v in op_scores_post.items()},
                "raw_operator_scores": {k: float(v) for k, v in raw_operator_scores.items()},
                "diffused_operator_scores": {k: float(v) for k, v in diffused_operator_scores.items()},
                "orientation_diffusion_applied": bool(orientation_diffusion_applied),
                "diffusion_energy": float(diffusion_energy),
                "spread_measure": float(spread_measure),
                "orientation_diffusion_energy": float(orientation_diffusion_energy),
                "orientation_spread_measure": float(orientation_spread_measure),
                "band_widths_after": band_widths_after,
                "dominant_polarity_before": int(dominant_pol_before),
                "dominant_polarity_after":
 int(dominant_pol_after),
                "zero_crossings_before": zero_crossings_before,
                "zero_crossings_after": zero_crossings_after,
                "polarity_shift": float(polarity_shift),
                "dominant_component_summary_after": state.component_summary if enable_polarized_signature_summary else {},
                "topology_support_energy": float(details.get("topology_support_energy", 0.0)),
                "cancellation_penalty_energy": float(details.get("cancellation_penalty_energy", 0.0)),
                "signed_run_count": int(details.get("signed_run_count", 0)),
                "largest_signed_run_width": int(details.get("largest_signed_run_width", 0)),
                "corridor_topology_bias_applied": bool(details.get("corridor_topology_bias_applied", False)),
                "component_count": int(len(state.components)) if enable_components else 0,
                "active_component_id": state.active_component_id if enable_components else None,
                "active_component_sign": (int(active.get("sign", 0)) if (enable_components and active is not None) else 0),
                "active_component_width": (int(active.get("width", 0)) if (enable_components and active is not None) else 0),
                "active_component_support_mass": (float(active.get("support_mass", 0.0)) if (enable_components and active is not None) else 0.0),
                "operator_component_score": float(details.get("operator_component_score", 0.0)),
                "component_identity_persistence": float(component_meta.get("component_identity_persistence", 0.0)),
                "component_split_count": int(component_meta.get("component_split_count", 0)),
                "component_merge_count": int(component_meta.get("component_merge_count", 0)),
                "salience_max": float(salience_max),
                "similarity_max": float(similarity_max),
                "raw_caution_scalar": float(getattr(state, "raw_caution_scalar", 0.0)) if enable_trace_salience else 0.0,
                "caution_scalar": float(state.caution_scalar) if enable_trace_salience else 0.0,
                "caution_release_scalar": float(getattr(state, "caution_release_scalar", 0.0)) if enable_trace_salience else 0.0,
                "caution_after_recovery": float(caution_after_recovery) if enable_trace_salience else 0.0,
                "caution_peak": float(caution_peak) if enable_trace_salience else 0.0,
                "recovery_scalar": float(getattr(state, "recovery_scalar", 0.0)) if enable_trace_salience else 0.0,
                # Shell-friendly alias for trace summaries.
                "recovery": float(getattr(state, "recovery_scalar", 0.0)) if enable_trace_salience else 0.0,
                "recontextualization_score": float(getattr(state, "recontextualization_score", 0.0)) if enable_trace_salience else 0.0,
                "hold_triggered": bool(hold_triggered),
                "hold_state": bool(getattr(state, "hold_state", False)),
                "hold_reason": str(hold_reason),
                "hold_release_counter": int(getattr(state, "hold_release_counter", 0)) if enable_trace_salience else 0,
                "hold_released": bool(hold_released),
                "hold_release_reason": str(hold_release_reason),
                "hold_semantics": str(hold_semantics),
                "hold_frozen": bool(freeze_hold),
                "symmetry_mode_used": str(symmetry_mode_used),
                "boundary_penalty_energy": float(details.get("boundary_penalty_energy", 0.0)),
                "corridor_block_count": int(details.get("corridor_block_count", 0)),
            }
        )

        state.step += 1

    return trace
