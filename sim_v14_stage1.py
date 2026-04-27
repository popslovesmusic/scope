
import json
import hashlib
import numpy as np
from core.signature_state import SignatureState
from core.reasoning_loop import run_reasoning
from core.readout_heads import binary_readout
from core.peak_detector import Peak, Band, detect_peaks_and_bands
from core.corridor_gate import build_dynamic_corridor
from core.io_state import save_state
import argparse


def encode_input_to_signed_field(text: str, size: int, seed=None) -> np.ndarray:
    text = "" if text is None else str(text)
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    base_seed = int(h[:16], 16) ^ int(h[16:32], 16)
    if seed is not None:
        base_seed ^= int(seed) & 0xFFFFFFFF
    rng = np.random.default_rng(base_seed)
    # Signed primary surface in [-1, 1].
    signed = (rng.random(int(size)) * 2.0) - 1.0
    if text:
        for i, ch in enumerate(text):
            idx = (ord(ch) + i) % int(size)
            bump = 0.5 / max(1, len(text))
            signed[idx] += bump if ((ord(ch) + i) % 2 == 0) else -bump
    signed = np.clip(signed, -1.0, 1.0)
    return signed.astype(float)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", default="binary_demo")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", default="v14_output.json")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    signature_size = int(config.get("signature_size", 12))
    state = SignatureState(signature_size)
    state.input_trace.append({"input": args.input, "seed": args.seed})

    if "init_signed_field" in config:
        init = np.asarray(config["init_signed_field"], dtype=float)
        if init.shape != (signature_size,):
            raise ValueError("config.init_signed_field must match signature_size")
        state.signed_field = np.clip(init, -1.0, 1.0)
        state.derive_amplitude_from_signed(0.0, 1.0)
    elif "init_amplitude" in config:
        init = np.asarray(config["init_amplitude"], dtype=float)
        if init.shape != (signature_size,):
            raise ValueError("config.init_amplitude must match signature_size")
        # Compatibility: treat amplitude init as positive signed field.
        state.signed_field = np.clip(init, 0.0, 1.0)
        state.derive_amplitude_from_signed(0.0, 1.0)
    else:
        state.signed_field = encode_input_to_signed_field(args.input, signature_size, seed=args.seed)
        state.derive_amplitude_from_signed(0.0, 1.0)

    trace = run_reasoning(state, config)

    # Build a final corridor snapshot from current peaks/bands.
    peak_cfg = config.get("peak_detection", {})
    peaks, bands = detect_peaks_and_bands(
        state.amplitude,
        min_height=float(peak_cfg.get("min_height", config.get("peak_threshold", 0.10))),
        min_distance=int(peak_cfg.get("min_distance", 1)),
        merge_radius=int(peak_cfg.get("merge_radius", 1)),
        band_rel_threshold=float(config.get("band_rel_threshold", 0.50)),
        wraparound_lattice=bool(config.get("use_wraparound_lattice", True)),
    )
    corridor_cfg = config.get("corridor", {})
    corridor = build_dynamic_corridor(
        size=signature_size,
        peaks=peaks,
        bands=bands,
        stability=state.stability,
        amplitude=state.amplitude,
        polarity=state.polarity,
        zero_crossings=state.zero_crossings,
        components=state.components,
        caution_field=getattr(state, "caution_field", None),
        recovery_field=getattr(state, "recovery_field", None),
        wraparound_lattice=bool(config.get("use_wraparound_lattice", True)),
        base_width=float(corridor_cfg.get("base_width", 1.0)),
        width_gain=float(corridor_cfg.get("width_gain", 2.0)),
        stability_gain=float(corridor_cfg.get("stability_gain", 1.0)),
        floor=float(corridor_cfg.get("floor", 0.05)),
        threshold=float(corridor_cfg.get("threshold", 0.10)),
        exit_scale=float(corridor_cfg.get("exit_scale", 0.50)),
        corridor_polarity_consistency_bonus=float(config.get("corridor_polarity_consistency_bonus", 0.0)),
        corridor_zero_crossing_bonus=float(config.get("corridor_zero_crossing_bonus", 0.0)),
        corridor_topology_support_gain=float(config.get("corridor_topology_support_gain", 0.0)),
        corridor_zero_crossing_penalty=float(config.get("corridor_zero_crossing_penalty", 0.0)),
        corridor_run_support_gain=float(config.get("corridor_run_support_gain", 0.0)),
        corridor_crossing_decay=float(config.get("corridor_crossing_decay", 1.5)),
        corridor_require_signed_support_for_entry=bool(config.get("corridor_require_signed_support_for_entry", False)),
        caution_corridor_penalty=float(config.get("caution_corridor_penalty", 0.0)),
        caution_entry_penalty=config.get("caution_entry_penalty", None),
        caution_exit_penalty=config.get("caution_exit_penalty", None),
        corridor_recovery_gain=float(config.get("corridor_recovery_gain", 0.0)),
        max_recovery_fraction_of_base_window=float(config.get("max_recovery_fraction_of_base_window", 0.5)),
    )

    output_model = config.get("output_model", None)
    if output_model is None:
        # v14.json default: families 0-2 vs 9-11.
        output_model = {
            "classifier": {
                "mode": "binary",
                "decision_rule": "max_corridor_supported_energy",
                "class_heads": [
                    {"name": "class_0", "readout_family": [0, 1, 2]},
                    {"name": "class_1", "readout_family": [9, 10, 11]},
                ],
            }
        }

    class_heads = output_model["classifier"]["class_heads"]
    class_scores, selected_class, confidence = binary_readout(
        amplitude=state.amplitude,
        corridor_window=corridor.window,
        class_heads=class_heads,
    )

    last_trace = trace[-1] if trace else {}
    operator_family = ["++", "--", "+-", "-+"]
    raw_scores_dict = last_trace.get("raw_operator_scores", last_trace.get("operator_scores", {})) or {}
    diff_scores_dict = last_trace.get("diffused_operator_scores", raw_scores_dict) or {}
    operator_scores_raw = [float(raw_scores_dict.get(name, 0.0)) for name in operator_family]
    operator_scores_diffused = [float(diff_scores_dict.get(name, 0.0)) for name in operator_family]
    operator_scores_used = operator_scores_diffused if bool(last_trace.get("orientation_diffusion_applied", False)) else operator_scores_raw

    state_dict = {
        "version": "v14_stage1_state_v1",
        "step": int(state.step),
        "input_trace": state.input_trace,
        "signature": {
            "signed_field": state.signed_field.tolist(),
            "amplitude": state.amplitude.tolist(),
            "phase": state.phase.tolist(),
            "stability": state.stability.tolist(),
            "peaks": state.peaks,
            "bands": state.bands,
            "polarity": state.polarity.tolist(),
            "zero_crossings": state.zero_crossings,
            "mode_index": state.mode_index.tolist(),
            "component_summary": state.component_summary,
            "components": state.components,
            "active_component_id": state.active_component_id,
            "raw_caution_scalar": float(getattr(state, "raw_caution_scalar", 0.0)),
            "caution_scalar": float(getattr(state, "caution_scalar", 0.0)),
            "caution_release_scalar": float(getattr(state, "caution_release_scalar", 0.0)),
            "caution_field": (getattr(state, "caution_field", np.zeros(signature_size)).tolist()),
            "hold_state": bool(getattr(state, "hold_state", False)),
            "recovery_scalar": float(getattr(state, "recovery_scalar", 0.0)),
            "recontextualization_score": float(getattr(state, "recontextualization_score", 0.0)),
            "hold_release_counter": int(getattr(state, "hold_release_counter", 0)),
        },
        "corridor": {
            "window": corridor.window.tolist(),
            "entry_resistance": corridor.entry_resistance.tolist(),
            "exit_resistance": corridor.exit_resistance.tolist(),
            "threshold": float(corridor.threshold),
            "topology_support_energy": float(np.sum(np.abs(corridor.topology_support))) if corridor.topology_support is not None else 0.0,
            "cancellation_penalty_energy": float(np.sum(np.abs(corridor.cancellation_penalty))) if corridor.cancellation_penalty is not None else 0.0,
            "signed_run_count": int(getattr(corridor, "signed_run_count", 0)),
            "largest_signed_run_width": int(getattr(corridor, "largest_signed_run_width", 0)),
            "corridor_topology_bias_applied": bool(getattr(corridor, "topology_bias_applied", False)),
            "wraparound_lattice": bool(getattr(corridor, "wraparound_lattice", True)),
            "caution_window_delta_mean": float(getattr(corridor, "caution_window_delta_mean", 0.0)),
            "caution_entry_delta_mean": float(getattr(corridor, "caution_entry_delta_mean", 0.0)),
            "caution_exit_delta_mean": float(getattr(corridor, "caution_exit_delta_mean", 0.0)),
            "recovery_support_energy": float(getattr(corridor, "recovery_support_energy", 0.0)),
            "net_caution_after_recovery": float(getattr(corridor, "net_caution_after_recovery", 0.0)),
        },
        "orientation": {
            "active_operator": last_trace.get("selected_operator", None),
            "operator_family": operator_family,
            "operator_scores": operator_scores_used,
            "raw_operator_scores": raw_scores_dict,
            "diffused_operator_scores": diff_scores_dict,
            "orientation_diffusion_applied": bool(last_trace.get("orientation_diffusion_applied", False)),
        },
        "reasoning": {
            "phase_index": int(len(trace)),
            "phase_history": trace,

            "writeback_gain": float(config.get("kappa", 0.6)),
            "relaxation": float(config.get("lambda", 0.2)),
            "component_history": state.component_history,
            "hold_semantics": str(config.get("hold_semantics", "decay")),
            "symmetry_mode_used": str(last_trace.get("symmetry_mode_used", "")),
        },
        "output": {
            "class_scores": class_scores,
            "selected_class": int(selected_class),
            "confidence": float(confidence),
        },
    }

    print("Result:", state_dict["output"])
    save_state(args.out, state_dict, trace)

if __name__ == "__main__":
    main()
