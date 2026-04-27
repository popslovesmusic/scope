import copy
import json
import os
import sys
from dataclasses import dataclass

import numpy as np

# Ensure repo root is on sys.path when executed as a script.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from core.corridor_gate import build_dynamic_corridor
from core.peak_detector import detect_peaks_and_bands
from core.readout_heads import binary_readout
from core.reasoning_loop import run_reasoning
from core.signature_state import SignatureState
from sim_v14_stage1 import encode_input_to_signed_field


@dataclass(frozen=True)
class CaseResult:
    name: str
    ok: bool
    details: dict


def _load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _run_state(config: dict, *, signed_field: np.ndarray) -> dict:
    cfg = copy.deepcopy(config)
    size = int(cfg.get("signature_size", 12))

    state = SignatureState(size)
    state.signed_field = np.asarray(signed_field, dtype=float).copy()
    state.derive_amplitude_from_signed(0.0, 1.0)

    trace = run_reasoning(state, cfg)

    peak_cfg = cfg.get("peak_detection", {})
    peaks, bands = detect_peaks_and_bands(
        state.amplitude,
        min_height=float(peak_cfg.get("min_height", cfg.get("peak_threshold", 0.10))),
        min_distance=int(peak_cfg.get("min_distance", 1)),
        merge_radius=int(peak_cfg.get("merge_radius", 1)),
        band_rel_threshold=float(cfg.get("band_rel_threshold", 0.50)),
        wraparound_lattice=bool(cfg.get("use_wraparound_lattice", True)),
    )

    corridor_cfg = cfg.get("corridor", {})
    corridor = build_dynamic_corridor(
        size=size,
        peaks=peaks,
        bands=bands,
        stability=state.stability,
        amplitude=state.amplitude,
        polarity=state.polarity,
        zero_crossings=state.zero_crossings,
        components=state.components,
        base_width=float(corridor_cfg.get("base_width", 1.0)),
        width_gain=float(corridor_cfg.get("width_gain", 2.0)),
        stability_gain=float(corridor_cfg.get("stability_gain", 1.0)),
        floor=float(corridor_cfg.get("floor", 0.05)),
        threshold=float(corridor_cfg.get("threshold", 0.10)),
        exit_scale=float(corridor_cfg.get("exit_scale", 0.50)),
        corridor_polarity_consistency_bonus=float(cfg.get("corridor_polarity_consistency_bonus", 0.0)),
        corridor_zero_crossing_bonus=float(cfg.get("corridor_zero_crossing_bonus", 0.0)),
        corridor_topology_support_gain=float(cfg.get("corridor_topology_support_gain", 0.0)),
        corridor_zero_crossing_penalty=float(cfg.get("corridor_zero_crossing_penalty", 0.0)),
        corridor_run_support_gain=float(cfg.get("corridor_run_support_gain", 0.0)),
        corridor_crossing_decay=float(cfg.get("corridor_crossing_decay", 1.5)),
        corridor_require_signed_support_for_entry=bool(cfg.get("corridor_require_signed_support_for_entry", False)),
    )

    output_model = cfg.get("output_model", None)
    class_heads = output_model["classifier"]["class_heads"] if output_model else []
    class_scores, selected_class, confidence = binary_readout(
        amplitude=state.amplitude,
        corridor_window=corridor.window,
        class_heads=class_heads,
    )

    return {
        "state": state,
        "trace": trace,
        "corridor": corridor,
        "output": {
            "class_scores": class_scores,
            "selected_class": int(selected_class),
            "confidence": float(confidence),
        },
    }


def _case_determinism(config: dict) -> CaseResult:
    size = int(config.get("signature_size", 12))
    signed = encode_input_to_signed_field("determinism", size, seed=123)
    r1 = _run_state(dict(config, phases=4), signed_field=signed)
    r2 = _run_state(dict(config, phases=4), signed_field=signed)

    ok = (
        np.allclose(r1["state"].signed_field, r2["state"].signed_field)
        and r1["output"]["selected_class"] == r2["output"]["selected_class"]
        and r1["trace"][-1]["selected_operator"] == r2["trace"][-1]["selected_operator"]
    )
    return CaseResult(
        name="determinism_same_init",
        ok=bool(ok),
        details={
            "selected_class": r1["output"]["selected_class"],
            "selected_operator_last": r1["trace"][-1]["selected_operator"] if r1["trace"] else None,
        },
    )


def _case_binary_head_sanity(config: dict) -> CaseResult:
    # Hold dynamics constant: no phases, no diffusion.
    cfg = dict(config, phases=0, family_diffusion=0.0, corridor_require_signed_support_for_entry=False)
    size = int(cfg.get("signature_size", 12))

    signed0 = np.zeros(size, dtype=float)
    signed0[[0, 1, 2]] = 1.0
    r0 = _run_state(cfg, signed_field=signed0)

    signed1 = np.zeros(size, dtype=float)
    signed1[[9, 10, 11]] = 1.0
    r1 = _run_state(cfg, signed_field=signed1)

    ok = (r0["output"]["selected_class"] == 0) and (r1["output"]["selected_class"] == 1)
    return CaseResult(
        name="binary_heads_sanity",
        ok=bool(ok),
        details={
            "class0_selected": r0["output"]["selected_class"],
            "class1_selected": r1["output"]["selected_class"],
        },
    )


def _case_components_present(config: dict) -> CaseResult:
    cfg = dict(config, enable_component_promotion=True, phases=3)
    size = int(cfg.get("signature_size", 12))
    signed = encode_input_to_signed_field("components", size, seed=7)
    r = _run_state(cfg, signed_field=signed)

    comps = r["state"].components
    ok = (len(comps) >= 1) and (r["trace"][-1].get("component_count", 0) >= 1)
    return CaseResult(
        name="components_extracted",
        ok=bool(ok),
        details={
            "component_count": int(len(comps)),
            "active_component_id": r["state"].active_component_id,
            "identity_persistence_last": float(r["trace"][-1].get("component_identity_persistence", 0.0)) if r["trace"] else None,
        },
    )


def _case_corridor_topology_disable_is_neutral(config: dict) -> CaseResult:
    size = int(config.get("signature_size", 12))
    signed = encode_input_to_signed_field("corridor", size, seed=5)

    cfg_a = dict(
        config,
        phases=0,
        corridor_topology_support_gain=0.0,
        corridor_zero_crossing_penalty=0.0,
        corridor_run_support_gain=0.0,
        corridor_require_signed_support_for_entry=False,
        corridor_polarity_consistency_bonus=0.0,
        corridor_zero_crossing_bonus=0.0,
    )
    cfg_b = dict(
        cfg_a,
        corridor_topology_support_gain=0.0,
        corridor_zero_crossing_penalty=0.0,
        corridor_run_support_gain=0.0,
        corridor_require_signed_support_for_entry=False,
    )
    ra = _run_state(cfg_a, signed_field=signed)
    rb = _run_state(cfg_b, signed_field=signed)
    ok = np.allclose(ra["corridor"].window, rb["corridor"].window)
    return CaseResult(
        name="corridor_topology_gains_zero_neutral",
        ok=bool(ok),
        details={"max_abs_diff": float(np.max(np.abs(ra["corridor"].window - rb["corridor"].window)))},
    )


def main():
    config = _load_config("config/config_v14_scaffold.json")

    cases = [
        _case_determinism(config),
        _case_binary_head_sanity(config),
        _case_components_present(config),
        _case_corridor_topology_disable_is_neutral(config),
    ]

    report = {
        "ok": all(c.ok for c in cases),
        "cases": [{"name": c.name, "ok": c.ok, "details": c.details} for c in cases],
    }

    print("v14 model test:", "PASS" if report["ok"] else "FAIL")
    for c in cases:
        print(f"- {c.name}: {'ok' if c.ok else 'FAIL'} {c.details}")

    with open("v14_model_test_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
