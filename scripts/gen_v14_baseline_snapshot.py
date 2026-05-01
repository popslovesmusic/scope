from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


REPO_ROOT_PATH = Path(REPO_ROOT).resolve()


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _round(x: Any, ndigits: int = 6) -> Any:
    if isinstance(x, float):
        return float(round(x, ndigits))
    return x


def _round_dict(d: Dict[str, Any], ndigits: int = 6) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = _round_dict(v, ndigits=ndigits)
        elif isinstance(v, list):
            out[k] = [_round(i, ndigits) if not isinstance(i, dict) else _round_dict(i, ndigits=ndigits) for i in v]
        else:
            out[k] = _round(v, ndigits)
    return out


def run_one(*, config: Dict[str, Any], prompt: str, seed: Optional[int]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    signature_size = int(config.get("signature_size", 12))
    state = SignatureState(signature_size)
    state.input_trace.append({"input": prompt, "seed": seed})
    state.signed_field = encode_input_to_signed_field(prompt, signature_size, seed=seed)
    state.derive_amplitude_from_signed(0.0, 1.0)

    trace = run_reasoning(state, dict(config))

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

    output_model = config.get("output_model", None) or {
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

    last = trace[-1] if trace else {}
    operator_family = ["++", "--", "+-", "-+"]
    raw_scores = last.get("raw_operator_scores", last.get("operator_scores", {})) or {}
    diff_scores = last.get("diffused_operator_scores", raw_scores) or {}
    used_scores = diff_scores if bool(last.get("orientation_diffusion_applied", False)) else raw_scores

    summary = {
        "selected_class": int(selected_class),
        "confidence": float(confidence),
        "active_operator": last.get("selected_operator", None),
        "phase_count": int(len(trace)),
        "component_count": int(len(getattr(state, "components", []) or [])),
        "active_component_id": getattr(state, "active_component_id", None),
        "raw_caution_scalar": float(getattr(state, "raw_caution_scalar", 0.0)),
        "caution_scalar": float(getattr(state, "caution_scalar", 0.0)),
        "recovery_scalar": float(getattr(state, "recovery_scalar", 0.0)),
        "hold_state": bool(getattr(state, "hold_state", False)),
        "hold_semantics": str(config.get("hold_semantics", "decay")),
        "symmetry_mode_used": str(last.get("symmetry_mode_used", "")),
        "wraparound_lattice": bool(getattr(corridor, "wraparound_lattice", True)),
        "orientation_diffusion_applied": bool(last.get("orientation_diffusion_applied", False)),
        "operator_scores_used": [float(used_scores.get(k, 0.0)) for k in operator_family],
        "trace_last": {
            "hold_reason": str(last.get("hold_reason", "")),
            "hold_released": bool(last.get("hold_released", False)),
            "hold_release_reason": str(last.get("hold_release_reason", "")),
            "hold_release_counter": int(last.get("hold_release_counter", 0)),
            "caution_after_recovery": float(last.get("caution_after_recovery", 0.0)),
        },
    }

    return _round_dict(summary, ndigits=6), trace


def compute_snapshot(
    *,
    prompt: str = "v14_baseline_freeze",
    seed: int = 123,
    config_paths: Optional[List[str]] = None,
) -> Dict[str, Any]:
    if config_paths is None:
        config_paths = ["config/config_v14_scaffold.json", "config/config_v14_terminal.json"]

    configs: Dict[str, Any] = {}
    trace_digests: Dict[str, str] = {}

    for rel in config_paths:
        p = REPO_ROOT_PATH / rel
        cfg = _load_json(p)
        summary, trace = run_one(config=cfg, prompt=prompt, seed=seed)
        configs[rel] = summary

        # Digest trace without huge arrays (trace already avoids raw arrays).
        payload = json.dumps(_round_dict({"trace": trace}, ndigits=6), sort_keys=True).encode("utf-8")
        trace_digests[rel] = _sha256_bytes(payload)

    return {
        "snapshot_version": 1,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "prompt": prompt,
        "seed": int(seed),
        "configs": configs,
        "trace_digests": trace_digests,
    }


def compute_manifest(*, snapshot: Dict[str, Any], include_globs: Optional[List[str]] = None) -> Dict[str, Any]:
    if include_globs is None:
        include_globs = [
            "config/*.json",
            "core/*.py",
            "interfaces/*.py",
            "scripts/*.py",
            "sim_v14_stage1.py",
            "tests/*.py",
            "V14_BASELINE_FREEZE.md",
            "V14_SCHEMA_NOTE.md",
        ]

    seen: set[Path] = set()
    files: List[Dict[str, Any]] = []
    for pattern in include_globs:
        for p in REPO_ROOT_PATH.glob(pattern):
            if p.is_dir():
                continue
            rp = p.resolve()
            if rp in seen:
                continue
            seen.add(rp)
            b = p.read_bytes()
            files.append(
                {
                    "path": str(p.relative_to(REPO_ROOT_PATH)).replace("\\", "/"),
                    "bytes": int(len(b)),
                    "sha256": _sha256_bytes(b),
                }
            )

    files.sort(key=lambda d: d["path"])
    snapshot_bytes = (json.dumps(snapshot, indent=2, sort_keys=True) + "\n").encode("utf-8")
    return {
        "baseline": {"name": "v14", "date": "2026-04-11"},
        "provenance": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": getattr(np, "__version__", "unknown"),
        },
        "snapshot": {"path": "baseline/v14_baseline_snapshot.json", "sha256": _sha256_bytes(snapshot_bytes), "bytes": int(len(snapshot_bytes))},
        "files": files,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="write snapshot + manifest to baseline/")
    ap.add_argument("--prompt", default="v14_baseline_freeze")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    snapshot = compute_snapshot(prompt=args.prompt, seed=args.seed)
    manifest = compute_manifest(snapshot=snapshot)

    if args.write:
        out_dir = REPO_ROOT_PATH / "baseline"
        out_dir.mkdir(parents=True, exist_ok=True)
        snapshot_bytes = (json.dumps(snapshot, indent=2, sort_keys=True) + "\n").encode("utf-8")
        manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
        (out_dir / "v14_baseline_snapshot.json").write_bytes(snapshot_bytes)
        (out_dir / "v14_baseline_manifest.json").write_bytes(manifest_bytes)
        print("wrote baseline snapshot + manifest")
    else:
        print(json.dumps(snapshot, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
