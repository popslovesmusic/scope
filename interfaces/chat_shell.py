
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from core.corridor_gate import build_dynamic_corridor
from core.io_state import load_session_state, save_session_state
from core.memory_layer import (
    PersistentMemoryState,
    apply_commit_gate_and_persistence,
    append_turn_residue,
    append_committed_residue_record,
    build_turn_residue,
    classify_intent,
    CommittedResidueRecord,
    detect_structured_input,
    load_memory_state,
    memory_panel_text,
    save_memory_state,
    qualify_residue,
)
from core.peak_detector import detect_peaks_and_bands
from core.readout_heads import binary_readout
from core.reasoning_loop import run_reasoning
from core.review_suppression_policy import evaluate_review_triggered_suppression
from core.admissibility_gate import summarize_shadow_admissibility
from core.misleading_positive_detector import detect_misleading_positive
from core.projection_path import build_projection_path
from core.semantic_readout import generate_reply
from core.trace_properties import (
    TRACE_HIGHLIGHT_EXTRACTOR_V1,
    extract_triggered_highlights,
    format_highlight_stream,
    format_trace_property_highlights,
    infer_trace_properties,
)
from core.turn_record_pack import build_turn_record, build_turn_record_pack, format_turn_record_json
from core.session_state import SessionState
from core.signature_state import SignatureState
from interfaces.response_renderer import render_debug_panel, render_readback, render_response, render_trace_summary


def encode_input_to_signed_field(text: str, size: int, seed: Optional[int] = None) -> np.ndarray:
    import hashlib

    text = "" if text is None else str(text)
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    base_seed = int(h[:16], 16) ^ int(h[16:32], 16)
    if seed is not None:
        base_seed ^= int(seed) & 0xFFFFFFFF
    rng = np.random.default_rng(base_seed)
    signed = (rng.random(int(size)) * 2.0) - 1.0
    signed = np.clip(signed, -1.0, 1.0)
    return signed.astype(float)


def _load_config(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        repo_root = Path(__file__).resolve().parents[1]
        p = repo_root / p
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _policy_review_reply(*, highlights_text: str, tag: str, refuse: bool) -> str:
    head = "Review:" if not refuse else "Review/Refuse:"
    tag_txt = f" tag={tag}" if tag else ""
    if refuse:
        return "\n".join(
            [
                f"{head}{tag_txt}",
                "I am not committing persistence for this turn because continuation did not restabilize.",
                highlights_text,
            ]
        )
    return "\n".join([f"{head}{tag_txt}", "I am suppressing persistence for this turn due to instability/high pressure.", highlights_text])


def _policy_repair_reply(*, expected_mode: str, actual_mode: str, intent_class: str) -> str:
    return (
        "Repair-to-intent:\n"
        + f"inferred_intent={intent_class} expected_mode={expected_mode} actual_mode={actual_mode}\n"
        + "I will re-anchor to your task mode. Try rephrasing your request in one sentence, or ask a direct question."
    )


def _resolve_path_like_load_config(path: str) -> Path:
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        repo_root = Path(__file__).resolve().parents[1]
        p = repo_root / p
    return p.resolve()


class ChatShell:
    def __init__(self, *, config_path: str = "config/config_v14_terminal.json", seed: Optional[int] = None):
        self.session = SessionState(current_seed=seed, current_config_path=str(config_path))
        self.config = _load_config(self.session.current_config_path)
        self._config_path_resolved = _resolve_path_like_load_config(self.session.current_config_path)
        self._sync_session_flags_from_config()
        self._memory_enabled_config, self._memory_cfg = self._load_memory_cfg(self.config)
        self._memory_override: Optional[bool] = None
        self._memory_enabled = bool(self._memory_enabled_config)
        self._memory_last: Dict[str, Any] = {}
        self._memory_state = self._load_or_init_memory_state()

    def _sync_session_flags_from_config(self) -> None:
        self.session.debug_mode = bool(self.config.get("debug_mode", self.session.debug_mode))
        self.session.inline_trace_mode = bool(self.config.get("inline_trace_mode", self.session.inline_trace_mode))

    @staticmethod
    def _load_memory_cfg(config: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        ml = config.get("memory_layer", {}) if isinstance(config, dict) else {}
        ml = ml if isinstance(ml, dict) else {}
        enabled = bool(ml.get("enabled", False))
        return enabled, dict(ml)

    def _memory_effective_enabled(self) -> bool:
        return bool(self._memory_override) if (self._memory_override is not None) else bool(self._memory_enabled_config)

    def _load_or_init_memory_state(self) -> PersistentMemoryState:
        if isinstance(self.session.memory_state, dict):
            try:
                return PersistentMemoryState.from_dict(self.session.memory_state)
            except Exception:
                pass

        self._memory_enabled = self._memory_effective_enabled()
        if not self._memory_enabled:
            return PersistentMemoryState()

        path = str(self._memory_cfg.get("memory_state_path", self.session.memory_state_path))
        residue_path = str(self._memory_cfg.get("turn_residue_path", self.session.turn_residue_path))
        committed_path = str(self._memory_cfg.get("committed_residue_path", self.session.committed_residue_path))
        self.session.memory_state_path = path
        self.session.turn_residue_path = residue_path
        self.session.committed_residue_path = committed_path
        return load_memory_state(path)

    def _persist_memory_state(self) -> None:
        self.session.memory_state = self._memory_state.to_dict()
        if not self._memory_effective_enabled():
            return
        if not bool(self._memory_cfg.get("persist", True)):
            return
        save_memory_state(self.session.memory_state_path, self._memory_state)
        self._memory_last["saved_at"] = datetime.utcnow().isoformat() + "Z"

    def _memory_delta_hint_text(self) -> str:
        if not self._memory_effective_enabled():
            return ""
        last = dict(self._memory_last or {})
        if not last.get("residue_appended", False):
            return ""

        post_committed = int(getattr(self._memory_state, "committed_residue_count", 0))
        delta_committed = last.get("delta_committed_residue_count", None)

        commit_flag = bool(getattr(self._memory_state, "last_commit", False))
        reason = str(getattr(self._memory_state, "last_commit_reason", "") or getattr(self._memory_state, "last_reject_reason", "") or "")

        ob = getattr(self._memory_state, "operator_bias", {}) or {}
        top_op = None
        top_val = 0.0
        if isinstance(ob, dict) and ob:
            try:
                top_op, top_val = max(((str(k), float(v)) for k, v in ob.items()), key=lambda kv: kv[1])
            except Exception:
                top_op, top_val = None, 0.0

        shift = float(getattr(self._memory_state, "caution_baseline_shift", 0.0))

        pieces = [
            f"MemoryDelta: commit={1 if commit_flag else 0}",
            (f"reason={reason}" if reason else None),
            (f"committed={post_committed} ({int(delta_committed):+d})" if isinstance(delta_committed, int) else f"committed={post_committed}"),
            (f"bias={top_op}:{top_val:.2f}" if top_op is not None else None),
            f"shift={shift:.3f}",
        ]
        return " | ".join([p for p in pieces if p])

    def _memory_status_text(self) -> str:
        enabled_config = bool(self._memory_enabled_config)
        enabled_effective = bool(self._memory_effective_enabled())
        override = self._memory_override

        state_path = Path(self.session.memory_state_path).resolve()
        residue_path = Path(self.session.turn_residue_path).resolve()
        committed_path = Path(self.session.committed_residue_path).resolve()
        state_exists = state_path.exists()
        residue_exists = residue_path.exists()
        committed_exists = committed_path.exists()
        residue_lines = 0
        if residue_exists:
            try:
                with open(residue_path, "r", encoding="utf-8") as f:
                    for residue_lines, _ in enumerate(f, start=1):
                        pass
            except Exception:
                residue_lines = -1
        committed_lines = 0
        if committed_exists:
            try:
                with open(committed_path, "r", encoding="utf-8") as f:
                    for committed_lines, _ in enumerate(f, start=1):
                        pass
            except Exception:
                committed_lines = -1

        last = dict(self._memory_last or {})
        injected = bool(last.get("injected", False))
        last_error = str(last.get("error", "") or "")

        lines = [
            "Memory status:",
            f"enabled_effective={enabled_effective} enabled_config={enabled_config} override={override}",
            f"config_path={self.session.current_config_path}",
            f"config_path_resolved={self._config_path_resolved}",
            f"memory_state_path={state_path} exists={state_exists}",
            f"turn_residue_path={residue_path} exists={residue_exists} raw_residue_count={residue_lines}",
            f"committed_residue_path={committed_path} exists={committed_exists} committed_residue_log_count={committed_lines}",
            f"turn_counter={int(getattr(self._memory_state,
 'turn_counter', 0))}",
            f"qualified_residue_count={int(getattr(self._memory_state, 'qualified_residue_count', 0))}",
            f"committed_residue_count={int(getattr(self._memory_state, 'committed_residue_count', 0))}",
            f"rejected_residue_count={int(getattr(self._memory_state, 'rejected_residue_count', 0))}",
            f"average_stability_score={float(getattr(self._memory_state, 'average_stability_score', 0.0))}",
            f"last_injected={injected}",
        ]
        if last:
            if "operator_bias_strength" in last:
                lines.append(f"last_operator_bias_strength={last.get('operator_bias_strength')}")
            if "operator_bias_l1" in last:
                lines.append(f"last_operator_bias_l1={last.get('operator_bias_l1')}")
            if "caution_baseline_shift" in last:
                lines.append(f"last_caution_baseline_shift={last.get('caution_baseline_shift')}")
            if "caution_adjustment" in last:
                lines.append(f"last_caution_adjustment={last.get('caution_adjustment')}")
            if "residue_appended" in last:
                lines.append(f"last_residue_appended={last.get('residue_appended')}")
            if "committed_record_appended" in last:
                lines.append(f"last_committed_record_appended={last.get('committed_record_appended')}")
            if "residue_turn_id" in last:
                lines.append(f"last_residue_turn_id={last.get('residue_turn_id')}")
            if "residue_carry_weight" in last:
                lines.append(f"last_residue_carry_weight={last.get('residue_carry_weight')}")
            if "residue_dominant_operator" in last:
                lines.append(f"last_residue_dominant_operator={last.get('residue_dominant_operator')}")
            if "residue_structured_input" in last:
                lines.append(f"last_residue_structured_input={last.get('residue_structured_input')}")
            if "residue_stability_score" in last:
                lines.append(f"last_residue_stability_score={last.get('residue_stability_score')}")
            if "residue_is_qualified" in last:
                lines.append(f"last_is_qualified={last.get('residue_is_qualified')}")
            if "residue_is_committed" in last:
                lines.append(f"last_is_committed={last.get('residue_is_committed')}")
            if "last_commit" in last:
                lines.append(f"last_commit={last.get('last_commit')}")
            if "last_commit_reason" in last:
                lines.append(f"last_commit_reason={last.get('last_commit_reason')}")
            if "last_reject_reason" in last:
                lines.append(f"last_reject_reason={last.get('last_reject_reason')}")
            if "saved_at" in last:
                lines.append(f"last_saved_at={last.get('saved_at')}")
            if "policy_rule" in last:
                lines.append(f"last_policy_rule={last.get('policy_rule')}")
            if "policy_tag" in last:
                lines.append(f"last_policy_tag={last.get('policy_tag')}")
            if "policy_response_mode" in last:
                lines.append(f"last_policy_response_mode={last.get('policy_response_mode')}")
        if last_error:
            lines.append(f"last_error={last_error}")
        return "\n".join(lines)

    def _recent_runtime_records(self, turns: int) -> list[dict]:
        turns = int(max(1, turns))
        max_turns = int(TRACE_HIGHLIGHT_EXTRACTOR_V1["windowing"]["max_turns"])
        turns = int(min(turns, max_turns))
        hist = [r for r in (self.session.runtime_history or []) if isinstance(r, dict)]
        if hist:
            return hist[-turns:]
        out = self.session.last_runtime_output
        return [out] if isinstance(out, dict) else []

    def _trace_highlights_text(self, *, turns: int, include_supporting_values: bool) -> str:
        default_recent = int(TRACE_HIGHLIGHT_EXTRACTOR_V1["windowing"]["default_recent_turns"])
        turns = int(turns) if turns is not None else int(default_recent)
        turn_recs = self._recent_runtime_records(turns)
        items = []
        for rec in turn_recs:
            cfg_used = rec.get("config_used", None) if isinstance(rec, dict) else None
            cfg = cfg_used if isinstance(cfg_used, dict) and cfg_used else self.config
            mem = rec.get("memory_summary", None) if isinstance(rec, dict) else None
            hs = extract_triggered_highlights(
                runtime_output=rec,
                config=cfg,
                memory_last=(mem if isinstance(mem, dict) else self._memory_last),
                max_per_turn=int(TRACE_HIGHLIGHT_EXTRACTOR_V1["selection"]["max_highlights_per_turn"]),
                sort_by=list(TRACE_HIGHLIGHT_EXTRACTOR_V1["selection"]["sort_by"]),
            )
            turn_id = int(rec.get("turn_id", 0) or 0)
            items.append((turn_id, hs))
        return format_highlight_stream(highlights_by_turn=items, include_supporting_values=include_supporting_values)

    def _run_turn(self, prompt: str) -> Dict[str, Any]:
        cfg = dict(self.config)
        self._memory_enabled = self._memory_effective_enabled()
        self._memory_last = {"injected": False, "residue_appended": False, "error": ""}
        # Always compute a lightweight intent class for turn-level records (not memory).
        intent_category_global = classify_intent(prompt)
        if self._memory_enabled:
            self._memory_last["pre_committed_residue_count"] = int(getattr(self._memory_state, "committed_residue_count", 0))
            self._memory_last["pre_qualified_residue_count"] = int(getattr(self._memory_state, "qualified_residue_count", 0))
            self._memory_last["pre_rejected_residue_count"] = int(getattr(self._memory_state, "rejected_residue_count", 0))
            self._memory_last["pre_operator_bias"] = dict(getattr(self._memory_state, "operator_bias", {}) or {})
            self._memory_last["pre_caution_baseline_shift"] = float(getattr(self._memory_state, "caution_baseline_shift", 0.0))
            cfg["memory_enabled"] = True
            op_bias = dict(self._memory_state.operator_bias)
            op_strength = float(self._memory_cfg.get("operator_bias_strength", 0.08))
            caution_shift = float(self._memory_state.caution_baseline_shift)
            caution_strength = float(self._memory_cfg.get("caution_baseline_strength", 0.25))

            cfg["memory_operator_bias"] = op_bias
            cfg["memory_operator_bias_strength"] = op_strength
            cfg["memory_caution_baseline_shift"] = caution_shift
            cfg["memory_caution_baseline_strength"] = caution_strength

            self._memory_last["injected"] = True
            self._memory_last["operator_bias_strength"] = op_strength
            self._memory_last["operator_bias_l1"] = float(sum(abs(float(v)) for v in op_bias.values()))
            self._memory_last["caution_baseline_shift"] = caution_shift
            self._memory_last["caution_adjustment"] = float(caution_shift * caution_strength)
        signature_size = int(cfg.get("signature_size", 12))
        state = SignatureState(signature_size)
        state.input_trace.append({"input": prompt, "seed": self.session.current_seed})
        state.signed_field = encode_input_to_signed_field(prompt, signature_size, seed=self.session.current_seed)
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
            wraparound_lattice=bool(cfg.get("use_wraparound_lattice", True)),
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
            caution_corridor_penalty=float(cfg.get("caution_corridor_penalty", 0.0)),
            caution_entry_penalty=cfg.get("caution_entry_penalty", None),
            caution_exit_penalty=cfg.get("caution_exit_penalty", None),
            corridor_recovery_gain=float(cfg.get("corridor_recovery_gain",
 0.0)),
            max_recovery_fraction_of_base_window=float(cfg.get("max_recovery_fraction_of_base_window", 0.5)),
        )

        output_model = cfg.get("output_model", None) or {
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
        operator_scores_used = diff_scores_dict if bool(last_trace.get("orientation_diffusion_applied", False)) else raw_scores_dict

        state_dict = {
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
                "hold_state": bool(getattr(state, "hold_state", False)),
                "recovery_scalar": float(getattr(state, "recovery_scalar", 0.0)),
                "recontextualization_score": float(getattr(state, "recontextualization_score", 0.0)),
                "hold_release_counter": int(getattr(state, "hold_release_counter", 0)),
                "caution_field": getattr(state, "caution_field", np.zeros(signature_size)).tolist(),
            },
            "corridor": {
                "window": corridor.window.tolist(),
                "entry_resistance": corridor.entry_resistance.tolist(),
                "exit_resistance": corridor.exit_resistance.tolist(),
                "threshold": float(corridor.threshold),
                "wraparound_lattice": bool(getattr(corridor, "wraparound_lattice", True)),
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
                "hold_semantics": str(cfg.get("hold_semantics", "decay")),
                "symmetry_mode_used": str(last_trace.get("symmetry_mode_used", "")),
            },
        }

        out = {
            "prompt": prompt,
            "seed": self.session.current_seed,
            "intent_class": str(intent_category_global),
            "state": state_dict,
            "trace": trace,
            "output": {
                "class_scores": class_scores,
                "selected_class": int(selected_class),
                "confidence": float(confidence),  # legacy
                "practical_confidence": float(confidence),
            },
        }
        out["config_used"] = {
            "signature_size": int(signature_size),
            "caution_threshold": float(cfg.get("caution_threshold", 0.8)),
        }

        # Shadow admissibility + dual confidence (additive; does not enforce blocking by default).
        shadow = summarize_shadow_admissibility(trace=trace)
        out["admissibility_shadow"] = shadow
        caution_scalar = float(getattr(state, "caution_scalar", 0.0))
        hold_state = bool(getattr(state, "hold_state", False))
        adm_final = float(shadow.get("final_score", 0.0) or 0.0)
        structural_confidence = float(max(0.0, min(1.0, 0.55 * adm_final + 0.45 * (1.0 - max(0.0, min(1.0, caution_scalar))))))
        if hold_state:
            structural_confidence *= 0.5
        out["output"]["structural_confidence"] = float(structural_confidence)

        mp_cfg = cfg.get("misleading_positive_detector", {}) if isinstance(cfg.get("misleading_positive_detector", {}), dict) else {}
        mp = detect_misleading_positive(
            practical_confidence=float(confidence),
            structural_confidence=float(structural_confidence),
            practical_high=float(mp_cfg.get("practical_high", 0.60)),
            structural_low=float(mp_cfg.get("structural_low", 0.45)),
            gap_min=float(mp_cfg.get("gap_min", 0.20)),
        )
        out["misleading_positive"] = mp.to_dict()
        out["projection_path"] = build_projection_path(
            runtime_output=out,
            practical_confidence=float(confidence),
            structural_confidence=float(structural_confidence),
            misleading_positive=bool(mp.flagged),
        )

        reply = generate_reply(prompt=prompt, runtime_output=out, config=cfg)
        if reply:
            out["reply"] = reply

        # Review-triggered suppression policy (trace-derived; no explicit ethical scoring).
        suppression_cfg = cfg.get("review_triggered_suppression", {}) if isinstance(cfg.get("review_triggered_suppression", {}), dict) else {}
        suppression_enabled = bool(suppression_cfg.get("enabled", False))
        suppression_suppress_persistence = bool(suppression_cfg.get("suppress_persistence", True))
        suppression_override_reply = bool(suppression_cfg.get("override_reply", True))

        suppression_decision = None
        property_records = infer_trace_properties(runtime_output=out, config=cfg, memory_last=None)
        if suppression_enabled:
            suppression_decision = evaluate_review_triggered_suppression(property_records=property_records, enabled=True)
            out["review_triggered_suppression"] = suppression_decision.to_dict()
        # Trace-driven ethical trigger inputs (non-enforcing; exported for inspection).
        try:
            trig = []
            if isinstance(property_records, list):
                trig = [str(r.get("property_name")) for r in property_records if isinstance(r, dict) and bool(r.get("triggered", False))]
            out["ethical_trigger_inputs"] = {
                "turn_id": int(out.get("turn_id", 0) or 0),
                "triggered_properties": trig,
                "response_mode_override": str(out.get("response_mode_override", "") or ""),
                "suppression": (suppression_decision.to_dict() if suppression_decision is not None else {}),
                "misleading_positive": (out.get("misleading_positive", {}) if isinstance(out.get("misleading_positive", {}), dict) else {}),
                "admissibility_final": float(
                    (out.get("admissibility_shadow", {}) if isinstance(out.get("admissibility_shadow", {}), dict) else {}).get("final_score", 0.0) or 0.0
                ),
            }
        except Exception:
            pass

            # Optionally override reply behavior to a review/repair mode.
            mode = str(getattr(suppression_decision, "response_mode_override", "") or "")
            if suppression_override_reply and mode:
                highlights = extract_triggered_highlights(runtime_output=out, config=cfg, memory_last=None)
                highlights_text = format_highlight_stream(highlights_by_turn=[(int(self.session.turn_history.__len__() + 1), highlights)], include_supporting_values=True)
                tag = str(getattr(suppression_decision, "tag", "") or "")
                if mode in {"review", "review_or_refuse"}:
                    out["reply"] = _policy_review_reply(highlights_text=highlights_text, tag=tag, refuse=(mode == "review_or_refuse"))
                    out["response_mode_override"] = mode
                elif mode == "repair_to_intent":
                    sf = {}
                    if isinstance(property_records, list):
                        for r in property_records:
                            if isinstance(r, dict) and r.get("property_name") == "intent_mode_mismatch":
                                sf = r.get("supporting_fields", {}) if isinstance(r.get("supporting_fields", {}), dict) else {}
                                break
                    out["reply"] = _policy_repair_reply(
                        expected_mode=str(sf.get("expected_reply_mode", "") or ""),
                        actual_mode=str(sf.get("actual_reply_mode", "") or ""),
                        intent_class=str(sf.get("intent_class", "") or ""),
                    )
                    out["response_mode_override"] = mode

        if self._memory_enabled:
            try:
                intent_category = classify_intent(prompt)
                sr = cfg.get("semantic_readout", {}) if isinstance(cfg.get("semantic_readout", {}), dict) else {}
                reply_mode = str(sr.get("backend", "local")) if bool(sr.get("enabled", False)) else "renderer"
                structured_mode = str(self._memory_cfg.get("structured_mode", "auto"))
                structured_input = detect_structured_input(prompt, mode=structured_mode)
                residue = build_turn_residue(
                    runtime_output=out,
                    prompt_text=prompt,
                    intent_category=intent_category,
                    reply_mode=reply_mode,
                    turn_id=int(self._memory_state.turn_counter) + 1,
                    structured_input=structured_input,
                )
                residue = qualify_residue(
                    residue,
                    structured_input=structured_input,
                    epsilon=float(self._memory_cfg.get("qualify_epsilon", 0.10)),
                    recovery_threshold=float(self._memory_cfg.get("qualify_recovery_threshold", 0.02)),
                    max_switch_freq=float(self._memory_cfg.get("qualify_max_switch_freq", 0.50)),
                    min_score=float(self._memory_cfg.get("qualify_min_score", 0.65)),
                    min_admissibility=float(self._memory_cfg.get("qualify_min_admissibility", 0.0)),
                )

                # Policy can suppress persistence even if the residue is otherwise qualified.
                if suppression_enabled and suppression_decision is not None and suppression_suppress_persistence:
                    if not bool(getattr(suppression_decision, "allow_commit", True)):
                        residue.is_qualified = False
                        residue.is_committed = False
                        residue.commit_reason = ""
                        residue.reject_reason = f"policy:{getattr(suppression_decision, 'rule_name', '')}:{getattr(suppression_decision, 'tag', '')}".strip(":")
                        self._memory_last["policy_rule"] = str(getattr(suppression_decision, "rule_name", "") or "")
                        self._memory_last["policy_tag"] = str(getattr(suppression_decision, "tag", "") or "")
                        self._memory_last["policy_response_mode"] = str(getattr(suppression_decision, "response_mode_override", "") or "")

                append_turn_residue(self.session.turn_residue_path, residue)
                self._memory_last["residue_appended"] = True
                self._memory_last["residue_turn_id"] = int(residue.turn_id)
                self._memory_last["residue_carry_weight"] = float(residue.carry_weight)
                self._memory_last["residue_dominant_operator"] = str(residue.dominant_operator)
                self._memory_last["residue_structured_input"] = bool(structured_input)
                self._memory_last["residue_stability_score"] = float(residue.stability_score)
                self._memory_last["residue_is_qualified"] = bool(residue.is_qualified)
                self._memory_last["residue_is_committed"] = bool(residue.is_committed)
                self._memory_last["residue_commit_reason"] = str(residue.commit_reason)
                self._memory_last["residue_reject_reason"] = str(residue.reject_reason)

                self._memory_state, residue = apply_commit_gate_and_persistence(
                    state=self._memory_state,
                    residue=residue,
                    base_duration=int(self._memory_cfg.get("base_persistence_duration", 3)),
                    reinforce=int(self._memory_cfg.get("reinforce_duration", 2)),
                    max_duration=int(self._memory_cfg.get("max_persistence_duration", 12)),
                )
                self._memory_last["delta_committed_residue_count"] = int(getattr(self._memory_state, "committed_residue_count", 0)) - int(
                    self._memory_last.get("pre_committed_residue_count", 0)
                )
                self._memory_last["delta_qualified_residue_count"] = int(getattr(self._memory_state, "qualified_residue_count", 0)) - int(
                    self._memory_last.get("pre_qualified_residue_count", 0)
                )
                self._memory_last["delta_rejected_residue_count"] = int(getattr(self._memory_state, "rejected_residue_count", 0)) - int(
                    self._memory_last.get("pre_rejected_residue_count", 0)
                )
                self._memory_last["residue_is_committed"] = bool(residue.is_committed)
                self._memory_last["memory_committed_residue_count"] = int(getattr(self._memory_state, "committed_residue_count", 0))
                self._memory_last["memory_qualified_residue_count"] = int(getattr(self._memory_state, "qualified_residue_count", 0))
                self._memory_last["memory_rejected_residue_count"] = int(getattr(self._memory_state, "rejected_residue_count", 0))
                self._memory_last["memory_average_stability_score"] = float(getattr(self._memory_state, "average_stability_score", 0.0))
                self._memory_last["last_commit"] = bool(getattr(self._memory_state, "last_commit", False))
                self._memory_last["last_commit_reason"] = str(getattr(self._memory_state, "last_commit_reason", ""))
                self._memory_last["last_reject_reason"] = str(getattr(self._memory_state, "last_reject_reason", ""))
                self._persist_memory_state()

                # Append a committed residue record (committed-only log).
                committed_record_dict = None
                if bool(residue.is_committed):
                    pre_bias = self._memory_last.get("pre_operator_bias", {}) or {}
                    post_bias = dict(getattr(self._memory_state, "operator_bias", {}) or {})
                    delta = {}
                    for k in set(list(pre_bias.keys()) + list(post_bias.keys())):
                        try:
                            delta[str(k)] = float(post_bias.get(k, 0.0)) - float(pre_bias.get(k, 0.0))
                        except Exception:
                            delta[str(k)] = 0.0
                    pre_shift = float(self._memory_last.get("pre_caution_baseline_shift", 0.0) or 0.0)
                    post_shift = float(getattr(self._memory_state, "caution_baseline_shift", 0.0))
                    committed_path = str(self._memory_cfg.get("committed_residue_path", self.session.committed_residue_path))
                    rec = CommittedResidueRecord(
                        turn_id=int(residue.turn_id),
                        commit_reasons=[str(residue.commit_reason or self._memory_state.last_commit_reason or "committed")],
                        operator_bias_delta=delta,
                        caution_baseline_delta=float(post_shift - pre_shift),
                        persistence_duration=int(residue.persistence_duration),
                    )
                    append_committed_residue_record(committed_path, rec)
                    self._memory_last["committed_record_appended"] = True
                    committed_record_dict = dict(rec.to_dict(), **{"commit_decision": "commit"})

                out["raw_residue_record"] = residue.to_dict()
                if committed_record_dict is not None:
                    out["committed_residue_record"] = committed_record_dict
                else:
                    out["committed_residue_record"] = {
                        "turn_id": int(residue.turn_id),
                        "commit_decision": "reject",
                        "commit_reasons": [],
                        "reject_reasons": ([str(residue.reject_reason)] if residue.reject_reason else []),
                        "operator_bias_delta": {},
                        "caution_baseline_delta": 0.0,
                        "persistence_duration": 0,
                    }

                out["memory_summary"] = {
                    "residue_appended": bool(self._memory_last.get("residue_appended", False)),
                    "residue_turn_id": int(self._memory_last.get("residue_turn_id", 0) or 0),
                    "residue_is_qualified": bool(self._memory_last.get("residue_is_qualified", False)),
                    "residue_is_committed": bool(self._memory_last.get("residue_is_committed", False)),
                    "residue_commit_reason": str(self._memory_last.get("residue_commit_reason", "") or ""),
                    "residue_reject_reason": str(self._memory_last.get("residue_reject_reason", "") or ""),
                    "commit_decision": ("commit" if bool(self._memory_last.get("residue_is_committed", False)) else "reject"),
                    "delta_committed": int(self._memory_last.get("delta_committed_residue_count", 0) or 0),
                    "intent_class": str(intent_category),
                    "reply_mode": str(reply_mode),
                }
            except Exception as e:
                self._memory_last["error"] = f"{type(e).__name__}: {e}"
        return out

    def process_line(self, line: str) -> Tuple[bool, str]:
        line = "" if line is None else str(line).strip()
        if not line:
            return True, ""

        if line.startswith("/"):
            return self._handle_command(line)

        runtime_output = self._run_turn(line)
        self.session.last_runtime_output = runtime_output
        self.session.add_turn(
            {
                "prompt": line,
                "seed": self.session.current_seed,
                "selected_class": runtime_output["output"]["selected_class"],
            }
        )
        # Stable shell-only turn index used by trace-property highlighting.
        try:
            runtime_output["turn_id"] = int(len(self.session.turn_history))
        except Exception:
            pass
        try:
            compact = {
                "turn_id": int(runtime_output.get("turn_id",
 0) or 0),
                "prompt": runtime_output.get("prompt", ""),
                "reply": runtime_output.get("reply", ""),
                "intent_class": runtime_output.get("intent_class", ""),
                "response_mode_override": runtime_output.get("response_mode_override", ""),
                "review_triggered_suppression": runtime_output.get("review_triggered_suppression", {}),
                "admissibility_shadow": runtime_output.get("admissibility_shadow", {}),
                "misleading_positive": runtime_output.get("misleading_positive", {}),
                "projection_path": runtime_output.get("projection_path", {}),
                "ethical_trigger_inputs": runtime_output.get("ethical_trigger_inputs", {}),
                "trace": runtime_output.get("trace", []),
                "output": runtime_output.get("output", {}),
                "config_used": runtime_output.get("config_used", {}),
                "memory_summary": runtime_output.get("memory_summary", {}),
                "raw_residue_record": runtime_output.get("raw_residue_record", {}),
                "committed_residue_record": runtime_output.get("committed_residue_record", {}),
            }
            self.session.runtime_history.append(compact)
            keep_n = int(TRACE_HIGHLIGHT_EXTRACTOR_V1["storage"].get("highlight_cache_turns", 16))
            if keep_n > 0 and len(self.session.runtime_history) > keep_n:
                self.session.runtime_history = self.session.runtime_history[-keep_n:]
        except Exception:
            pass

        parts = [
            render_response(runtime_output, user_text=line),
            render_readback(runtime_output),
            (self._memory_delta_hint_text() if self._memory_effective_enabled() else ""),
        ]
        if self.session.inline_trace_mode:
            parts.append(
                render_trace_summary(
                    runtime_output,
                    mode="compact",
                    limit=int(self.config.get("trace_default_depth", 1)),
                )
            )
        if self.session.debug_mode:
            parts.append(render_debug_panel(runtime_output))
        return True, "\n".join([p for p in parts if p])

    def _handle_command(self, cmdline: str) -> Tuple[bool, str]:
        parts = cmdline.split()
        cmd = parts[0].lower()
        arg = parts[1:] if len(parts) > 1 else []

        if cmd in ("/quit", "/exit"):
            return False, "bye"
        if cmd == "/help":
            return True, self.help_text()
        if cmd == "/debug":
            if arg and arg[0].lower() in ("on", "off"):
                self.session.debug_mode = arg[0].lower() == "on"
                return True, f"debug_mode={self.session.debug_mode}"
            return True, f"debug_mode={self.session.debug_mode}"
        if cmd == "/trace":
            out = self.session.last_runtime_output
            if not out:
                return True, "(no last turn)"
            if arg and str(arg[0]).lower() in {"highlights", "highlight", "review"}:
                sub = str(arg[0]).lower()
                turns = int(TRACE_HIGHLIGHT_EXTRACTOR_V1["windowing"]["default_recent_turns"])
                if len(arg) >= 2:
                    try:
                        turns = int(arg[1])
                    except Exception:
                        turns = int(TRACE_HIGHLIGHT_EXTRACTOR_V1["windowing"]["default_recent_turns"])
                include_support = True if sub == "review" else bool(TRACE_HIGHLIGHT_EXTRACTOR_V1["rendering"]["include_supporting_values"])
                return True, self._trace_highlights_text(turns=turns, include_supporting_values=include_support)
            if arg and str(arg[0]).lower() == "full":
                return True, render_trace_summary(out, mode="full")
            if arg:
                try:
                    limit = int(arg[0])
                except ValueError:
                    limit = int(self.config.get("trace_default_depth", 1))
            else:
                limit = int(self.config.get("trace_default_depth", 1))
            return True, render_trace_summary(out, mode="compact", limit=limit)
        if cmd in ("/props", "/properties"):
            out = self.session.last_runtime_output
            if not out:
                return True, "(no last turn)"
            mode = "compact"
            if arg and str(arg[0]).lower() in {"full", "all", "verbose"}:
                mode = "full"
            records = infer_trace_properties(runtime_output=out, config=self.config, memory_last=self._memory_last)
            return True, format_trace_property_highlights(records=records, mode=mode)
        if cmd in ("/turn", "/record"):
            out = self.session.last_runtime_output
            if not out:
                return True, "(no last turn)"
            sub = str(arg[0]).lower() if arg else ""
            if sub in {"pack", "history"}:
                n = 3
                if len(arg) >= 2:
                    try:
                        n = int(arg[1])
                    except Exception:
                        n = 3
                n = max(1, min(int(n), 12))
                pack = build_turn_record_pack(runtime_records=self._recent_runtime_records(n), config=self.config, memory_last=self._memory_last)
                return True, format_turn_record_json(pack)
            rec = build_turn_record(runtime_output=out, config=self.config, memory_last=self._memory_last)
            return True, format_turn_record_json(rec)
        if cmd == "/state":
            out = self.session.last_runtime_output
            if not out:
                return True, "(no last turn)"
            return True, render_readback(out)
        if cmd == "/memory":
            sub = str(arg[0]).lower() if arg else ""
            if sub in {"status", "info"}:
                return True, self._memory_status_text()
            if sub in {"on", "enable"}:
                self._memory_override = True
                self._memory_state = self._load_or_init_memory_state()
                self._memory_last = {}
                self._persist_memory_state()
                return True, "memory override ON\n" + self._memory_status_text()
            if sub in {"off", "disable"}:
                self._memory_override = False
                self._memory_enabled = self._memory_effective_enabled()
                self._memory_last = {}
                self._persist_memory_state()
                return True, "memory override OFF\n" + self._memory_status_text()
            if sub in {"reset", "clear"}:
                self._memory_state = PersistentMemoryState()
                self._memory_last = {}
                self._persist_memory_state()
                return True, "memory reset ok\n" + self._memory_status_text()
            # Default: show the memory panel or a clear off message.
            self._memory_enabled = self._memory_effective_enabled()
            if not self._memory_enabled:
                return True, "Memory: OFF (use `/memory on` to enable for this session, or `/memory status` for diagnostics)"
            return True, memory_panel_text(self._memory_state) + "\n\n" + self._memory_status_text()
        if cmd == "/reset":
            self.session.reset()
            return True, "reset ok"
        if cmd == "/seed":
            if not arg:
                return True, f"seed={self.session.current_seed}"
            self.session.current_seed = int(arg[0])
            return True, f"seed={self.session.current_seed}"
        if cmd == "/config":
            if not arg:
                return True, f"config={self.session.current_config_path}"
            path = str(arg[0])
            self.config = _load_config(path)
            self.session.current_config_path = path
            self._config_path_resolved = _resolve_path_like_load_config(path)
            self._sync_session_flags_from_config()
            self._memory_enabled_config, self._memory_cfg = self._load_memory_cfg(self.config)
            self._memory_override = None
            self._memory_state = self._load_or_init_memory_state()
            return True, f"config loaded: {path}"
        if cmd == "/save":
            if not arg:
                return True, "usage: /save <name>"
            name = arg[0]
            sessions_dir = Path("sessions")
            sessions_dir.mkdir(parents=True, exist_ok=True)
            path = sessions_dir / f"{name}.json"
            save_session_state(str(path), self.session)
            self.session.saved_state_reference = str(path)
            return True, f"saved: {path}"
        if cmd == "/load":
            if not arg:
                return True, "usage: /load <name>"
            name = arg[0]
            path = Path("sessions") / f"{name}.json"
            self.session = load_session_state(str(path))
            self.config = _load_config(self.session.current_config_path)
            self._config_path_resolved = _resolve_path_like_load_config(self.session.current_config_path)
            self._sync_session_flags_from_config()
            self._memory_enabled_config, self._memory_cfg = self._load_memory_cfg(self.config)
            self._memory_override = None
            self._memory_state = self._load_or_init_memory_state()
            return True, f"loaded: {path}"
        return True, "unknown command (try /help)"

    @staticmethod
    def help_text() -> str:
        return "\n".join(
            [
                "Commands:",
                "/help                show this help",
                "/trace               show last-turn compact trace summary",
                "/trace 5             show last 5 phase summaries",
                "/trace full          show detailed trace for last turn",
                "/trace highlights    show triggered highlights (last 3 turns)",
                "/trace highlights 5  show highlights for last 5 turns",
                "/trace review        verbose highlight review (last 3 turns)",
                "/props               show triggered trace properties (inferred)",
                "/props full          show all trace properties + supporting fields",
                "/turn                show last-turn turn_record json",
                "/turn pack 3         show recent turn_record pack json",
                "/state               show compact state snapshot",
                "/debug on|off         toggle debug panel",
                "/memory              show memory panel",
                "/memory status       show memory diagnostics",
                "/memory on|off       enable/disable memory for this session",
                "/memory reset        clear persistent memory state",
                "/reset               reset session history",
                "/seed <n>            set deterministic seed",
                "/config <path>       switch runtime config json",
                "/save <name>         save session to sessions/<name>.json",
                "/load <name>         load session from sessions/<name>.json",
                "/quit                exit shell",
            ]
        )

    def run(self) -> None:
        mem = "ON" if self._memory_effective_enabled() else "OFF"
        mem_cfg = "ON" if bool(self._memory_enabled_config) else "OFF"
        override = self._memory_override
        print(
            "sbllm v14 shell "
            + f"[config={self.session.current_config_path}] "
            + f"[memory:{mem} config={mem_cfg} override={override}] "
            + "Type /help for commands."
        )
        while True:
            try:
                line = input("> ")
            except (EOFError, KeyboardInterrupt):
                print()
                break
            keep, out = self.process_line(line)
            if out:
                print(out)
            if not keep:
                break
