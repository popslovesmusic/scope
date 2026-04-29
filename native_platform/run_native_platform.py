import os
import json
import numpy as np
from datetime import datetime

from .engine_bridge import EngineBridge
from .signalscope_core import SignalScope
from .hex_state import make_full_hex
from .wheel12_projection import project_to_12, apply_operator_pressure
from .v14_bridge import V14Bridge
from .residue_imprinter import qualify_and_commit
from .feedback_adapter import FeedbackAdapter
from .residue_feedback import residue_bias
from .phase_space import compute_phase_vector, phase_mismatch
from .residue_phase_continuation import ResiduePhaseContinuation
from .phase_operator_map import operator_pressure
from .operator_selection import select_operator, apply_operator
from .groove_router import GrooveRouter
from .signal_layer import compute_x_channel, get_consistency_level
from .inductive_transformer import InductiveTransformerLayer
from .recursive_motion_anchor import RecursiveMotionAnchor
from .phase_refraction_layer import PhaseRefractionLayer

from core.memory_layer import load_memory_state, save_memory_state

def run_platform(num_frames=100, num_nodes=100, engine_steps_per_frame=None, feedback_enabled=None, run_id=None, input_signals=None, memory_path="sessions/native_memory.json", connected=True, connected_state="train", teacher_theta=None, use_adaptive=True, refraction_enabled=True, w_trace=0.75, w_inductive=0.35, w_anchor=0.05, damping=0.88):
    print(f"🚀 Initializing Native Wave-Residue Platform (Run: {run_id}, Mode: {connected_state.upper()}, RefEnabled: {refraction_enabled}, Adaptive: {use_adaptive})...")
    
    if input_signals is not None:
        num_frames = len(input_signals)

    engine = EngineBridge(num_nodes=num_nodes)
    scope = SignalScope()
    v14 = V14Bridge()
    memory = load_memory_state(memory_path)
    
    phase_continuation = ResiduePhaseContinuation(
        history_size=64, trace_size=128, successful_traversals=memory.successful_traversals
    )
    router = GrooveRouter.from_dict(memory.groove_data)
    motion_anchor = RecursiveMotionAnchor.from_dict(memory.recursive_anchor_data)
    refraction = PhaseRefractionLayer.from_dict(
        memory.native_state.get("refraction"), channels=8
    )
    refraction.use_adaptive = use_adaptive
    
    fb_config_path = "native_platform/feedback_config.json"
    if os.path.exists(fb_config_path):
        with open(fb_config_path, 'r') as f:
            fb_config = json.load(f)
    else:
        fb_config = {"feedback": {"enabled": False}}
    
    if engine_steps_per_frame is not None:
        fb_config.setdefault("feedback", {})["engine_steps_per_frame"] = engine_steps_per_frame
    if feedback_enabled is not None:
        fb_config.setdefault("feedback", {})["enabled"] = feedback_enabled

    feedback = FeedbackAdapter(fb_config)
    
    last_state = type('obj', (object,), {'caution_scalar': 0.0, 'recovery_scalar': 0.0, 'hold_state': False, 'components': []})
    last_residue = None
    last_flow_bias = 0.0
    last_continuation_mismatch = 0.0
    prev_phi = None
    pending_phi_continued = None
    prev_phi_oriented = None
    mismatch_series = []
    last_omega = np.zeros(4)
    phi_inductive = None

    # 💡 Persistence Fix: Load from memory.native_state if available
    trace_raw = memory.native_state.get("trace_segments", [])
    phase_continuation.trace_segments = [np.array(s) for s in trace_raw]
    history_raw = memory.native_state.get("history", [])
    phase_continuation.history = [np.array(h) for h in history_raw]

    prev_phi_oriented_raw = memory.native_state.get("prev_phi_oriented")
    if prev_phi_oriented_raw is not None: prev_phi_oriented = np.array(prev_phi_oriented_raw)
    
    pending_phi_continued_raw = memory.native_state.get("pending_phi_continued")
    if pending_phi_continued_raw is not None: pending_phi_continued = np.array(pending_phi_continued_raw)

    transformer = InductiveTransformerLayer.from_dict(
        memory.native_state.get("transformer"), channels=8
    )

    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    os.makedirs('logs', exist_ok=True)
    feedback_trace_path = f"logs/feedback_trace_{run_id}.jsonl"
    
    with open(feedback_trace_path, "a", encoding="utf-8") as f_log:
        for t in range(num_frames):
            if input_signals is not None:
                raw_input = input_signals[t]
                if isinstance(raw_input, (np.ndarray, list)):
                    input_signal_actual = float(np.mean(raw_input))
                    scope_input_actual = np.asarray(raw_input)
                else:
                    input_signal_actual = float(raw_input)
                    scope_input_actual = input_signal_actual
            else:
                input_signal_actual = np.sin(t * 0.1)
                scope_input_actual = input_signal_actual

            input_signal = input_signal_actual if connected else 0.0
            scope_input = scope_input_actual if connected else np.zeros_like(scope_input_actual)

            base_bias = feedback.update(last_state, last_residue) if fb_config["feedback"]["enabled"] else 1.0
            r_bias = residue_bias(last_residue)
            control_pattern = base_bias * r_bias
            flow_feedback_gain = float(fb_config.get('feedback', {}).get('flow_feedback_gain', 0.2))
            control_pattern = control_pattern * (1.0 + flow_feedback_gain * last_flow_bias)
            smooth_mismatch = np.mean(mismatch_series[-5:]) if len(mismatch_series) > 0 else 0.0
            control_pattern *= (1.0 - 0.02 * smooth_mismatch)
            min_b, max_b = float(fb_config.get('feedback', {}).get('min_bias', 0.5)), float(fb_config.get('feedback', {}).get('max_bias', 2.0))
            control_pattern = float(np.clip(control_pattern, min_b, max_b))

            engine_steps = int(fb_config.get('feedback', {}).get('engine_steps_per_frame', 20))
            engine_mean = engine.evolve(input_signal, control_pattern, steps=engine_steps)
            node_outputs = engine.get_node_outputs()
            
            # 💡 CRITICAL FIX: Ensure scope is only updated once and with the correct source
            if connected:
                if isinstance(scope_input_actual, np.ndarray):
                    scope_data_src = scope.update(scope_input_actual)
                else:
                    scope_data_src = scope.update(node_outputs)
            else:
                # When disconnected, the platform relies purely on internal node dynamics
                scope_data_src = scope.update(node_outputs)

            scope_data_actual = scope_data_src
            scope_data_internal = scope_data_src

            signal_x = compute_x_channel(scope_data_internal['W_local'], scope_data_internal['W_global'])
            
            # phi_actual represents the CURRENT state of the source we are tracking
            phi_actual = compute_phase_vector(scope_data_actual['W_local'], scope_data_actual['C'], scope_data_actual['E'], scope_data_actual['V'])
            op_star, op_cost = select_operator(phi_actual, prev_phi)
            phi_oriented_actual = apply_operator(phi_actual, op_star)

            if pending_phi_continued is not None:
                raw_mismatch = float(phase_mismatch(pending_phi_continued, phi_oriented_actual))
            else:
                raw_mismatch = 0.0

            # Refraction Update
            if refraction_enabled and connected and teacher_theta is not None:
                if connected_state == "train":
                    refraction.update_train(teacher_theta[t], phi_oriented_actual, np.zeros(8), transformer.omega, signal_x)
                else:
                    refraction.update_adaptive(teacher_theta[t], phi_oriented_actual, signal_x)

            # 💡 NEW: Handle internal refraction evolution (drift extrapolation)
            if refraction_enabled:
                refraction.step(connected=connected)

            # Diagnostics defaults for Patch 33
            diag_cl = {
                "closed_loop_w_trace": float(w_trace),
                "closed_loop_w_inductive": float(w_inductive),
                "closed_loop_w_anchor": float(w_anchor),
                "closed_loop_damping": float(damping),
                "trace_vec_available": False,
                "trace_vec_norm": 0.0,
                "blend_norm": 0.0,
                "blend_cosine_to_prev": 0.0,
                "blend_cosine_to_trace": 0.0,
                "blend_cosine_to_inductive": 0.0
            }

            # Patch 32/33: Closed-Loop Continuation
            if connected:
                phi_for_internal = phi_oriented_actual
            else:
                base = pending_phi_continued if pending_phi_continued is not None else phi_oriented_actual

                # Trace feedback (directional, not absolute)
                trace_vec = phase_continuation.trace_feedback_vector()

                # Inductive bias (already reference-aligned)
                inductive_vec = phi_inductive

                # Weak global anchor (prevents full drift of reference)
                anchor_vec = prev_phi_oriented if prev_phi_oriented is not None else base

                # Blend weights (Patch 33 tuning)
                w_base_val = 1.0
                
                diag_cl["trace_vec_available"] = trace_vec is not None
                if trace_vec is not None:
                    diag_cl["trace_vec_norm"] = float(np.linalg.norm(trace_vec))

                blended = w_base_val * base
                if trace_vec is not None:
                    blended += w_trace * trace_vec
                if inductive_vec is not None:
                    blended += w_inductive * inductive_vec
                blended += w_anchor * anchor_vec

                diag_cl["blend_norm"] = float(np.linalg.norm(blended))
                phi_for_internal = blended / (diag_cl["blend_norm"] + 1e-9)

                def cosine_sim(v1, v2):
                    if v1 is None or v2 is None: return 0.0
                    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                    if n1 < 1e-9 or n2 < 1e-9: return 0.0
                    return float(np.dot(v1, v2) / (n1 * n2))

                diag_cl["blend_cosine_to_prev"] = cosine_sim(phi_for_internal, prev_phi_oriented)
                diag_cl["blend_cosine_to_trace"] = cosine_sim(phi_for_internal, trace_vec)
                diag_cl["blend_cosine_to_inductive"] = cosine_sim(phi_for_internal, inductive_vec)

                # Optional damping to prevent runaway drift
                phi_for_internal = damping * phi_for_internal + (1.0 - damping) * (prev_phi_oriented if prev_phi_oriented is not None else phi_for_internal)
                phi_for_internal = phi_for_internal / (np.linalg.norm(phi_for_internal) + 1e-9)

            if refraction_enabled:
                anchor_input = refraction.unrefract(phi_for_internal)
            else:
                anchor_input = phi_for_internal
            
            phi_unrefracted_anchor = motion_anchor.update(
                anchor_input, 
                scope_data_internal['C'], 
                signal_x, 
                connected=connected,
                L=transformer.L 
            )
            anchor_state = motion_anchor.get_state()

            decision, failed_tests = phase_continuation.evaluate_survivability(phi_oriented_actual, raw_mismatch, op_star, signal_x)
            if not connected:
                decision = "reinforce"

            continuation_mismatch = raw_mismatch if connected else last_continuation_mismatch

            phi_unrefracted_inductive = transformer.update(phi_unrefracted_anchor, scope_data_internal['C'], signal_x, connected=connected)
            
            if refraction_enabled:
                phi_inductive = refraction.refract(phi_unrefracted_inductive)
            else:
                phi_inductive = phi_unrefracted_inductive

            phase_error = float(phase_mismatch(phi_oriented_actual, phi_inductive))
            freq_drift = float(np.linalg.norm(transformer.omega - last_omega))
            last_omega = transformer.omega.copy()

            if connected:
                active_groove, route_score = router.route(prev_phi_oriented, phi_oriented_actual, op_star)
            else:
                active_groove, route_score = None, 0.0
            groove_feedback_vec = router.active_feedback_vector()

            phi_continued = phase_continuation.continue_next(
                phi_inductive, decision,
                external_feedback_vec=groove_feedback_vec,
                inductive_feedback_vec=phi_inductive
            )
            
            mismatch_series.append(continuation_mismatch)

            if connected:
                phase_continuation.store_trace_segment(prev_phi_oriented, phi_oriented_actual, continuation_mismatch, decision)
                router.reinforce_active(prev_phi_oriented, phi_oriented_actual, op_star, decision, threshold=0.020)
                phase_continuation.reinforce_trace(phi_oriented_actual, continuation_mismatch, threshold=0.02)
            
            prev_phi_oriented = phi_oriented_actual.copy()
            op_pressure = operator_pressure(continuation_mismatch, last_continuation_mismatch, scope_data_actual['C'], scope_data_actual['E'], scope_data_actual['V'])
            last_flow_bias = float(np.tanh(np.mean(scope_data_actual['V'])))
            last_continuation_mismatch = continuation_mismatch
            pending_phi_continued = phi_continued.copy()

            full_hex = make_full_hex(scope_data_actual["W_local"], scope_data_actual["W_global"], scope_data_actual["W_meta"])
            signature_12, orientation_bias = project_to_12(scope_data_actual["W_local"], scope_data_actual["C"], scope_data_actual["E"], scope_data_actual["V"])
            signature_12 = apply_operator_pressure(signature_12, op_pressure)
            trace, state = v14.run_turn(signature_12, orientation_bias)
            memory, residue = qualify_and_commit(trace, state, memory, t, fb_config, metadata={"phi": phi_actual.tolist(), "hex": full_hex, "continuation_mismatch": continuation_mismatch, "op_pressure": op_pressure})
            
            geom = transformer.get_raw_geometry()
            ref_diag = refraction.get_diagnostics()
            
            log_entry = {
                "t": t, "input_signal": float(input_signal_actual), "control_pattern": float(control_pattern),
                "residue_committed": bool(residue.is_committed), "hex": full_hex,
                "phi_current": phi_actual.tolist(), "phi_continued": phi_continued.tolist(),
                "continuation_mismatch": continuation_mismatch, "active_groove_id": router.active_groove_id,
                "survivability_decision": decision, "signal_x": signal_x, "phase_error": phase_error,
                "teacher_theta": geom["teacher_theta"], "student_theta": geom["student_theta"],
                "connected": connected, "connected_state": connected_state,
                "refraction_enabled": bool(refraction_enabled),
                "adaptive_delta_hat": ref_diag["adaptive_delta_hat"]
            }
            log_entry.update(diag_cl) # Patch 33 Diagnostics
            f_log.write(json.dumps(log_entry) + "\n")

            last_state = state
            last_residue = residue
            prev_phi = phi_actual.copy()

    memory.successful_traversals = int(phase_continuation.successful_traversals)
    memory.groove_data = router.to_dict()
    memory.recursive_anchor_data = motion_anchor.to_dict()
    
    # 💡 Persistence Fix: Save to memory.native_state
    memory.native_state["transformer"] = transformer.to_dict()
    memory.native_state["refraction"] = refraction.to_dict()
    memory.native_state["trace_segments"] = [s.tolist() for s in phase_continuation.trace_segments]
    memory.native_state["history"] = [h.tolist() for h in phase_continuation.history]
    memory.native_state["prev_phi_oriented"] = prev_phi_oriented.tolist() if prev_phi_oriented is not None else None
    memory.native_state["pending_phi_continued"] = pending_phi_continued.tolist() if pending_phi_continued is not None else None
    
    save_memory_state(memory_path, memory)
    
    return {
        "run_id": run_id, "frames": num_frames, "memory_path": memory_path,
        "feedback_trace_path": feedback_trace_path,
        "refraction_diagnostics": refraction.get_diagnostics()
    }

if __name__ == "__main__":
    run_platform()
