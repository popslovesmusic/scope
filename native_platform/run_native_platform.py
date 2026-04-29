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

def run_platform(num_frames=100, num_nodes=100, engine_steps_per_frame=None, feedback_enabled=None, run_id=None, input_signals=None, memory_path="sessions/native_memory.json", connected=True, connected_state="train", teacher_theta=None):
    print(f"🚀 Initializing Native Wave-Residue Platform (Mode: {connected_state.upper()})...")
    
    # Check input signal length
    if input_signals is not None:
        num_frames = len(input_signals)

    # 1. Initialize Components
    engine = EngineBridge(num_nodes=num_nodes)
    scope = SignalScope()
    v14 = V14Bridge()
    
    memory = load_memory_state(memory_path)
    
    # Components
    phase_continuation = ResiduePhaseContinuation(
        history_size=64, 
        trace_size=128, 
        successful_traversals=memory.successful_traversals
    )
    router = GrooveRouter.from_dict(memory.groove_data)
    transformer = InductiveTransformerLayer(channels=8)
    motion_anchor = RecursiveMotionAnchor.from_dict(memory.recursive_anchor_data)
    
    # Patch 29: Phase Refraction Layer
    refraction = PhaseRefractionLayer(channels=8)
    
    # Load Feedback Config
    fb_config_path = "native_platform/feedback_config.json"
    if os.path.exists(fb_config_path):
        with open(fb_config_path, 'r') as f:
            fb_config = json.load(f)
    else:
        fb_config = {"feedback": {"enabled": False}}
    
    # Apply CLI/Interface Overrides
    if engine_steps_per_frame is not None:
        fb_config.setdefault("feedback", {})["engine_steps_per_frame"] = engine_steps_per_frame
    if feedback_enabled is not None:
        fb_config.setdefault("feedback", {})["enabled"] = feedback_enabled

    feedback = FeedbackAdapter(fb_config)
    
    # States
    last_state = type('obj', (object,), {'caution_scalar': 0.0, 'recovery_scalar': 0.0, 'hold_state': False, 'components': []})
    last_residue = None
    last_flow_bias = 0.0
    last_continuation_mismatch = 0.0
    prev_phi = None
    pending_phi_continued = None
    prev_phi_oriented = None
    mismatch_series = []
    last_omega = np.zeros(8)

    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    os.makedirs('logs', exist_ok=True)
    os.makedirs('sessions', exist_ok=True)
    feedback_trace_path = f"logs/feedback_trace_{run_id}.jsonl"
    
    print(f"Starting loop for {num_frames} frames (Run ID: {run_id})...")
    
    with open(feedback_trace_path, "a", encoding="utf-8") as f_log:
        for t in range(num_frames):
            # A. Signal Generation
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

            # Leakage Control
            input_signal = input_signal_actual if connected else 0.0
            scope_input = scope_input_actual if connected else np.zeros_like(scope_input_actual)

            # Feedback
            base_bias = feedback.update(last_state, last_residue) if fb_config["feedback"]["enabled"] else 1.0
            r_bias = residue_bias(last_residue)
            control_pattern = base_bias * r_bias
            flow_feedback_gain = float(fb_config.get('feedback', {}).get('flow_feedback_gain', 0.2))
            control_pattern = control_pattern * (1.0 + flow_feedback_gain * last_flow_bias)
            smooth_mismatch = np.mean(mismatch_series[-5:]) if len(mismatch_series) > 0 else 0.0
            control_pattern *= (1.0 - 0.02 * smooth_mismatch)
            min_b, max_b = float(fb_config.get('feedback', {}).get('min_bias', 0.5)), float(fb_config.get('feedback', {}).get('max_bias', 2.0))
            control_pattern = float(np.clip(control_pattern, min_b, max_b))

            # B. Step Engine
            engine_steps = int(fb_config.get('feedback', {}).get('engine_steps_per_frame', 20))
            engine_mean = engine.evolve(input_signal, control_pattern, steps=engine_steps)
            node_outputs = engine.get_node_outputs()
            
            # C. Update SignalScope
            if isinstance(scope_input_actual, np.ndarray):
                scope_data_actual = scope.update(scope_input_actual)
            else:
                scope_data_actual = scope.update(node_outputs)
            
            if not connected:
                scope_data_internal = scope.update(np.zeros_like(node_outputs))
            else:
                scope_data_internal = scope_data_actual

            signal_x = compute_x_channel(scope_data_internal['W_local'], scope_data_internal['W_global'])
            
            # Phase Space
            phi_actual = compute_phase_vector(scope_data_actual['W_local'], scope_data_actual['C'], scope_data_actual['E'], scope_data_actual['V'])
            op_star, op_cost = select_operator(phi_actual, prev_phi)
            phi_oriented_actual = apply_operator(phi_actual, op_star)

            if pending_phi_continued is not None:
                raw_mismatch = float(phase_mismatch(pending_phi_continued, phi_oriented_actual))
            else:
                raw_mismatch = 0.0

            # Patch 29: Refraction Tracking (Train Only)
            if connected and teacher_theta is not None:
                refraction.update_train(
                    teacher_theta[t], 
                    phi_oriented_actual, 
                    np.zeros(8), 
                    transformer.omega, 
                    signal_x
                )
            
            # Patch 27: Recursive Motion Anchor Update
            phi_motion_anchor = motion_anchor.update(
                phi_oriented_actual, 
                scope_data_actual['C'], 
                signal_x, 
                connected=connected,
                L=transformer.L 
            )
            anchor_state = motion_anchor.get_state()

            # Survivability Gating
            if connected:
                decision, failed_tests = phase_continuation.evaluate_survivability(phi_oriented_actual, raw_mismatch, op_star, signal_x)
            else:
                decision, failed_tests = "hold", ["disconnected_protocol"]
            
            continuation_mismatch = raw_mismatch if connected else last_continuation_mismatch

            # Patch 24: Inductive Transformer Layer Update
            phi_inductive = transformer.update(phi_motion_anchor, scope_data_internal['C'], signal_x, connected=connected)
            phase_error = float(phase_mismatch(phi_oriented_actual, phi_inductive))
            freq_drift = float(np.linalg.norm(transformer.omega - last_omega))
            last_omega = transformer.omega.copy()

            # Groove Routing
            if connected:
                active_groove, route_score = router.route(prev_phi_oriented, phi_oriented_actual, op_star)
            else:
                active_groove, route_score = None, 0.0
            groove_feedback_vec = router.active_feedback_vector()

            # Patch 29: Apply Refraction Compensation during disconnect
            if not connected:
                # Evolve in unrefracted frame
                phi_unrefracted = refraction.unrefract(phi_motion_anchor)
                phi_for_continuation = refraction.refract(phi_unrefracted)
            else:
                phi_for_continuation = phi_oriented_actual

            # Generate continuation
            phi_continued = phase_continuation.continue_next(
                phi_for_continuation, 
                decision,
                external_feedback_vec=groove_feedback_vec,
                inductive_feedback_vec=phi_inductive
            )
            
            mismatch_series.append(continuation_mismatch)

            # Reinforcement
            if connected:
                phase_continuation.store_trace_segment(prev_phi_oriented, phi_oriented_actual, continuation_mismatch, decision)
                router.reinforce_active(prev_phi_oriented, phi_oriented_actual, op_star, decision, threshold=0.020)
                phase_continuation.reinforce_trace(phi_oriented_actual, continuation_mismatch, threshold=0.02)
            
            prev_phi_oriented = phi_oriented_actual.copy()
            op_pressure = operator_pressure(continuation_mismatch, last_continuation_mismatch, scope_data_actual['C'], scope_data_actual['E'], scope_data_actual['V'])
            last_flow_bias = float(np.tanh(np.mean(scope_data_actual['V'])))
            last_continuation_mismatch = continuation_mismatch
            pending_phi_continued = phi_continued.copy()

            # D. Hex Encoding
            full_hex = make_full_hex(scope_data_actual["W_local"], scope_data_actual["W_global"], scope_data_actual["W_meta"])
            signature_12, orientation_bias = project_to_12(scope_data_actual["W_local"], scope_data_actual["C"], scope_data_actual["E"], scope_data_actual["V"])
            signature_12 = apply_operator_pressure(signature_12, op_pressure)
            
            # F. SBLLM Turn
            trace, state = v14.run_turn(signature_12, orientation_bias)
            memory, residue = qualify_and_commit(trace, state, memory, t, fb_config, metadata={"phi": phi_actual.tolist(), "hex": full_hex, "continuation_mismatch": continuation_mismatch, "op_pressure": op_pressure})
            
            # H. Log
            status = "IMPRINTED" if residue.is_committed else "SKIPPED"
            geom = transformer.get_raw_geometry()
            ref_diag = refraction.get_diagnostics()
            
            log_entry = {
                "t": t,
                "input_signal": float(input_signal_actual),
                "control_pattern": float(control_pattern),
                "caution": float(state.caution_scalar),
                "recovery": float(state.recovery_scalar),
                "residue_committed": bool(residue.is_committed),
                "hex": full_hex,
                "C": float(scope_data_actual['C']),
                "E": float(scope_data_actual['E']),
                "V": scope_data_actual['V'].tolist(),
                "phi_current": phi_actual.tolist(),
                "phi_continued": phi_continued.tolist(),
                "continuation_mismatch": continuation_mismatch,
                "active_groove_id": router.active_groove_id,
                "survivability_decision": decision,
                "signal_x": signal_x,
                "phase_error": phase_error,
                "frequency_drift": freq_drift,
                "teacher_theta": geom["teacher_theta"],
                "student_theta": geom["student_theta"],
                "connected": connected,
                "connected_state": connected_state,
                "anchor_confidence": anchor_state["confidence"],
                "refraction_confidence": ref_diag["refraction_confidence"],
                "refraction_variance": ref_diag["refraction_variance"],
                "medium_classification": ref_diag["medium_classification"],
                "delta_hat": ref_diag["delta_hat"]
            }
            f_log.write(json.dumps(log_entry) + "\n")

            if t % 10 == 0:
                gid = router.active_groove_id or "none"
                print(f"Frame {t}: [{full_hex}] C={scope_data_actual['C']:.2f} X={signal_x:.2f} Err={phase_error:.4f} G={gid} ({decision}) -> {status}")

            last_state = state
            last_residue = residue
            prev_phi = phi_actual.copy()

    # 2. Finalize
    if connected and len(mismatch_series) > 0:
        q = len(mismatch_series) // 4
        if q > 0:
            if np.mean(mismatch_series[-q:]) <= np.mean(mismatch_series[:q]):
                phase_continuation.mark_successful_traversal()
            else:
                phase_continuation.mark_failed_traversal()
        else:
            phase_continuation.mark_successful_traversal()

    phase_continuation.mark_traversal_complete()
    memory.successful_traversals = int(phase_continuation.successful_traversals)
    memory.groove_data = router.to_dict()
    memory.recursive_anchor_data = motion_anchor.to_dict()
    save_memory_state(memory_path, memory)
    print(f"✅ Run Complete. Trace: logs/feedback_trace_{run_id}.jsonl")
    
    return {
        "run_id": run_id,
        "frames": num_frames,
        "memory_path": memory_path,
        "feedback_trace_path": feedback_trace_path,
        "groove_summary": router.summary(),
        "refraction_diagnostics": refraction.get_diagnostics()
    }

if __name__ == "__main__":
    run_platform()
