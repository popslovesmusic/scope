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

from core.memory_layer import load_memory_state, save_memory_state

def run_platform(num_frames=100, num_nodes=100, engine_steps_per_frame=None, feedback_enabled=None, run_id=None, input_signals=None, memory_path="sessions/native_memory.json", connected=True, connected_state="train"):
    print("🚀 Initializing Native Wave-Residue Platform (Continuation Mode)...")
    
    # Check input signal length
    if input_signals is not None:
        num_frames = len(input_signals)

    # 1. Initialize Components
    engine = EngineBridge(num_nodes=num_nodes)
    scope = SignalScope()
    v14 = V14Bridge()
    
    memory = load_memory_state(memory_path)
    
    # Patch 17/20/23: Phase Continuation with survivability gating
    phase_continuation = ResiduePhaseContinuation(
        history_size=64, 
        trace_size=128, 
        successful_traversals=memory.successful_traversals
    )
    
    # Patch 22: Initialize GrooveRouter from memory
    router = GrooveRouter.from_dict(memory.groove_data)
    
    # Patch 24: Inductive Transformer Layer
    transformer = InductiveTransformerLayer(channels=8)
    
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
    
    # Initial states for feedback
    last_state = type('obj', (object,), {'caution_scalar': 0.0, 'recovery_scalar': 0.0, 'hold_state': False, 'components': []})
    last_residue = None
    
    # Track metrics for engine injection and transition
    last_flow_bias = 0.0
    last_continuation_mismatch = 0.0
    
    # Prior states for selections
    prev_phi = None
    pending_phi_continued = None
    prev_phi_oriented = None
    mismatch_series = []
    
    # Patch 24: Track frequency drift
    last_omega = np.zeros(8)

    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    os.makedirs('logs', exist_ok=True)
    os.makedirs('sessions', exist_ok=True)
    feedback_trace_path = f"logs/feedback_trace_{run_id}.jsonl"
    
    print(f"Starting loop for {num_frames} frames (Run ID: {run_id})...")
    
    # Open log file once for buffered writing
    with open(feedback_trace_path, "a", encoding="utf-8") as f_log:
        for t in range(num_frames):
            # A. Signal Generation
            if input_signals is not None:
                raw_input = input_signals[t]
                # If raw_input is a vector (EEG features), use its mean for the scalar engine input
                if isinstance(raw_input, (np.ndarray, list)):
                    input_signal = float(np.mean(raw_input))
                    scope_input = np.asarray(raw_input)
                else:
                    input_signal = float(raw_input)
                    scope_input = input_signal
            else:
                input_signal = np.sin(t * 0.1)
                scope_input = input_signal
            
            # Feedback calculation using PREVIOUS frame metrics
            base_bias = feedback.update(last_state, last_residue) if fb_config["feedback"]["enabled"] else 1.0
            r_bias = residue_bias(last_residue)
            
            # Base control pattern
            control_pattern = base_bias * r_bias
            
            # Inject flow bias and continuation mismatch from previous turn
            flow_feedback_gain = float(fb_config.get('feedback', {}).get('flow_feedback_gain', 0.2))
            control_pattern = control_pattern * (1.0 + flow_feedback_gain * last_flow_bias)
            
            # Patch 23: use smoothed mismatch for control feedback
            smooth_mismatch = np.mean(mismatch_series[-5:]) if len(mismatch_series) > 0 else 0.0
            control_pattern *= (1.0 - 0.02 * smooth_mismatch)

            # Clamp control pattern
            min_b = float(fb_config.get('feedback', {}).get('min_bias', 0.5))
            max_b = float(fb_config.get('feedback', {}).get('max_bias', 2.0))
            control_pattern = float(np.clip(control_pattern, min_b, max_b))

            # B. Step Engine
            engine_steps = int(fb_config.get('feedback', {}).get('engine_steps_per_frame', 20))
            engine_mean = engine.evolve(input_signal, control_pattern, steps=engine_steps)
            node_outputs = engine.get_node_outputs()
            
            # C. Update SignalScope
            if isinstance(scope_input, np.ndarray):
                scope_data = scope.update(scope_input)
            else:
                scope_data = scope.update(node_outputs)
            
            # Patch 23: Compute X channel (cross-view consistency)
            signal_x = compute_x_channel(scope_data['W_local'], scope_data['W_global'])
            
            # Phase Space & Continuation
            phi_current = compute_phase_vector(
                scope_data['W_local'],
                scope_data['C'],
                scope_data['E'],
                scope_data['V']
            )
            
            # Patch 18: Operator Selection for Local Reference -(i)
            op_star, op_cost = select_operator(phi_current, prev_phi)
            i_local = apply_operator(phi_current, op_star)
            phi_oriented = i_local

            # Patch 17/20: Real continuation alignment error
            # This mismatch is used for DECISION making
            if pending_phi_continued is not None:
                raw_mismatch = float(phase_mismatch(pending_phi_continued, phi_oriented))
            else:
                raw_mismatch = 0.0

            # Patch 23: Survivability Gating (uses raw mismatch for gating)
            decision, failed_tests = phase_continuation.evaluate_survivability(
                phi_oriented, 
                raw_mismatch, 
                op_star, 
                signal_x
            )
            
            # The EFFECTIVE mismatch used for reinforcement trend is filtered
            continuation_mismatch = raw_mismatch if decision != "reject" else last_continuation_mismatch

            # Patch 24: Inductive Transformer Layer Update
            # Connected state can be overridden per run
            phi_inductive = transformer.update(phi_oriented, scope_data['C'], signal_x, connected=connected)
            
            # New Metric: phase_error (between oriented input and inductive prediction)
            phase_error = float(phase_mismatch(phi_oriented, phi_inductive))
            
            # New Metric: frequency_drift
            freq_drift = float(np.linalg.norm(transformer.omega - last_omega))
            last_omega = transformer.omega.copy()

            # Patch 22: Groove Routing
            active_groove, route_score = router.route(prev_phi_oriented, phi_oriented, op_star)
            groove_feedback_vec = router.active_feedback_vector()

            # Patch 22/23/24: Generate internal continuation with external groove + inductive feedback
            phi_continued = phase_continuation.continue_next(
                phi_oriented, 
                decision,
                external_feedback_vec=groove_feedback_vec,
                inductive_feedback_vec=phi_inductive
            )
            
            mismatch_series.append(continuation_mismatch)

            # Patch 20/23: Store trace segment feedback (short-term) gated by decision
            phase_continuation.store_trace_segment(prev_phi_oriented, phi_oriented, continuation_mismatch, decision)
            
            # Patch 22/23: Reinforce active groove (long-term identity) gated by decision
            router.reinforce_active(prev_phi_oriented, phi_oriented, op_star, decision, threshold=0.020)
            
            prev_phi_oriented = phi_oriented.copy()

            # Reinforce trace groove if mismatch is low (legacy buffer)
            phase_continuation.reinforce_trace(phi_oriented, continuation_mismatch, threshold=0.02)

            # Map phase flow to operator pressure
            op_pressure = operator_pressure(continuation_mismatch, last_continuation_mismatch, scope_data['C'], scope_data['E'], scope_data['V'])

            # Update for next frame
            last_flow_bias = float(np.tanh(np.mean(scope_data['V'])))
            last_continuation_mismatch = continuation_mismatch
            pending_phi_continued = phi_continued.copy()
            
            continuation_mismatch_next = continuation_mismatch

            # D. Hex Encoding
            full_hex = make_full_hex(scope_data["W_local"], scope_data["W_global"], scope_data["W_meta"])
            
            # E. 12-Wheel Projection
            signature_12, orientation_bias = project_to_12(
                scope_data["W_local"], 
                scope_data["C"], 
                scope_data["E"], 
                scope_data["V"]
            )
            signature_12 = apply_operator_pressure(signature_12, op_pressure)
            
            # F. Run SBLLM v14 Reasoning
            trace, state = v14.run_turn(signature_12, orientation_bias)
            
            # G. Imprint Residue
            meta_dict = {
                "phi": phi_current.tolist(),
                "hex": full_hex,
                "continuation_mismatch": continuation_mismatch,
                "op_pressure": op_pressure
            }
            memory, residue = qualify_and_commit(trace, state, memory, t, fb_config, metadata=meta_dict)
            
            # H. Log Progress
            if residue.is_committed:
                status = "IMPRINTED"
            else:
                status = "SKIPPED"
                
            # Patch 25: Enhanced Geometry Logging
            geom = transformer.get_raw_geometry()
            
            log_entry = {
                "t": t,
                "input_signal": float(input_signal),
                "control_pattern": float(control_pattern),
                "caution": float(state.caution_scalar),
                "recovery": float(state.recovery_scalar),
                "residue_committed": bool(residue.is_committed),
                "residue_reject_reasons": getattr(residue, 'reject_reasons', []),
                "residue_score": float(getattr(residue, 'stability_score', 0.0)),
                "bias": float(r_bias),
                "hex": full_hex,
                "C": float(scope_data['C']),
                "E": float(scope_data['E']),
                "V": scope_data['V'].tolist(),
                "phi_current": phi_current.tolist(),
                "phi_continued": phi_continued.tolist(),
                "continuation_mismatch": continuation_mismatch,
                "continuation_mismatch_next": continuation_mismatch_next,
                "trace_groove_size": len(phase_continuation.trace_buffer),
                "trace_segment_count": len(phase_continuation.trace_segments),
                "trace_feedback_gain": float(phase_continuation.groove_gain()),
                "successful_traversals": int(phase_continuation.successful_traversals),
                "traversal_count": phase_continuation.traversal_count,
                "op_pressure": op_pressure,
                "active_groove_id": router.active_groove_id,
                "groove_count": len(router.grooves),
                "groove_score": float(route_score),
                "operator_star": op_star,
                "survivability_decision": decision,
                "failed_tests": failed_tests,
                "signal_x": signal_x,
                "consistency_level": get_consistency_level(signal_x),
                # Patch 24/25 Geometry
                "phase_error": phase_error,
                "frequency_drift": freq_drift,
                "inductive_L": transformer.L.tolist(),
                "teacher_theta": geom["teacher_theta"],
                "student_theta": geom["student_theta"],
                "teacher_amp": geom["teacher_amp"],
                "student_amp": geom["student_amp"],
                "teacher_omega": geom["teacher_omega"],
                "student_omega": geom["student_omega"],
                "connected": connected,
                "connected_state": connected_state
            }
            f_log.write(json.dumps(log_entry) + "\n")

            if t % 10 == 0:
                gid = router.active_groove_id or "none"
                print(f"Frame {t}: [{full_hex}] C={scope_data['C']:.2f} X={signal_x:.2f} Err={phase_error:.4f} G={gid} ({decision}) -> {status}")

            last_state = state
            last_residue = residue
            prev_phi = phi_current.copy()

    # 2. Finalize
    if len(mismatch_series) > 0:
        q = len(mismatch_series) // 4
        if q > 0:
            first_mean = np.mean(mismatch_series[:q])
            last_mean = np.mean(mismatch_series[-q:])
            if last_mean <= first_mean:
                phase_continuation.mark_successful_traversal()
            else:
                phase_continuation.mark_failed_traversal()
        else:
            phase_continuation.mark_successful_traversal()

    phase_continuation.mark_traversal_complete()
    memory.successful_traversals = int(phase_continuation.successful_traversals)
    memory.groove_data = router.to_dict()
    
    save_memory_state(memory_path, memory)
    print(f"✅ Run Complete. Trace: logs/feedback_trace_{run_id}.jsonl")
    
    return {
        "run_id": run_id,
        "frames": num_frames,
        "memory_path": memory_path,
        "feedback_trace_path": feedback_trace_path,
        "groove_summary": router.summary()
    }

if __name__ == "__main__":
    run_platform()
