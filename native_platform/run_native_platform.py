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

from core.memory_layer import load_memory_state, save_memory_state

def run_platform(num_frames=100, num_nodes=100, engine_steps_per_frame=None, feedback_enabled=None, run_id=None, input_signals=None):
    print("🚀 Initializing Native Wave-Residue Platform (Continuation Mode)...")
    
    # Check input signal length
    if input_signals is not None:
        num_frames = len(input_signals)

    # 1. Initialize Components
    engine = EngineBridge(num_nodes=num_nodes)
    scope = SignalScope()
    v14 = V14Bridge()
    
    # Patch 17: Phase Continuation with deeper history
    phase_continuation = ResiduePhaseContinuation(history_size=64, trace_size=128)
    
    memory_path = "sessions/native_memory.json"
    memory = load_memory_state(memory_path)
    
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
    
    # Patch 17: Prior continuation for next-frame accuracy
    pending_phi_continued = None

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
                input_signal = input_signals[t]
            else:
                input_signal = np.sin(t * 0.1)
            
            # Feedback calculation using PREVIOUS frame metrics
            base_bias = feedback.update(last_state, last_residue) if fb_config["feedback"]["enabled"] else 1.0
            r_bias = residue_bias(last_residue)
            
            # Base control pattern
            control_pattern = base_bias * r_bias
            
            # Inject flow bias and continuation mismatch from previous turn
            flow_feedback_gain = float(fb_config.get('feedback', {}).get('flow_feedback_gain', 0.2))
            control_pattern = control_pattern * (1.0 + flow_feedback_gain * last_flow_bias)
            control_pattern *= (1.0 - 0.02 * last_continuation_mismatch)

            # Clamp control pattern
            min_b = float(fb_config.get('feedback', {}).get('min_bias', 0.5))
            max_b = float(fb_config.get('feedback', {}).get('max_bias', 2.0))
            control_pattern = float(np.clip(control_pattern, min_b, max_b))

            # B. Step Engine
            engine_steps = int(fb_config.get('feedback', {}).get('engine_steps_per_frame', 20))
            engine_mean = engine.evolve(input_signal, control_pattern, steps=engine_steps)
            node_outputs = engine.get_node_outputs()
            
            # C. Update SignalScope
            scope_data = scope.update(node_outputs)
            
            # Phase Space & Continuation
            phi_current = compute_phase_vector(
                scope_data['W_local'],
                scope_data['C'],
                scope_data['E'],
                scope_data['V']
            )
            
            # Patch 17: Real continuation alignment error
            if pending_phi_continued is not None:
                continuation_mismatch_next = float(phase_mismatch(pending_phi_continued, phi_current))
            else:
                continuation_mismatch_next = 0.0

            # Generate internal continuation
            phi_continued = phase_continuation.continue_next(phi_current, last_mismatch=last_continuation_mismatch)
            continuation_mismatch = float(phase_mismatch(phi_continued, phi_current))

            # Reinforce trace groove if mismatch is low
            phase_continuation.reinforce_trace(phi_current, continuation_mismatch, threshold=0.02)

            # Map phase flow to operator pressure
            op_pressure = operator_pressure(continuation_mismatch, last_continuation_mismatch, scope_data['C'], scope_data['E'], scope_data['V'])

            # Update for next frame
            last_flow_bias = float(np.tanh(np.mean(scope_data['V'])))
            last_continuation_mismatch = continuation_mismatch
            pending_phi_continued = phi_continued.copy()

            # D. Hex Encoding
            full_hex = make_full_hex(scope_data["W_local"], scope_data["W_global"], scope_data["W_meta"])
            
            # E. 12-Wheel Projection
            signature_12, orientation_bias = project_to_12(
                scope_data["W_local"], 
                scope_data["C"], 
                scope_data["E"], 
                scope_data["V"]
            )
            
            # Apply operator pressure to signature (legal input-space shaping)
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
                "traversal_count": phase_continuation.traversal_count,
                "op_pressure": op_pressure
            }
            f_log.write(json.dumps(log_entry) + "\n")

            if t % 10 == 0:
                print(f"Frame {t}: [{full_hex}] C={scope_data['C']:.2f} Mismatch={continuation_mismatch:.4f} Groove={len(phase_continuation.trace_buffer)} -> {status}")

            last_state = state
            last_residue = residue

    # 2. Finalize
    phase_continuation.mark_traversal_complete()
    save_memory_state(memory_path, memory)
    print(f"✅ Run Complete. Trace: logs/feedback_trace_{run_id}.jsonl")
    
    return {
        "run_id": run_id,
        "frames": num_frames,
        "memory_path": memory_path,
        "feedback_trace_path": feedback_trace_path
    }

if __name__ == "__main__":
    run_platform()
