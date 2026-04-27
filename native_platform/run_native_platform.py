
import os
import json
import numpy as np
from datetime import datetime

from .engine_bridge import EngineBridge
from .signalscope_core import SignalScope
from .hex_state import make_full_hex
from .wheel12_projection import project_to_12
from .v14_bridge import V14Bridge
from .residue_imprinter import qualify_and_commit
from .feedback_adapter import FeedbackAdapter
from .residue_feedback import residue_bias

from core.memory_layer import load_memory_state, save_memory_state

def run_platform(num_frames=100, num_nodes=100):
    print("🚀 Initializing Native Wave-Residue Platform...")
    
    # 1. Initialize Components
    engine = EngineBridge(num_nodes=num_nodes)
    scope = SignalScope()
    v14 = V14Bridge()
    memory = load_memory_state("sessions/native_memory.json")
    
    # Load Feedback Config
    fb_config_path = "native_platform/feedback_config.json"
    if os.path.exists(fb_config_path):
        with open(fb_config_path, 'r') as f:
            fb_config = json.load(f)
    else:
        fb_config = {"feedback": {"enabled": False}}
    
    feedback = FeedbackAdapter(fb_config)
    
    # Initial states for feedback
    last_state = type('obj', (object,), {'caution_scalar': 0.0, 'recovery_scalar': 0.0, 'hold_state': False, 'components': []})
    last_residue = None

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('logs', exist_ok=True)
    feedback_trace_path = f"logs/feedback_trace_{run_id}.jsonl"
    
    print(f"Starting loop for {num_frames} frames...")
    
    for t in range(num_frames):
        # A. Signal Generation (Example: sine wave)
        input_signal = np.sin(t * 0.1)
        
        # New: Compute dynamic feedback bias (Control Pattern)
        # Combine adapter bias and residue-level reward
        base_bias = feedback.update(last_state, last_residue) if fb_config["feedback"]["enabled"] else 1.0
        r_bias = residue_bias(last_residue)
        control_pattern = base_bias * r_bias

        # B. Step Engine with dynamic control pattern
        # Patch 11: Multi-step evolution
        engine_steps = int(fb_config.get('feedback', {}).get('engine_steps_per_frame', 20))
        engine_mean = engine.evolve(input_signal, control_pattern, steps=engine_steps)
        node_outputs = engine.get_node_outputs()
        
        # C. Update SignalScope
        scope_data = scope.update(node_outputs)
        
        # Patch 11: Directional Feedback (react to trajectory flow)
        flow_feedback_gain = float(fb_config.get('feedback', {}).get('flow_feedback_gain', 0.2))
        flow_bias = float(np.tanh(np.mean(scope_data['V'])))
        control_pattern = control_pattern * (1.0 + flow_feedback_gain * flow_bias)

        # D. Hex Encoding
        full_hex = make_full_hex(scope_data["W_local"], scope_data["W_global"], scope_data["W_meta"])
        
        # E. 12-Wheel Projection
        signature_12, orientation_bias = project_to_12(
            scope_data["W_local"], 
            scope_data["C"], 
            scope_data["E"], 
            scope_data["V"]
        )
        
        # F. Run SBLLM v14 Reasoning
        trace, state = v14.run_turn(signature_12, orientation_bias)
        
        # G. Imprint Residue
        memory, residue = qualify_and_commit(trace, state, memory, t, v14.config)
        
        # H. Log Progress
        if residue.is_committed:
            status = "IMPRINTED"
        else:
            status = "SKIPPED"
            
        # Logging to feedback_trace.jsonl (Patch 11 upgraded)
        log_entry = {
            "t": t,
            "input_signal": float(input_signal),
            "control_pattern": float(control_pattern),
            "caution": float(state.caution_scalar),
            "recovery": float(state.recovery_scalar),
            "residue_committed": bool(residue.is_committed),
            "bias": float(r_bias),
            "hex": full_hex,
            "C": float(scope_data['C']),
            "E": float(scope_data['E']),
            "V": scope_data['V'].tolist(),
            "engine_mean": float(engine_mean),
            "engine_steps": int(engine_steps)
        }
        with open(feedback_trace_path, "a", encoding="utf-8") as f_log:
            f_log.write(json.dumps(log_entry) + "\n")

        if t % 10 == 0:
            print(f"Frame {t}: [{full_hex}] C={scope_data['C']:.2f} Control={control_pattern:.2f} -> {status}")

        # Update last states for next feedback cycle
        last_state = state
        last_residue = residue

    # 2. Finalize
    save_memory_state("sessions/native_memory.json", memory)
    print("✅ Run Complete. Native memory updated.")

if __name__ == "__main__":
    run_platform()
