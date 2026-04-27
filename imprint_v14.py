
import os
import json
import numpy as np
import soundfile as sf
import scipy.signal as signal
from signal_scope import SignalScope
from scopes.hex_state import make_full_hex, w_to_hex
from scopes.hex_trace import HexTraceFrame, append_hex_trace_jsonl
from scopes.hex_residue_codec import make_hex_residue_candidate, score_hex_stability

from core.signature_state import SignatureState
from core.reasoning_loop import run_reasoning
from core.memory_layer import (
    PersistentMemoryState,
    load_memory_state,
    save_memory_state,
    build_turn_residue,
    qualify_residue,
    apply_commit_gate_and_persistence,
    append_turn_residue
)

def extract_audio_features(frame, sr, unwrapped_phases, start_idx):
    # RMS Amplitude
    rms = np.sqrt(np.mean(frame**2))
    
    # Phase progression
    frame_phases = unwrapped_phases[start_idx:start_idx+len(frame)]
    phase_diff = np.mean(np.diff(frame_phases))
    
    # Spectral Centroid
    fft_data = np.abs(np.fft.rfft(frame))
    freqs = np.fft.rfftfreq(len(frame), 1/sr)
    if np.sum(fft_data) > 0:
        centroid = np.sum(freqs * fft_data) / np.sum(fft_data)
    else:
        centroid = 0
    
    return {
        "rms": float(rms),
        "phase_diff": float(phase_diff),
        "centroid": float(centroid / (sr/2))
    }

def project_to_12_wheel(features, size=12):
    """
    Project 3 features into 12 channels.
    """
    signed_field = np.zeros(size, dtype=float)
    
    # Feature 1: RMS -> Channels 0-3
    f1 = np.clip(features["rms"] * 5, 0, 1) # Scale RMS a bit
    for i in range(4):
        dist = abs(f1 - i/3.0)
        signed_field[i] = np.exp(-dist * 5) * f1
        
    # Feature 2: Phase -> Channels 4-7
    f2 = np.clip(abs(features["phase_diff"]) * 10, 0, 1)
    for i in range(4):
        dist = abs(f2 - i/3.0)
        signed_field[i+4] = np.exp(-dist * 5) * f2
        
    # Feature 3: Centroid -> Channels 8-11
    f3 = np.clip(features["centroid"], 0, 1)
    for i in range(4):
        dist = abs(f3 - i/3.0)
        signed_field[i+8] = np.exp(-dist * 5) * f3
        
    # Center the signed field to [-1, 1]
    signed_field = (signed_field * 2.0) - 1.0
    return signed_field

def run_imprinting(audio_path, config_path, memory_path="sessions/memory_state.json"):
    # 1. Setup
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Load Hex Config
    try:
        with open("scopes/scope_config.json", 'r') as f:
            hex_cfg = json.load(f).get("hex_encoding", {})
    except Exception:
        hex_cfg = {"enabled": False}

    # Override for imprinting: we want to capture structure even under pressure
    config["enable_hold_state"] = False
    config["enable_trace_salience"] = False # Reduce caution sensitivity for raw imprinting
    
    memory = load_memory_state(memory_path)
    audio, sr = sf.read(audio_path)
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
        
    analytic_signal = signal.hilbert(audio)
    unwrapped_phases = np.unwrap(np.angle(analytic_signal))
    
    frame_size = 1024
    hop_size = 1024 # Larger hop for efficiency in this demo
    frames = [audio[i:i+frame_size] for i in range(0, len(audio)-frame_size, hop_size)]
    
    from datetime import datetime
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    signature_size = int(config.get("signature_size", 12))
    mem_cfg = config.get("memory_layer", {})
    os.makedirs('logs', exist_ok=True)
    os.makedirs('sessions', exist_ok=True)
    
    residue_log_path = f"logs/residue_imprinting_{run_id}.jsonl"
    hex_trace_path = f"sessions/hex_trace_{run_id}.jsonl"
    # Use run-specific memory path if not explicitly provided
    if memory_path == "sessions/memory_state.json":
        memory_path = f"sessions/memory_state_{run_id}.json"
    
    scope = SignalScope()
    hex_buffer = [] # To detect motifs across frames

    print(f"Imprinting wave from {audio_path} ({len(frames)} frames)...")
    
    for i, frame in enumerate(frames):
        start_idx = i * hop_size
        features = extract_audio_features(frame, sr, unwrapped_phases, start_idx)
        
        # --- SignalScope Update ---
        raw_w = np.array([features["rms"]*5, abs(features["phase_diff"])*10, features["centroid"]])
        scope_data = scope.update(raw_w)
        local_obs = scope_data["local"]
        global_obs = scope_data["global"]
        meta_obs = scope_data["meta"]

        # --- Hex Generation ---
        full_hex = make_full_hex(local_obs["W"], global_obs["W"], meta_obs["W"])

        # 2. Project to 12-wheel
        signed_field = project_to_12_wheel(features, size=signature_size)
        
        # 3. Initialize State
        state = SignatureState(signature_size)
        state.signed_field = signed_field
        state.derive_amplitude_from_signed()
        state.input_trace.append({"frame": i, "features": features, "hex": full_hex})
        
        # 4. Run Reasoning Loop (Orientation, Corridor, etc.)
        # Inject memory biases from previous frames
        config["memory_enabled"] = True
        config["memory_operator_bias"] = memory.operator_bias
        config["memory_caution_baseline_shift"] = memory.caution_baseline_shift
        config["memory_operator_bias_strength"] = float(mem_cfg.get("operator_bias_strength", 0.08))
        config["memory_caution_baseline_strength"] = float(mem_cfg.get("caution_baseline_strength", 0.25))

        trace = run_reasoning(state, config)
        selected_op = trace[-1]["selected_operator"] if trace else "++"

        # --- Hex Trace Log ---
        hex_frame = HexTraceFrame(
            t=i,
            local=w_to_hex(local_obs["W"]),
            global_hex=w_to_hex(global_obs["W"]),
            meta=w_to_hex(meta_obs["W"]),
            full_code=full_hex,
            C=local_obs["C"],
            E_scope=local_obs["E_scope"],
            speed=local_obs["speed"],
            curvature=local_obs["curvature"],
            direction_8way=local_obs["direction_8way"],
            operator=selected_op,
            qualified=False, # Will update after qualification
            events=local_obs["events"]
        )
        hex_buffer.append(hex_frame)
        if len(hex_buffer) > 24: # Keep a small buffer for motif analysis
            hex_buffer.pop(0)

        # 5. Build Residue
        runtime_output = {
            "state": {
                "signature": {
                    "caution_scalar": state.caution_scalar,
                    "recovery_scalar": state.recovery_scalar,
                    "hold_state": state.hold_state,
                    "components": state.components,
                    "active_component_id": state.active_component_id
                },
                "orientation": {
                    "active_operator": selected_op
                },
                "hex": full_hex
            },
            "output": {
                "selected_class": 1, 
                "confidence": 0.8
            },
            "trace": trace
        }
        
        residue = build_turn_residue(
            runtime_output=runtime_output,
            prompt_text=f"audio_frame_{i}",
            intent_category="instruction",
            reply_mode="research_shell",
            turn_id=i
        )
        
        # 6. Qualify and Commit (The actual imprinting)
        # Patch9: Only pass stable motifs into residue qualification
        motif_window = int(hex_cfg.get("motif_window", 12))
        recent_frames = hex_buffer[-motif_window:]
        hex_stability = score_hex_stability(recent_frames)
        
        # Add hex info to residue for logging
        residue.hex_code = full_hex
        residue.hex_stability = hex_stability

        # v14 Qualification
        residue = qualify_residue(
            residue,
            structured_input=False,
            epsilon=float(mem_cfg.get("qualify_epsilon", 0.1)),
            recovery_threshold=-1.0, # Force recovery_present for wave imprinting
            max_switch_freq=float(mem_cfg.get("qualify_max_switch_freq", 0.5)),
            min_score=float(mem_cfg.get("qualify_min_score", 0.55)),
            min_admissibility=float(mem_cfg.get("qualify_min_admissibility", 0.0))
        )
        
        # Final gate: Hex stability must meet threshold if enabled
        if hex_cfg.get("enabled", False):
            if hex_stability < float(hex_cfg.get("stability_min", 0.72)):
                if residue.is_qualified:
                    residue.is_qualified = False
                    residue.reject_reason = "hex_instability"

        memory, residue = apply_commit_gate_and_persistence(
            state=memory,
            residue=residue,
            base_duration=int(mem_cfg.get("base_persistence_duration", 3)),
            reinforce=int(mem_cfg.get("reinforce_duration", 2)),
            max_duration=int(mem_cfg.get("max_persistence_duration", 12))
        )
        
        # Update hex frame qualification status
        hex_frame.qualified = residue.is_committed
        if hex_cfg.get("enabled", False):
            append_hex_trace_jsonl(hex_trace_path, hex_frame)

        # 7. Log and Print Progress
        append_turn_residue(residue_log_path, residue)
        if residue.is_committed:
            print(f"Frame {i}: [{full_hex}] Imprinted (Stability: {residue.stability_score:.3f}, HexStab: {hex_stability:.2f})")
        else:
            print(f"Frame {i}: [{full_hex}] Rejected ({residue.reject_reason})")

    # 8. Save final memory state
    save_memory_state(memory_path, memory)
    print(f"Imprinting complete. Memory saved to {memory_path}")
    if hex_cfg.get("enabled", False):
        print(f"Hex trace saved to {hex_trace_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", default="test.wav")
    parser.add_argument("--config", default="config/config_v14_terminal.json")
    args = parser.parse_args()
    
    if not os.path.exists(args.audio):
        print(f"Generating {args.audio}...")
        sr = 44100
        t = np.linspace(0, 5, sr * 5, endpoint=False)
        audio = 0.5 * np.sin(2 * np.pi * (220 + 440 * t) * t)
        sf.write(args.audio, audio, sr)
        
    run_imprinting(args.audio, args.config)
