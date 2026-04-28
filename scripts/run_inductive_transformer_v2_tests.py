import os
import json
import numpy as np
import csv
from datetime import datetime
from native_platform.run_native_platform import run_platform
from native_platform.run_state import tail_jsonl
from native_platform.eeg_synthetic_signals import generate_alpha
from native_platform.eeg_feature_adapter import signal_to_input_frames

def get_protocol_metrics(trace_path, frames):
    records = tail_jsonl(trace_path, frames)
    if not records:
        return {}
    
    mismatches = [r.get("continuation_mismatch", 0.0) for r in records]
    phase_errors = [r.get("phase_error", 0.0) for r in records]
    drifts = [r.get("frequency_drift", 0.0) for r in records]
    decisions = [r.get("survivability_decision", "unknown") for r in records]
    
    return {
        "mismatch_mean": float(np.mean(mismatches)),
        "phase_error_mean": float(np.mean(phase_errors)),
        "drift_mean": float(np.mean(drifts)),
        "reinforce_rate": np.mean([1.0 if d == "reinforce" else 0.0 for d in decisions]),
        "raw_records": records
    }

def run_experiment():
    output_dir = "runs/inductive_v2"
    os.makedirs(output_dir, exist_ok=True)
    mem_path = "sessions/inductive_test_memory.json"
    if os.path.exists(mem_path): os.remove(mem_path)
    
    sr = 256
    # Protocol durations
    train_dur = 20
    disconnect_dur = 10
    recouple_dur = 10
    
    print("🧠 Generating Synthetic EEG for Inductive Protocol...")
    full_signal = generate_alpha(sr, train_dur + disconnect_dur + recouple_dur)
    full_frames = signal_to_input_frames(full_signal, sr, window_sec=2.0, overlap=0.9)
    
    # Split frames
    # Each window is 2s, overlap 0.9 -> step is 0.2s.
    # Total frames ~ (40 - 2) / 0.2 = 190.
    n_total = len(full_frames)
    n_train = int(n_total * (train_dur / (train_dur + disconnect_dur + recouple_dur)))
    n_disconnect = int(n_total * (disconnect_dur / (train_dur + disconnect_dur + recouple_dur)))
    
    train_frames = full_frames[:n_train]
    disconnect_frames = full_frames[n_train : n_train + n_disconnect]
    recouple_frames = full_frames[n_train + n_disconnect :]
    
    results = {}

    print(f"▶️ Phase 1: Train ({len(train_frames)} frames)...")
    summ_train = run_platform(input_signals=train_frames, memory_path=mem_path, run_id="V2_TRAIN", connected=True)
    results["train"] = get_protocol_metrics(summ_train["feedback_trace_path"], len(train_frames))

    print(f"▶️ Phase 2: Disconnect ({len(disconnect_frames)} frames)...")
    # Disconnect means we still provide 'input_signals' for verification (to measure error), 
    # but run_platform uses connected=False internally.
    summ_disc = run_platform(input_signals=disconnect_frames, memory_path=mem_path, run_id="V2_DISCONNECT", connected=False)
    results["disconnect"] = get_protocol_metrics(summ_disc["feedback_trace_path"], len(disconnect_frames))

    print(f"▶️ Phase 3: Recouple ({len(recouple_frames)} frames)...")
    summ_rec = run_platform(input_signals=recouple_frames, memory_path=mem_path, run_id="V2_RECOUPLE", connected=True)
    results["recouple"] = get_protocol_metrics(summ_rec["feedback_trace_path"], len(recouple_frames))

    # --- Generate Report ---
    report_path = os.path.join(output_dir, "INDUCTIVE_TRANSFORMER_V2_REPORT.md")
    with open(report_path, "w") as f:
        f.write("# Inductive Transformer v2 Validation Report\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 1. Protocol Summary\n")
        f.write(f"- Train Duration: {train_dur}s\n")
        f.write(f"- Disconnect Duration: {disconnect_dur}s\n")
        f.write(f"- Recouple Duration: {recouple_dur}s\n\n")
        
        f.write("## 2. Key Metrics\n")
        f.write("| Phase | Mismatch Mean | Phase Error | Drift | Reinforce Rate |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        for phase in ["train", "disconnect", "recouple"]:
            m = results[phase]
            f.write(f"| {phase.capitalize()} | {m['mismatch_mean']:.4f} | {m['phase_error_mean']:.4f} | {m['drift_mean']:.4f} | {m['reinforce_rate']:.2f} |\n")
        
        f.write("\n## 3. Analysis\n")
        
        # Calculate survival quality
        disc_mismatch = results["disconnect"]["mismatch_mean"]
        train_mismatch = results["train"]["mismatch_mean"]
        f.write(f"- **Continuation Stability**: Disconnect mismatch is {disc_mismatch/train_mismatch:.1f}x training levels.\n")
        
        # Calculate recoupling shock
        rec_first_mismatch = results["recouple"]["raw_records"][0]["continuation_mismatch"]
        f.write(f"- **Recoupling Shock**: First frame mismatch after recouple: {rec_first_mismatch:.4f}\n")
        
    print(f"\n✅ Experiment Complete. Report: {report_path}")

if __name__ == "__main__":
    run_experiment()
