import os
import json
import numpy as np
from datetime import datetime
from native_platform.run_native_platform import run_platform
from native_platform.run_state import tail_jsonl
from native_platform.eeg_synthetic_signals import generate_alpha
from native_platform.eeg_feature_adapter import signal_to_input_frames

def wrap_to_pi(v):
    return (v + np.pi) % (2 * np.pi) - np.pi

def calculate_geometry_metrics(records):
    if not records:
        return {}
    
    # Extract arrays (channels averaged or per-channel if desired, here we average across 8 channels)
    # student_theta: [N, 8], teacher_theta: [N, 8]
    st = np.array([r["student_theta"] for r in records])
    tt = np.array([r["teacher_theta"] for r in records])
    so = np.array([r["student_omega"] for r in records])
    to = np.array([r["teacher_omega"] for r in records])
    sa = np.array([r["student_amp"] for r in records])
    ta = np.array([r["teacher_amp"] for r in records])
    
    # 1. Phase Delta
    phase_delta = wrap_to_pi(st - tt)
    phase_cosine = np.cos(phase_delta)
    
    # 2. Phase Locking Value (PLV)
    # mean over time, then abs of complex mean
    # we compute per channel and then average across channels
    plv_per_channel = np.abs(np.mean(np.exp(1j * phase_delta), axis=0))
    plv = float(np.mean(plv_per_channel))
    
    # 3. Circular Mean Phase Delta (degrees)
    mean_delta_vec = np.mean(np.exp(1j * phase_delta), axis=0)
    circular_mean_deg = float(np.mean(np.angle(mean_delta_vec))) * (180.0 / np.pi)
    
    # 4. Frequency MAE
    freq_mae = float(np.mean(np.abs(so - to)))
    
    # 5. Phase Slip Count (abs diff > pi/2)
    # diff across time for each channel
    slip_threshold = 1.57079632679
    diffs = np.abs(wrap_to_pi(np.diff(phase_delta, axis=0)))
    slip_count = int(np.sum(diffs > slip_threshold))
    
    # 6. Motion Floor Violation Rate
    omega_floor = 0.001
    violations = np.logical_and(np.abs(so) < omega_floor, np.abs(to) >= omega_floor)
    violation_rate = float(np.mean(violations))
    
    # 7. Lagged Cross Correlation (on amplitudes or phases? spec says corr)
    # Let's use cosine similarity as a proxy for phase correlation
    corr = float(np.mean(phase_cosine))
    
    # 8. Trajectory Alignment Score
    alignment_score = plv * (1.0 - min(1.0, freq_mae * 2.0)) # scale freq_mae arbitrarily for score
    
    # 9. Phase Classification
    p_cos_mean = float(np.mean(phase_cosine))
    classification = "ambiguous"
    if p_cos_mean >= 0.75 and plv >= 0.70: classification = "in_phase"
    elif p_cos_mean <= -0.75 and plv >= 0.70: classification = "anti_phase"
    elif abs(p_cos_mean) < 0.25 and plv >= 0.70: classification = "quadrature"
    elif plv < 0.50 and slip_count > 0: classification = "drifting"
    
    if violation_rate > 0.25: classification = "frozen_false_stability"
    elif plv < 0.30: classification = "decorrelated"

    return {
        "value_mismatch_mean": float(np.mean([r.get("continuation_mismatch", 0.0) for r in records])),
        "phase_cosine_mean": p_cos_mean,
        "mean_phase_delta_degrees": circular_mean_deg,
        "phase_locking_value": plv,
        "frequency_mae": freq_mae,
        "phase_slip_count": slip_count,
        "motion_floor_violation_rate": violation_rate,
        "lagged_cross_correlation": corr,
        "best_lag_samples": 0, # not implemented
        "trajectory_alignment_score": alignment_score,
        "phase_classification": classification
    }

def run_experiment():
    output_dir = "runs/inductive_v2"
    os.makedirs(output_dir, exist_ok=True)
    mem_path = "sessions/inductive_test_memory.json"
    if os.path.exists(mem_path): os.remove(mem_path)
    
    sr = 256
    train_dur, disconnect_dur, recouple_dur = 20, 10, 10
    
    print("🧠 Generating Synthetic EEG for Inductive Protocol...")
    full_signal = generate_alpha(sr, train_dur + disconnect_dur + recouple_dur)
    full_frames = signal_to_input_frames(full_signal, sr, window_sec=2.0, overlap=0.9)
    
    n_total = len(full_frames)
    n_train = int(n_total * (train_dur / (train_dur + disconnect_dur + recouple_dur)))
    n_disconnect = int(n_total * (disconnect_dur / (train_dur + disconnect_dur + recouple_dur)))
    
    train_frames = full_frames[:n_train]
    disconnect_frames = full_frames[n_train : n_train + n_disconnect]
    recouple_frames = full_frames[n_train + n_disconnect :]
    
    results = {}

    print(f"▶️ Phase 1: Train ({len(train_frames)} frames)...")
    summ_train = run_platform(input_signals=train_frames, memory_path=mem_path, run_id="V2_TRAIN", connected=True, connected_state="train")
    results["train"] = calculate_geometry_metrics(tail_jsonl(summ_train["feedback_trace_path"], len(train_frames)))

    print(f"▶️ Phase 2: Disconnect ({len(disconnect_frames)} frames)...")
    summ_disc = run_platform(input_signals=disconnect_frames, memory_path=mem_path, run_id="V2_DISCONNECT", connected=False, connected_state="disconnect")
    results["disconnect"] = calculate_geometry_metrics(tail_jsonl(summ_disc["feedback_trace_path"], len(disconnect_frames)))

    print(f"▶️ Phase 3: Recouple ({len(recouple_frames)} frames)...")
    summ_rec = run_platform(input_signals=recouple_frames, memory_path=mem_path, run_id="V2_RECOUPLE", connected=True, connected_state="recouple")
    results["recouple"] = calculate_geometry_metrics(tail_jsonl(summ_rec["feedback_trace_path"], len(recouple_frames)))

    # --- Generate Rigorous Report ---
    report_path = os.path.join(output_dir, "PHASE_GEOMETRY_VALIDATION_REPORT.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Phase Geometry Validation Report (Patch 25)\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Raw Log Path: {summ_rec['feedback_trace_path']}\n\n")
        
        f.write("## 1. Metric Definitions\n")
        f.write("- **PLV (Phase Locking Value)**: Stability of angular relation [0, 1].\n")
        f.write("- **Phase Cosine**: Average alignment (1=in-phase, -1=anti-phase).\n")
        f.write("- **Freq MAE**: Mean velocity mismatch between teacher and student.\n")
        f.write("- **Alignment Score**: PLV penalized by frequency error.\n\n")
        
        f.write("## 2. Phase-by-Phase Geometry\n")
        f.write("| Phase | Mismatch | Cosine | PLV | Freq MAE | Slips | Floor Viol | Alignment | Classification |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for phase in ["train", "disconnect", "recouple"]:
            m = results[phase]
            f.write(f"| {phase.capitalize()} | {m['value_mismatch_mean']:.4f} | {m['phase_cosine_mean']:.3f} | {m['phase_locking_value']:.3f} | {m['frequency_mae']:.4f} | {m['phase_slip_count']} | {m['motion_floor_violation_rate']:.2f} | {m['trajectory_alignment_score']:.3f} | **{m['phase_classification']}** |\n")
        
        f.write("\n## 3. Claim Verification\n")
        d = results["disconnect"]
        is_true = (d["phase_locking_value"] >= 0.70 and 
                   abs(d["mean_phase_delta_degrees"]) <= 45 and 
                   d["motion_floor_violation_rate"] <= 0.10 and 
                   d["trajectory_alignment_score"] >= 0.65)
        
        status = "✅ TRUE CONTINUATION" if is_true else "❌ FAILED RIGOROUS CLAIM"
        f.write(f"### Overall Claim: {status}\n\n")
        
        if not is_true:
            f.write("#### Failure Diagnosis:\n")
            if d["phase_locking_value"] < 0.70: f.write("- Low PLV: Phase relation is wandering.\n")
            if abs(d["mean_phase_delta_degrees"]) > 45: f.write("- Phase Offset: Stable but incorrect angular alignment.\n")
            if d["motion_floor_violation_rate"] > 0.10: f.write("- Frozen Stability: Model stopped moving to hide error.\n")
            if d["trajectory_alignment_score"] < 0.65: f.write("- Low Alignment: Frequency or phase mismatch too high.\n")

    print(f"\n✅ Rigorous Experiment Complete. Report: {report_path}")

if __name__ == "__main__":
    run_experiment()
