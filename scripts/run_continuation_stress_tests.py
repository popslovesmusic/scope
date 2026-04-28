import os
import json
import numpy as np
from datetime import datetime
from native_platform.run_native_platform import run_platform
from native_platform.run_state import tail_jsonl
from native_platform.eeg_synthetic_signals import generate_alpha, generate_theta, generate_mixed_alpha_theta
from native_platform.eeg_feature_adapter import signal_to_input_frames, extract_window_features, features_to_scope_input
from native_platform.phase_space import compute_phase_vector
from native_platform.signalscope_core import SignalScope

def wrap_to_pi(v):
    return (v + np.pi) % (2 * np.pi) - np.pi

def map_features_to_phi(frames):
    scope = SignalScope()
    phis = []
    for f in frames:
        sd = scope.update(f)
        phi = compute_phase_vector(sd['W_local'], sd['C'], sd['E'], sd['V'])
        phis.append(phi)
    return np.array(phis)

def calculate_geometry_metrics(records, ground_truth_phi=None, shuffle_teacher=False):
    if not records:
        return {}
    
    st = np.array([r["student_theta"] for r in records]) # [N, 8]
    if ground_truth_phi is not None:
        tt = np.array(ground_truth_phi)
    else:
        tt = np.array([r["teacher_theta"] for r in records])
    
    if shuffle_teacher:
        # Shift teacher by 50% of the window to break timing alignment
        tt = np.roll(tt, len(tt)//2, axis=0)
    
    # Unit vector alignment (Cosine)
    st_norm = st / (np.linalg.norm(st, axis=1, keepdims=True) + 1e-9)
    tt_norm = tt / (np.linalg.norm(tt, axis=1, keepdims=True) + 1e-9)
    cosines = np.sum(st_norm * tt_norm, axis=1)
    
    # PLV on Velocity (Omega) - More sensitive to motion continuation
    so = np.diff(st, axis=0)
    to = np.diff(tt, axis=0)
    
    # Project 8-dim velocity to an angle
    st_vel_angles = np.arctan2(so[:, 1], so[:, 0])
    tt_vel_angles = np.arctan2(to[:, 1], to[:, 0])
    phase_delta = wrap_to_pi(st_vel_angles - tt_vel_angles)
    plv = float(np.abs(np.mean(np.exp(1j * phase_delta))))
    
    freq_mae = float(np.mean(np.abs(so - to)))
    alignment_score = plv * (1.0 - min(1.0, freq_mae * 5.0)) # Higher penalty for stress
    
    return {
        "mismatch": float(np.mean([r.get("continuation_mismatch", 0.0) for r in records])),
        "cosine": float(np.mean(cosines)),
        "plv": plv,
        "freq_mae": freq_mae,
        "alignment_score": alignment_score
    }

def run_stress_test_battery():
    output_dir = "runs/stress_tests"
    os.makedirs(output_dir, exist_ok=True)
    mem_path = "sessions/stress_test_memory.json"
    
    sr = 256
    results = {}

    # Control A: Shuffle Control
    print("\n🕵️ Running Control A: Shuffle Control (with Temporal Shift)...")
    if os.path.exists(mem_path): os.remove(mem_path)
    alpha_raw = generate_alpha(sr, 15)
    alpha_frames = signal_to_input_frames(alpha_raw, sr)
    summ = run_platform(input_signals=alpha_frames, memory_path=mem_path, run_id="CTRL_SHUFFLE", connected=True)
    recs = tail_jsonl(summ["feedback_trace_path"], len(alpha_frames))
    
    results["shuffle_control_normal"] = calculate_geometry_metrics(recs, shuffle_teacher=False)
    results["shuffle_control_shuffled"] = calculate_geometry_metrics(recs, shuffle_teacher=True)
    print(f"    Normal PLV: {results['shuffle_control_normal']['plv']:.3f}, Shuffled PLV: {results['shuffle_control_shuffled']['plv']:.3f}")

    # S1: Long Disconnect Survival
    print("\n🏃 Running S1: Long Disconnect (60s)...")
    if os.path.exists(mem_path): os.remove(mem_path)
    full_alpha = generate_alpha(sr, 80)
    full_frames = signal_to_input_frames(full_alpha, sr)
    n_train = int(len(full_frames) * (20/80))
    n_disc = int(len(full_frames) * (60/80))
    
    run_platform(input_signals=full_frames[:n_train], memory_path=mem_path, run_id="S1_TRAIN", connected=True)
    teacher_disc_phi = map_features_to_phi(full_frames[n_train:n_train+n_disc])
    summ_disc = run_platform(input_signals=full_frames[n_train:n_train+n_disc], memory_path=mem_path, run_id="S1_DISC", connected=False)
    disc_recs = tail_jsonl(summ_disc["feedback_trace_path"], n_disc)
    
    n_10s = int(len(disc_recs) * (10/60))
    results["S1_early"] = calculate_geometry_metrics(disc_recs[:n_10s], ground_truth_phi=teacher_disc_phi[:n_10s])
    results["S1_late"] = calculate_geometry_metrics(disc_recs[-n_10s:], ground_truth_phi=teacher_disc_phi[-n_10s:])
    print(f"    Early PLV: {results['S1_early']['plv']:.3f}, Late PLV: {results['S1_late']['plv']:.3f}")

    # S3: Multi-Band Signal
    print("\n🎹 Running S3: Multi-Band (Mixed Alpha/Theta)...")
    if os.path.exists(mem_path): os.remove(mem_path)
    mixed = generate_mixed_alpha_theta(sr, 30)
    mixed_frames = signal_to_input_frames(mixed, sr)
    n_tr = int(len(mixed_frames) * 0.6)
    
    run_platform(input_signals=mixed_frames[:n_tr], memory_path=mem_path, run_id="S3_TRAIN", connected=True)
    teacher_s3_phi = map_features_to_phi(mixed_frames[n_tr:])
    summ_s3 = run_platform(input_signals=mixed_frames[n_tr:], memory_path=mem_path, run_id="S3_DISC", connected=False)
    results["S3"] = calculate_geometry_metrics(tail_jsonl(summ_s3["feedback_trace_path"], len(teacher_s3_phi)), ground_truth_phi=teacher_s3_phi)
    print(f"    Mixed PLV: {results['S3']['plv']:.3f}")

    # Generate Report
    report_path = os.path.join(output_dir, "CONTINUATION_STRESS_TEST_REPORT.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Continuation Stress Test Report (Patch 26)\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 1. Controls\n")
        f.write("| Control | Normal PLV | Shuffled PLV | Result |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        # Shuffle control passes if shifted PLV is significantly lower
        res = "PASSED" if results["shuffle_control_shuffled"]["plv"] < results["shuffle_control_normal"]["plv"] * 0.7 else "FAILED"
        f.write(f"| Shuffle | {results['shuffle_control_normal']['plv']:.3f} | {results['shuffle_control_shuffled']['plv']:.3f} | {res} |\n\n")
        
        f.write("## 2. Stress Test Results\n")
        f.write("| ID | Test Name | PLV | Cosine | Freq MAE | Alignment | Status |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for tid, key in [("S1e", "S1_early"), ("S1l", "S1_late"), ("S3", "S3")]:
            m = results[key]
            status = "STABLE" if m['plv'] > 0.6 else "DRIFTING"
            f.write(f"| {tid} | {key} | {m['plv']:.3f} | {m['cosine']:.3f} | {m['freq_mae']:.4f} | {m['alignment_score']:.3f} | {status} |\n")

    print(f"\n✅ Stress Tests Complete. Report: {report_path}")

if __name__ == "__main__":
    run_stress_test_battery()
