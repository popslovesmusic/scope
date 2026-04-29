import os
import json
import numpy as np
from datetime import datetime
from native_platform.run_native_platform import run_platform
from native_platform.run_state import tail_jsonl
from native_platform.eeg_synthetic_signals import generate_alpha, generate_theta, generate_mixed_alpha_theta
from native_platform.eeg_feature_adapter import signal_to_input_frames
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

def calculate_detailed_metrics(records, ground_truth_phi=None, shuffle_teacher=False):
    if not records:
        return {}
    
    # st: Student Theta [N, 8]
    st = np.array([r.get("anchor_theta", r["student_theta"]) for r in records])
    
    if ground_truth_phi is not None:
        tt = np.array(ground_truth_phi)
    else:
        tt = np.array([r["teacher_theta"] for r in records])
    
    if shuffle_teacher:
        indices = np.random.permutation(len(tt))
        tt = tt[indices]
    
    # UNIT VECTOR COSINE
    st_norm = st / (np.linalg.norm(st, axis=1, keepdims=True) + 1e-9)
    tt_norm = tt / (np.linalg.norm(tt, axis=1, keepdims=True) + 1e-9)
    cosines = np.sum(st_norm * tt_norm, axis=1)
    
    # VELOCITY COHERENCE (Omega PLV)
    # This is the most sensitive metric to break illusions
    so = np.diff(st, axis=0) # [N-1, 8]
    to = np.diff(tt, axis=0) # [N-1, 8]
    
    # Normalize velocities
    so_n = so / (np.linalg.norm(so, axis=1, keepdims=True) + 1e-9)
    to_n = to / (np.linalg.norm(to, axis=1, keepdims=True) + 1e-9)
    
    # Dot product of unit velocity vectors
    vel_cosines = np.sum(so_n * to_n, axis=1)
    
    # True PLV: stability of the direction of the error in 8D
    # (magnitude of the mean error-direction vector)
    # We'll use the mean unit alignment as a PLV proxy for vectors
    plv = float(np.abs(np.mean(vel_cosines)))
    
    # Frequency accuracy
    freq_mae = float(np.mean(np.abs(so - to)))
    violation_rate = float(np.mean(np.logical_and(np.abs(so) < 0.001, np.abs(to) >= 0.001)))
    alignment_score = plv * (1.0 - min(1.0, freq_mae * 5.0))

    return {
        "mismatch": float(np.mean([r.get("continuation_mismatch", 0.0) for r in records])),
        "cosine": float(np.mean(cosines)),
        "plv": plv,
        "freq_mae": freq_mae,
        "violation_rate": violation_rate,
        "alignment_score": alignment_score,
        "circular_mean_deg": 0.0
    }

def run_patch27_validation():
    output_dir = "runs/patch27"
    os.makedirs(output_dir, exist_ok=True)
    mem_path = "sessions/patch27_test_memory.json"
    
    sr = 256
    results = {}

    print("\n🕵️ Step 1: Leakage Audit (Normal vs Shuffled Control)...")
    if os.path.exists(mem_path): os.remove(mem_path)
    alpha_raw = generate_alpha(sr, 25) # Longer to break correlations
    alpha_frames = signal_to_input_frames(alpha_raw, sr)
    n_tr = 50
    run_platform(input_signals=alpha_frames[:n_tr], memory_path=mem_path, run_id="P27_CTRL_TRAIN", connected=True)
    
    teacher_eval_phi = map_features_to_phi(alpha_frames[n_tr:])
    summ = run_platform(input_signals=alpha_frames[n_tr:], memory_path=mem_path, run_id="P27_CTRL_DISC", connected=False)
    recs = tail_jsonl(summ["feedback_trace_path"], len(alpha_frames)-n_tr)
    
    results["control_normal"] = calculate_detailed_metrics(recs, ground_truth_phi=teacher_eval_phi)
    results["control_shuffled"] = calculate_detailed_metrics(recs, ground_truth_phi=teacher_eval_phi, shuffle_teacher=True)
    print(f"    Normal Velocity PLV: {results['control_normal']['plv']:.3f}, Shuffled Velocity PLV: {results['control_shuffled']['plv']:.3f}")

    print("\n🏃 Step 2: S1 Long Disconnect (Trajectory Persistence)...")
    if os.path.exists(mem_path): os.remove(mem_path)
    full_alpha = generate_alpha(sr, 60)
    full_frames = signal_to_input_frames(full_alpha, sr)
    n_train = int(len(full_frames) * (20/60))
    n_disc = int(len(full_frames) * (40/60))
    
    run_platform(input_signals=full_frames[:n_train], memory_path=mem_path, run_id="P27_S1_TRAIN", connected=True)
    teacher_disc_phi = map_features_to_phi(full_frames[n_train:n_train+n_disc])
    summ_disc = run_platform(input_signals=full_frames[n_train:n_train+n_disc], memory_path=mem_path, run_id="P27_S1_DISC", connected=False)
    disc_recs = tail_jsonl(summ_disc["feedback_trace_path"], n_disc)
    
    results["S1_early"] = calculate_detailed_metrics(disc_recs[:50], ground_truth_phi=teacher_disc_phi[:50])
    results["S1_late"] = calculate_detailed_metrics(disc_recs[-50:], ground_truth_phi=teacher_disc_phi[-50:])
    print(f"    Early PLV: {results['S1_early']['plv']:.3f}, Late PLV: {results['S1_late']['plv']:.3f}")

    report_path = os.path.join(output_dir, "PATCH_27_RECURSIVE_MOTION_ANCHOR_REPORT.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Patch 27: Recursive Motion Anchor Validation Report\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 1. Leakage Audit & Controls\n")
        f.write("| Test | Normal Vel-PLV | Shuffled Vel-PLV | Result |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        leak_res = "PASSED" if results["control_shuffled"]["plv"] < results["control_normal"]["plv"] * 0.7 else "FAIL (INSIGNIFICANT MARGIN)"
        f.write(f"| Shuffle Control | {results['control_normal']['plv']:.3f} | {results['control_shuffled']['plv']:.3f} | {leak_res} |\n\n")
        
        f.write("## 2. Recursive Continuation Metrics\n")
        f.write("| ID | Test Name | PLV | Freq MAE | Floor Viol | Alignment | |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for tid, key in [("S1e", "S1_early"), ("S1l", "S1_late")]:
            m = results[key]
            f.write(f"| {tid} | {key} | {m['plv']:.3f} | {m['freq_mae']:.4f} | {m['violation_rate']:.2f} | {m['alignment_score']:.3f} | |\n")

    print(f"\n✅ Validation Complete. Report: {report_path}")

if __name__ == "__main__":
    run_patch27_validation()
