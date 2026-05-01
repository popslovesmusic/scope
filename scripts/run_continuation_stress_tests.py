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

    st = np.array([r["phi_continued"] for r in records], dtype=float)  # [N, 8]
    if ground_truth_phi is not None:
        tt = np.array(ground_truth_phi, dtype=float)
    else:
        tt = np.array([r["phi_current"] for r in records], dtype=float)

    if shuffle_teacher:
        tt = np.roll(tt, len(tt)//2, axis=0)

    # Full-state cosine alignment: static similarity of teacher/student state vectors.
    st_norm = st / (np.linalg.norm(st, axis=1, keepdims=True) + 1e-9)
    tt_norm = tt / (np.linalg.norm(tt, axis=1, keepdims=True) + 1e-9)
    cosines = np.sum(st_norm * tt_norm, axis=1)
    state_cosine = float(np.mean(cosines))

    # Full-8D velocity coherence: continuation-direction similarity.
    so = np.diff(st, axis=0)
    to = np.diff(tt, axis=0)

    if len(so) == 0 or len(to) == 0:
        velocity_coherence = 0.0
        omega_mae = 0.0
    else:
        so_norm = so / (np.linalg.norm(so, axis=1, keepdims=True) + 1e-9)
        to_norm = to / (np.linalg.norm(to, axis=1, keepdims=True) + 1e-9)
        velocity_coherence = float(np.mean(np.sum(so_norm * to_norm, axis=1)))
        omega_mae = float(np.mean(np.abs(so - to)))

    # Bounded per-channel state similarity. Useful for identity/persistence quality.
    state_sim_per_frame = 1.0 - (np.abs(st - tt) / (np.abs(st) + np.abs(tt) + 1e-9))
    state_similarity = float(np.mean(state_sim_per_frame))

    # Legacy-compatible penalty score, but now based on meaningful full-8D metrics.
    alignment_score = velocity_coherence * state_similarity * (1.0 - min(1.0, omega_mae * 5.0))

    return {
        "mismatch": float(np.mean([r.get("continuation_mismatch", 0.0) for r in records])),
        "cosine": state_cosine,
        "state_cosine": state_cosine,
        "velocity_coherence": velocity_coherence,
        "state_similarity": state_similarity,
        "omega_mae": omega_mae,
        "freq_mae": omega_mae,
        "alignment_score": alignment_score,

        "plv": velocity_coherence,
        "plv_deprecated_alias": True,
        "metric_patch": "patch_31_full_8d_metric_fix"
    }

def run_stress_test_battery(w_trace=0.75, w_inductive=0.15, w_anchor=0.03, damping=0.92, sweep_id=None, experimental_rotation=False):
    output_dir = "runs/stress_tests"
    os.makedirs(output_dir, exist_ok=True)
    mem_path = f"sessions/stress_test_memory_{sweep_id}.json" if sweep_id else "sessions/stress_test_memory.json"
    
    sr = 256
    results = {}

    # Control A: Shuffle Control
    print(f"\n🕵️ Running Control A: Shuffle Control (Sweep: {sweep_id}, ExpRot: {experimental_rotation})...")
    if os.path.exists(mem_path): os.remove(mem_path)
    alpha_raw = generate_alpha(sr, 15)
    alpha_frames = signal_to_input_frames(alpha_raw, sr)
    summ = run_platform(input_signals=alpha_frames, memory_path=mem_path, run_id=f"CTRL_{sweep_id}" if sweep_id else "CTRL_SHUFFLE", connected=True, w_trace=w_trace, w_inductive=w_inductive, w_anchor=w_anchor, damping=damping, experimental_rotation=experimental_rotation)
    recs = tail_jsonl(summ["feedback_trace_path"], len(alpha_frames))
    
    results["shuffle_control_normal"] = calculate_geometry_metrics(recs, shuffle_teacher=False)
    results["shuffle_control_shuffled"] = calculate_geometry_metrics(recs, shuffle_teacher=True)
    print(f"    Normal VelCoherence: {results['shuffle_control_normal']['velocity_coherence']:.3f}, Shuffled VelCoherence: {results['shuffle_control_shuffled']['velocity_coherence']:.3f}")

    # S1: Long Disconnect Survival
    print(f"\n🏃 Running S1: Long Disconnect (60s) (Sweep: {sweep_id})...")
    if os.path.exists(mem_path): os.remove(mem_path)
    full_alpha = generate_alpha(sr, 80)
    full_frames = signal_to_input_frames(full_alpha, sr)
    n_train = int(len(full_frames) * (20/80))
    n_disc = int(len(full_frames) * (60/80))
    
    run_platform(input_signals=full_frames[:n_train], memory_path=mem_path, run_id=f"S1_TRAIN_{sweep_id}" if sweep_id else "S1_TRAIN", connected=True, w_trace=w_trace, w_inductive=w_inductive, w_anchor=w_anchor, damping=damping, experimental_rotation=experimental_rotation)
    teacher_disc_phi = map_features_to_phi(full_frames[n_train:n_train+n_disc])
    summ_disc = run_platform(input_signals=full_frames[n_train:n_train+n_disc], memory_path=mem_path, run_id=f"S1_DISC_{sweep_id}" if sweep_id else "S1_DISC", connected=False, w_trace=w_trace, w_inductive=w_inductive, w_anchor=w_anchor, damping=damping, experimental_rotation=experimental_rotation)
    disc_recs = tail_jsonl(summ_disc["feedback_trace_path"], n_disc)
    
    n_10s = int(len(disc_recs) * (10/60))
    results["S1_early"] = calculate_geometry_metrics(disc_recs[:n_10s], ground_truth_phi=teacher_disc_phi[:n_10s])
    results["S1_late"] = calculate_geometry_metrics(disc_recs[-n_10s:], ground_truth_phi=teacher_disc_phi[-n_10s:])
    print(f"    Early VelCoherence: {results['S1_early']['velocity_coherence']:.3f}, Late VelCoherence: {results['S1_late']['velocity_coherence']:.3f}")

    # S3: Multi-Band Signal
    print(f"\n🎹 Running S3: Multi-Band (Mixed Alpha/Theta) (Sweep: {sweep_id})...")
    if os.path.exists(mem_path): os.remove(mem_path)
    mixed = generate_mixed_alpha_theta(sr, 30)
    mixed_frames = signal_to_input_frames(mixed, sr)
    n_tr = int(len(mixed_frames) * 0.6)
    
    run_platform(input_signals=mixed_frames[:n_tr], memory_path=mem_path, run_id=f"S3_TRAIN_{sweep_id}" if sweep_id else "S3_TRAIN", connected=True, w_trace=w_trace, w_inductive=w_inductive, w_anchor=w_anchor, damping=damping, experimental_rotation=experimental_rotation)
    teacher_s3_phi = map_features_to_phi(mixed_frames[n_tr:])
    summ_s3 = run_platform(input_signals=mixed_frames[n_tr:], memory_path=mem_path, run_id=f"S3_DISC_{sweep_id}" if sweep_id else "S3_DISC", connected=False, w_trace=w_trace, w_inductive=w_inductive, w_anchor=w_anchor, damping=damping, experimental_rotation=experimental_rotation)
    results["S3"] = calculate_geometry_metrics(tail_jsonl(summ_s3["feedback_trace_path"], len(teacher_s3_phi)), ground_truth_phi=teacher_s3_phi)
    print(f"    Mixed VelCoherence: {results['S3']['velocity_coherence']:.3f}")

    # Generate Report
    report_name = f"CONTINUATION_STRESS_TEST_REPORT_{sweep_id}.md" if sweep_id else "CONTINUATION_STRESS_TEST_REPORT.md"
    report_path = os.path.join(output_dir, report_name)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# Continuation Stress Test Report (Patch 33-35 - Sweep: {sweep_id if sweep_id else 'Default'})\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Config: w_trace={w_trace}, w_inductive={w_inductive}, w_anchor={w_anchor}, damping={damping}, experimental_rotation={experimental_rotation}\n\n")
        
        f.write("## 1. Controls\n")
        f.write("| Control | Normal VelCoherence | Shuffled VelCoherence | Result |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        # Shuffle control passes if shifted coherence is significantly lower
        res = "PASSED" if results["shuffle_control_shuffled"]["velocity_coherence"] < results["shuffle_control_normal"]["velocity_coherence"] * 0.7 else "FAILED"
        f.write(f"| Shuffle | {results['shuffle_control_normal']['velocity_coherence']:.3f} | {results['shuffle_control_shuffled']['velocity_coherence']:.3f} | {res} |\n\n")
        
        f.write("## 2. Stress Test Results\n")
        f.write("| ID | Test Name | VelCoherence | StateSim | Freq MAE | Alignment | Status |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for tid, key in [("S1e", "S1_early"), ("S1l", "S1_late"), ("S3", "S3")]:
            m = results[key]
            status = "STABLE" if (m['velocity_coherence'] > 0.5 and m['state_similarity'] > 0.65) else "DRIFTING"
            f.write(f"| {tid} | {key} | {m['velocity_coherence']:.3f} | {m['state_similarity']:.3f} | {m['omega_mae']:.4f} | {m['alignment_score']:.3f} | {status} |\n")

    return results

def run_gain_sweep():
    # Patch 34 Sweep Values
    w_traces = [0.5, 0.75, 1.0] 
    w_inductives = [0.15, 0.25, 0.35]
    dampings = [0.88, 0.92, 0.97]
    w_anchor = 0.03 # Patch 34 recommendation
    
    best_coherence = -1.0
    best_config = None
    
    sweep_results = []
    
    for wt in w_traces:
        for wi in w_inductives:
            for d in dampings:
                sid = f"wt{wt}_wi{wi}_d{d}"
                res = run_stress_test_battery(w_trace=wt, w_inductive=wi, w_anchor=w_anchor, damping=d, sweep_id=sid)
                avg_coherence = (res["S1_late"]["velocity_coherence"] + res["S3"]["velocity_coherence"]) / 2.0
                sweep_results.append({"config": sid, "avg_coherence": avg_coherence})
                if avg_coherence > best_coherence:
                    best_coherence = avg_coherence
                    best_config = sid
                    
    print("\n📊 Sweep Results Summary:")
    for r in sweep_results:
        print(f"  Config: {r['config']}, Avg Coherence: {r['avg_coherence']:.3f}")
    print(f"\n🏆 Best Config: {best_config} with Coherence: {best_coherence:.3f}")

if __name__ == "__main__":
    import sys
    if "--sweep" in sys.argv:
        run_gain_sweep()
    else:
        run_stress_test_battery()
