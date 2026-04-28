import os
import json
import numpy as np
import csv
from datetime import datetime
from native_platform.run_native_platform import run_platform
from native_platform.run_state import tail_jsonl
from core.memory_layer import load_memory_state

def generate_sine(frequency, amplitude, frames, sr=10.0):
    t = np.linspace(0, frames/sr, frames)
    return amplitude * np.sin(2 * np.pi * frequency * t)

def clear_test_memory(mem_path):
    if os.path.exists(mem_path):
        os.remove(mem_path)

def get_detailed_metrics(trace_path, frames):
    records = tail_jsonl(trace_path, frames)
    if not records:
        return {}
    
    mismatches = [r.get("continuation_mismatch", 0.0) for r in records]
    groove_ids = [str(r.get("active_groove_id") or "none") for r in records]
    commit_rate = np.mean([1.0 if r.get("residue_committed") else 0.0 for r in records])
    
    unique_grooves, counts = np.unique(groove_ids, return_counts=True)
    dominant_groove = unique_grooves[np.argmax(counts)] if len(unique_grooves) > 0 else "none"
    dominant_frac = np.max(counts) / len(groove_ids) if len(groove_ids) > 0 else 0.0
    
    # Estimate lock breaks by looking at groove transitions
    switch_indices = np.where(np.array(groove_ids)[:-1] != np.array(groove_ids)[1:])[0]
    
    # successful_traversals and trace_segment_count from last record
    last_rec = records[-1]
    
    return {
        "continuation_mismatch_mean": float(np.mean(mismatches)),
        "continuation_mismatch_std": float(np.std(mismatches)),
        "continuation_mismatch_max": float(np.max(mismatches)),
        "dominant_groove": dominant_groove,
        "dominant_groove_fraction": float(dominant_frac),
        "groove_switch_count": len(switch_indices),
        "residue_commit_rate": float(commit_rate),
        "successful_traversals": int(last_rec.get("successful_traversals", 0)),
        "trace_segment_count": int(last_rec.get("trace_segment_count", 0)),
        "lock_break_count": len(switch_indices)
    }

def run_validation():
    output_dir = "runs/identity_lock_validation"
    os.makedirs(output_dir, exist_ok=True)
    
    seeds = [101, 202, 303, 404, 505]
    frames = 500
    wave_A = generate_sine(5.0, 1.0, frames)
    wave_B = generate_sine(12.0, 1.0, frames)
    
    all_seed_results = []
    
    for seed in seeds:
        print(f"\n🌱 Running Validation with Seed: {seed}")
        np.random.seed(seed)
        mem_path = f"sessions/test_memory_seed_{seed}.json"
        clear_test_memory(mem_path)
        
        seed_results = {}
        
        # --- T1: Repeated A learning curve ---
        t1_passes = []
        for i in range(5):
            summ = run_platform(input_signals=wave_A, memory_path=mem_path, run_id=f"T1_S{seed}_A_pass{i+1}")
            m = get_detailed_metrics(summ["feedback_trace_path"], frames)
            t1_passes.append(m)
            
        summ = run_platform(input_signals=wave_A, memory_path=mem_path, run_id=f"T1_S{seed}_replay_A")
        m_replay = get_detailed_metrics(summ["feedback_trace_path"], frames)
        t1_passes.append(m_replay)
        seed_results["T1"] = t1_passes

        # --- T2: A vs B directionality ---
        summ = run_platform(input_signals=wave_B, memory_path=mem_path, run_id=f"T2_S{seed}_run_B")
        m_B = get_detailed_metrics(summ["feedback_trace_path"], frames)
        
        summ = run_platform(input_signals=wave_A, memory_path=mem_path, run_id=f"T2_S{seed}_run_A_after_B")
        m_A_after = get_detailed_metrics(summ["feedback_trace_path"], frames)
        seed_results["T2"] = {"B": m_B, "A_after_B": m_A_after}

        # --- T3: Catastrophic forgetting ---
        clear_test_memory(mem_path)
        for i in range(5): run_platform(input_signals=wave_A, memory_path=mem_path, run_id=f"T3_S{seed}_A_train{i+1}")
        for i in range(5): run_platform(input_signals=wave_B, memory_path=mem_path, run_id=f"T3_S{seed}_B_train{i+1}")
        
        summ = run_platform(input_signals=wave_A, memory_path=mem_path, run_id=f"T3_S{seed}_replay_A")
        m_A_forget = get_detailed_metrics(summ["feedback_trace_path"], frames)
        
        summ = run_platform(input_signals=wave_B, memory_path=mem_path, run_id=f"T3_S{seed}_replay_B")
        m_B_forget = get_detailed_metrics(summ["feedback_trace_path"], frames)
        seed_results["T3"] = {"replay_A": m_A_forget, "replay_B": m_B_forget}

        # --- T4: Withheld continuation ---
        clear_test_memory(mem_path)
        for i in range(5): run_platform(input_signals=wave_A, memory_path=mem_path, run_id=f"T4_S{seed}_A_train{i+1}")
        
        wave_A_masked = wave_A.copy()
        wave_A_masked[int(frames*0.6):] = 0.0
        
        summ = run_platform(input_signals=wave_A_masked, memory_path=mem_path, run_id=f"T4_S{seed}_A_masked_trained")
        m_trained = get_detailed_metrics(summ["feedback_trace_path"], frames)
        
        clear_test_memory(mem_path)
        summ = run_platform(input_signals=wave_A_masked, memory_path=mem_path, run_id=f"T4_S{seed}_A_masked_untrained")
        m_untrained = get_detailed_metrics(summ["feedback_trace_path"], frames)
        seed_results["T4"] = {"trained": m_trained, "untrained": m_untrained}
        
        all_seed_results.append(seed_results)

    # --- Aggregation and Reporting ---
    print("\n📊 Aggregating Results across seeds...")
    
    t1_means = []
    for pass_idx in range(6):
        mismatch_vals = [res["T1"][pass_idx]["continuation_mismatch_mean"] for res in all_seed_results]
        t1_means.append({
            "pass": pass_idx + 1 if pass_idx < 5 else "replay",
            "mismatch_mean": float(np.mean(mismatch_vals)),
            "mismatch_std": float(np.std(mismatch_vals))
        })
    
    csv_path = os.path.join(output_dir, "pass_curve_A.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["pass", "mismatch_mean", "mismatch_std"])
        writer.writeheader()
        writer.writerows(t1_means)
    
    with open(os.path.join(output_dir, "comparison_report.json"), "w") as f:
        json.dump(all_seed_results, f, indent=2)

    agg = {
        "T1_learning": {
            "A_pass1_mean": float(np.mean([res["T1"][0]["continuation_mismatch_mean"] for res in all_seed_results])),
            "replay_A_mean": float(np.mean([res["T1"][-1]["continuation_mismatch_mean"] for res in all_seed_results])),
            "switch_count_mean": float(np.mean([res["T1"][-1]["groove_switch_count"] for res in all_seed_results]))
        },
        "T2_directionality": {
            "B_mean": float(np.mean([res["T2"]["B"]["continuation_mismatch_mean"] for res in all_seed_results])),
            "A_after_B_mean": float(np.mean([res["T2"]["A_after_B"]["continuation_mismatch_mean"] for res in all_seed_results]))
        },
        "T4_withheld": {
            "trained_mean": float(np.mean([res["T4"]["trained"]["continuation_mismatch_mean"] for res in all_seed_results])),
            "untrained_mean": float(np.mean([res["T4"]["untrained"]["continuation_mismatch_mean"] for res in all_seed_results]))
        }
    }
    
    with open(os.path.join(output_dir, "aggregate_stats.json"), "w") as f:
        json.dump(agg, f, indent=2)

    print(f"\n✅ Validation Complete. Reports in: {output_dir}")

if __name__ == "__main__":
    run_validation()
