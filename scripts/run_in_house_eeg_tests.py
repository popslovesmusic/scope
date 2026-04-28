import os
import json
import numpy as np
import csv
from datetime import datetime
from native_platform.run_native_platform import run_platform
from native_platform.run_state import tail_jsonl
from native_platform.eeg_synthetic_signals import (
    generate_alpha, generate_theta, generate_alpha_with_noise,
    generate_alpha_tail_removed, generate_alpha_to_spike_burst,
    generate_dropout
)
from native_platform.eeg_feature_adapter import signal_to_input_frames

def clear_test_memory(mem_path):
    if os.path.exists(mem_path):
        os.remove(mem_path)

def get_eeg_metrics(trace_path, frames):
    records = tail_jsonl(trace_path, frames)
    if not records:
        return {}
    
    mismatches = [r.get("continuation_mismatch", 0.0) for r in records]
    groove_ids = [str(r.get("active_groove_id") or "none") for r in records]
    
    unique_grooves, counts = np.unique(groove_ids, return_counts=True)
    dominant_groove = unique_grooves[np.argmax(counts)] if len(unique_grooves) > 0 else "none"
    dominant_frac = np.max(counts) / len(groove_ids) if len(groove_ids) > 0 else 0.0
    
    decisions = [r.get("survivability_decision", "unknown") for r in records]
    reinforce_rate = np.mean([1.0 if d == "reinforce" else 0.0 for d in decisions])
    reject_rate = np.mean([1.0 if d == "reject" else 0.0 for d in decisions])
    
    last_rec = records[-1]
    
    return {
        "continuation_mismatch_mean": float(np.mean(mismatches)),
        "continuation_mismatch_std": float(np.std(mismatches)),
        "dominant_groove": dominant_groove,
        "dominant_groove_fraction": float(dominant_frac),
        "groove_switch_count": len(np.where(np.array(groove_ids)[:-1] != np.array(groove_ids)[1:])[0]),
        "reinforce_rate": float(reinforce_rate),
        "reject_rate": float(reject_rate),
        "successful_traversals": int(last_rec.get("successful_traversals", 0)),
        "trace_segment_count": int(last_rec.get("trace_segment_count", 0)),
        "signal_x_mean": float(np.mean([r.get("signal_x", 0.0) for r in records]))
    }

def run_eeg_tests():
    output_dir = "runs/in_house_eeg"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load config
    config_path = "configs/in_house_eeg_test_config.json"
    with open(config_path, 'r') as f:
        cfg = json.load(f)
    
    sr = cfg["sample_rate"]
    dur = cfg["duration_sec"]
    seeds = cfg["fixed_seeds"]
    
    all_results = []
    
    for seed in seeds:
        print(f"\n🧠 Running Rigorous EEG-like Validation with Seed: {seed}")
        np.random.seed(seed)
        mem_path = f"sessions/eeg_test_memory_seed_{seed}.json"
        clear_test_memory(mem_path)
        
        # Fresh signals per seed
        alpha_A = signal_to_input_frames(generate_alpha(sr, dur), sr, window_sec=cfg["window_sec"], overlap=cfg["overlap"])
        theta_B = signal_to_input_frames(generate_theta(sr, dur), sr, window_sec=cfg["window_sec"], overlap=cfg["overlap"])
        alpha_noise = signal_to_input_frames(generate_alpha_with_noise(sr, dur), sr, window_sec=cfg["window_sec"], overlap=cfg["overlap"])
        alpha_tail = signal_to_input_frames(generate_alpha_tail_removed(sr, dur), sr, window_sec=cfg["window_sec"], overlap=cfg["overlap"])
        alpha_spike = signal_to_input_frames(generate_alpha_to_spike_burst(sr, dur), sr, window_sec=cfg["window_sec"], overlap=cfg["overlap"])
        
        raw_alpha = generate_alpha(sr, dur)
        dropout_alpha = signal_to_input_frames(generate_dropout(raw_alpha), sr, window_sec=cfg["window_sec"], overlap=cfg["overlap"])

        seed_res = {}
        
        # T1: alpha_learning_curve
        print("  Running T1: Alpha Learning Curve...")
        t1_passes = []
        for i in range(5):
            summ = run_platform(input_signals=alpha_A, memory_path=mem_path, run_id=f"EEG_T1_S{seed}_pass{i+1}")
            t1_passes.append(get_eeg_metrics(summ["feedback_trace_path"], len(alpha_A)))
        
        summ = run_platform(input_signals=alpha_A, memory_path=mem_path, run_id=f"EEG_T1_S{seed}_replay")
        t1_passes.append(get_eeg_metrics(summ["feedback_trace_path"], len(alpha_A)))
        seed_res["T1"] = t1_passes

        # T2: theta_contrast
        print("  Running T2: Theta Contrast...")
        summ = run_platform(input_signals=theta_B, memory_path=mem_path, run_id=f"EEG_T2_S{seed}_theta")
        m_theta = get_eeg_metrics(summ["feedback_trace_path"], len(theta_B))
        summ = run_platform(input_signals=alpha_A, memory_path=mem_path, run_id=f"EEG_T2_S{seed}_alpha_return")
        m_alpha_ret = get_eeg_metrics(summ["feedback_trace_path"], len(alpha_A))
        seed_res["T2"] = {"theta": m_theta, "alpha_return": m_alpha_ret}

        # T3: alpha_tail_removed
        print("  Running T3: Alpha Tail Removed...")
        summ = run_platform(input_signals=alpha_tail, memory_path=mem_path, run_id=f"EEG_T3_S{seed}_trained")
        m_tail_tr = get_eeg_metrics(summ["feedback_trace_path"], len(alpha_tail))
        clear_test_memory(mem_path) # Baseline untrained
        summ = run_platform(input_signals=alpha_tail, memory_path=mem_path, run_id=f"EEG_T3_S{seed}_untrained")
        m_tail_ut = get_eeg_metrics(summ["feedback_trace_path"], len(alpha_tail))
        seed_res["T3"] = {"trained": m_tail_tr, "untrained": m_tail_ut}

        # T4: alpha_noise_robustness
        print("  Running T4: Noise Robustness...")
        summ = run_platform(input_signals=alpha_noise, memory_path=mem_path, run_id=f"EEG_T4_S{seed}_noise")
        seed_res["T4"] = get_eeg_metrics(summ["feedback_trace_path"], len(alpha_noise))

        # T5: alpha_to_spike_burst_transition
        print("  Running T5: Spike Transition...")
        summ = run_platform(input_signals=alpha_spike, memory_path=mem_path, run_id=f"EEG_T5_S{seed}_spike")
        seed_res["T5"] = get_eeg_metrics(summ["feedback_trace_path"], len(alpha_spike))

        # T6: artifact_rejection
        print("  Running T6: Artifact Rejection...")
        summ = run_platform(input_signals=dropout_alpha, memory_path=mem_path, run_id=f"EEG_T6_S{seed}_dropout")
        seed_res["T6"] = get_eeg_metrics(summ["feedback_trace_path"], len(dropout_alpha))
        
        all_results.append(seed_res)

    # Aggregation logic
    print("\n📊 Aggregating Results across seeds...")
    
    agg = {
        "T1_learning": {
            "P1_mismatch_mean": float(np.mean([res["T1"][0]["continuation_mismatch_mean"] for res in all_results])),
            "Replay_mismatch_mean": float(np.mean([res["T1"][-1]["continuation_mismatch_mean"] for res in all_results])),
            "Delta": float(np.mean([res["T1"][0]["continuation_mismatch_mean"] for res in all_results]) - np.mean([res["T1"][-1]["continuation_mismatch_mean"] for res in all_results]))
        },
        "T2_directionality": {
            "Theta_B_mismatch": float(np.mean([res["T2"]["theta"]["continuation_mismatch_mean"] for res in all_results])),
            "Alpha_Return_mismatch": float(np.mean([res["T2"]["alpha_return"]["continuation_mismatch_mean"] for res in all_results]))
        },
        "T3_withheld": {
            "Trained_mismatch": float(np.mean([res["T3"]["trained"]["continuation_mismatch_mean"] for res in all_results])),
            "Untrained_mismatch": float(np.mean([res["T3"]["untrained"]["continuation_mismatch_mean"] for res in all_results]))
        },
        "T6_survivability": {
            "Dropout_reject_rate": float(np.mean([res["T6"]["reject_rate"] for res in all_results]))
        }
    }
    
    with open(os.path.join(output_dir, "aggregate_stats.json"), "w") as f:
        json.dump(agg, f, indent=2)
    
    with open(os.path.join(output_dir, "comparison_report.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✅ All In-House EEG Tests Complete. Report: {output_dir}/aggregate_stats.json")

if __name__ == "__main__":
    run_eeg_tests()
