import os
import json
import numpy as np
import shutil
from datetime import datetime
from native_platform.run_native_platform import run_platform
from native_platform.run_state import clear_memory, tail_jsonl
from core.memory_layer import load_memory_state

def generate_sine(frequency, amplitude, frames):
    t = np.linspace(0, frames/10.0, frames)
    return amplitude * np.sin(2 * np.pi * frequency * t)

def generate_noise(std, frames):
    return np.random.normal(0, std, frames)

def get_metrics(trace_path, frames):
    records = tail_jsonl(trace_path, frames)
    if not records:
        return {}
    mismatches = [r.get("continuation_mismatch", 0.0) for r in records]
    groove_ids = [str(r.get("active_groove_id") or "none") for r in records]
    
    unique_grooves, counts = np.unique(groove_ids, return_counts=True)
    dominant_groove = unique_grooves[np.argmax(counts)] if len(unique_grooves) > 0 else "none"
    dominant_frac = np.max(counts) / len(groove_ids) if len(groove_ids) > 0 else 0.0
    
    return {
        "continuation_mismatch_mean": float(np.mean(mismatches)),
        "continuation_mismatch_max": float(np.max(mismatches)),
        "dominant_groove": dominant_groove,
        "dominant_frac": float(dominant_frac),
        "groove_switch_count": len(np.where(np.array(groove_ids)[:-1] != np.array(groove_ids)[1:])[0])
    }

def clear_test_memory(mem_path="sessions/test_memory.json"):
    if os.path.exists(mem_path):
        os.remove(mem_path)

def run_rigorous_tests():
    output_dir = "runs/rigor"
    os.makedirs(output_dir, exist_ok=True)
    mem_path = "sessions/test_memory.json"
    
    frames = 500
    wave_A = generate_sine(5.0, 1.0, frames)
    wave_B = generate_sine(12.0, 1.0, frames)
    
    results = {}
    
    # --- T1: Repeated A learning curve ---
    print("\n=== T1: Repeated A Learning Curve ===")
    clear_test_memory(mem_path)
    t1_metrics = []
    for i in range(5):
        summ = run_platform(input_signals=wave_A, memory_path=mem_path, run_id=f"T1_A_pass{i+1}")
        m = get_metrics(summ["feedback_trace_path"], frames)
        print(f"Pass {i+1} Mean Mismatch: {m['continuation_mismatch_mean']:.4f} (Groove: {m['dominant_groove']})")
        t1_metrics.append(m)
        
    summ = run_platform(input_signals=wave_A, memory_path=mem_path, run_id=f"T1_replay_A")
    m_replay = get_metrics(summ["feedback_trace_path"], frames)
    print(f"Replay Mean Mismatch: {m_replay['continuation_mismatch_mean']:.4f}")
    t1_metrics.append(m_replay)
    results["T1"] = t1_metrics

    # --- T2: A vs B directionality ---
    print("\n=== T2: A vs B Directionality ===")
    summ = run_platform(input_signals=wave_B, memory_path=mem_path, run_id=f"T2_run_B")
    m_B = get_metrics(summ["feedback_trace_path"], frames)
    print(f"Run B Mean Mismatch: {m_B['continuation_mismatch_mean']:.4f} (Groove: {m_B['dominant_groove']})")
    
    summ = run_platform(input_signals=wave_A, memory_path=mem_path, run_id=f"T2_run_A_after_B")
    m_A_after = get_metrics(summ["feedback_trace_path"], frames)
    print(f"Run A After B Mean Mismatch: {m_A_after['continuation_mismatch_mean']:.4f} (Groove: {m_A_after['dominant_groove']})")
    results["T2"] = {"B": m_B, "A_after_B": m_A_after}

    # --- T3: Catastrophic forgetting ---
    print("\n=== T3: Catastrophic Forgetting ===")
    clear_test_memory(mem_path)
    for i in range(5): run_platform(input_signals=wave_A, memory_path=mem_path, run_id=f"T3_A_pass{i+1}")
    for i in range(5): run_platform(input_signals=wave_B, memory_path=mem_path, run_id=f"T3_B_pass{i+1}")
    
    summ = run_platform(input_signals=wave_A, memory_path=mem_path, run_id=f"T3_replay_A")
    m_A_forget = get_metrics(summ["feedback_trace_path"], frames)
    print(f"Replay A after B training Mean Mismatch: {m_A_forget['continuation_mismatch_mean']:.4f} (Groove: {m_A_forget['dominant_groove']})")
    results["T3"] = {"replay_A": m_A_forget}

    # --- T4: Withheld continuation ---
    print("\n=== T4: Withheld Continuation ===")
    clear_test_memory(mem_path)
    for i in range(5): run_platform(input_signals=wave_A, memory_path=mem_path, run_id=f"T4_A_pass{i+1}")
    
    wave_A_masked = wave_A.copy()
    wave_A_masked[int(frames*0.6):] = 0.0
    
    summ = run_platform(input_signals=wave_A_masked, memory_path=mem_path, run_id=f"T4_A_masked_trained")
    m_A_masked_trained = get_metrics(summ["feedback_trace_path"], frames)
    print(f"Trained Masked A Mean Mismatch: {m_A_masked_trained['continuation_mismatch_mean']:.4f}")
    
    clear_test_memory(mem_path)
    summ = run_platform(input_signals=wave_A_masked, memory_path=mem_path, run_id=f"T4_A_masked_untrained")
    m_A_masked_untrained = get_metrics(summ["feedback_trace_path"], frames)
    print(f"Untrained Masked A Mean Mismatch: {m_A_masked_untrained['continuation_mismatch_mean']:.4f}")
    results["T4"] = {"trained": m_A_masked_trained, "untrained": m_A_masked_untrained}

    # --- Save Report ---
    report_path = os.path.join(output_dir, "comparison_report.json")
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"\n✅ Rigorous Tests Complete. Report: {report_path}")

if __name__ == "__main__":
    run_rigorous_tests()
