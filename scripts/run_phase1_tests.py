import os
import json
import numpy as np
import shutil
from datetime import datetime
from native_platform.run_native_platform import run_platform
from native_platform.run_state import clear_memory, tail_jsonl

def generate_sine(frequency, amplitude, frames):
    t = np.linspace(0, frames/10.0, frames) # 10Hz sample rate for these tests
    return amplitude * np.sin(2 * np.pi * frequency * t)

def generate_sine_with_noise(frequency, amplitude, noise_std, frames):
    base = generate_sine(frequency, amplitude, frames)
    noise = np.random.normal(0, noise_std, frames)
    return base + noise

def execute_tests(config_path="config/test1.json"):
    with open(config_path, 'r') as f:
        test_config = json.load(f)
    
    output_base_dir = test_config["setup"]["output_dir"]
    os.makedirs(output_base_dir, exist_ok=True)
    
    generated_waves = {}
    
    # 1. Prepare Wave Generators
    for name, cfg in test_config["wave_generators"].items():
        w_type = cfg["type"]
        if w_type == "sine":
            generated_waves[name] = generate_sine(cfg["frequency"], cfg["amplitude"], cfg["duration_frames"])
        elif w_type == "sine_with_noise":
            generated_waves[name] = generate_sine_with_noise(cfg["frequency"], cfg.get("amplitude", 1.0), cfg["noise_std"], cfg["duration_frames"])
        elif w_type == "reuse_previous":
            generated_waves[name] = generated_waves[cfg["source"]]
        elif w_type == "truncate":
            source = generated_waves[cfg["source"]]
            keep = int(len(source) * cfg["keep_ratio"])
            generated_waves[name] = source[:keep]
        elif w_type == "composite":
            parts = []
            for seg in cfg["segments"]:
                if "generator" in seg:
                    if seg["generator"] == "sine_A":
                        parts.append(generate_sine(5.0, 1.0, seg["frames"]))
                    elif seg["generator"] == "sine_with_noise":
                        parts.append(generate_sine_with_noise(seg["frequency"], 1.0, seg["noise_std"], seg["frames"]))
            generated_waves[name] = np.concatenate(parts)

    results = []
    
    # 2. Execute Runs
    for run in test_config["runs"]:
        run_name = run["name"]
        print(f"\n▶️ Starting Run: {run_name}")
        
        if run.get("reset_memory", False):
            print("  Resetting native memory...")
            clear_memory()
            
        input_wave = generated_waves[run["input"]]
        
        # Override config for test
        engine_steps = test_config["setup"]["config_overrides"]["feedback.engine_steps_per_frame"]
        
        summary = run_platform(
            num_frames=len(input_wave),
            input_signals=input_wave,
            engine_steps_per_frame=engine_steps,
            feedback_enabled=True,
            run_id=f"phase1_{run_name}"
        )
        
        # 3. Collect Metrics
        trace_path = summary["feedback_trace_path"]
        records = tail_jsonl(trace_path, len(input_wave))
        
        # Compute summary metrics using new continuation keys
        mismatches = [r.get("continuation_mismatch", 0.0) for r in records]
        mismatches_next = [r.get("continuation_mismatch_next", 0.0) for r in records]
        cautions = [r["caution"] for r in records]
        recoveries = [r["recovery"] for r in records]
        commits = [1 if r["residue_committed"] else 0 for r in records]
        groove_sizes = [r.get("trace_groove_size", 0) for r in records]
        
        run_summary = {
            "name": run_name,
            "continuation_mismatch_mean": float(np.mean(mismatches)),
            "continuation_mismatch_max": float(np.max(mismatches)),
            "continuation_mismatch_next_mean": float(np.mean(mismatches_next)),
            "caution_mean": float(np.mean(cautions)),
            "recovery_mean": float(np.mean(recoveries)),
            "residue_commit_rate": float(np.sum(commits) / len(commits)),
            "final_groove_size": int(groove_sizes[-1]) if groove_sizes else 0,
            "trace_path": trace_path
        }
        
        # Save run summary
        run_out_dir = os.path.join(output_base_dir, run_name)
        os.makedirs(run_out_dir, exist_ok=True)
        with open(os.path.join(run_out_dir, "summary.json"), "w") as sf:
            json.dump(run_summary, sf, indent=2)
            
        shutil.copy2(trace_path, os.path.join(run_out_dir, "feedback_trace.jsonl"))
        
        results.append(run_summary)
        print(f"✅ Run Complete. Mean Mismatch: {run_summary['continuation_mismatch_mean']:.4f} Groove: {run_summary['final_groove_size']}")

    # 4. Generate Comparison Report
    report_path = os.path.join(output_base_dir, "comparison_report.json")
    with open(report_path, "w") as rf:
        json.dump(results, rf, indent=2)
        
    print(f"\n📊 Phase 1 Tests (Continuation Refactor) Complete. Report: {report_path}")

if __name__ == "__main__":
    execute_tests()
