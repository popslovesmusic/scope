import os
import json
import numpy as np
from datetime import datetime
from native_platform.engine_bridge import EngineBridge

def run_audit():
    print("🔬 Starting C++ Engine Semantic Correctness Audit (Patch 28)...")
    
    num_nodes = 256
    bridge = EngineBridge(num_nodes=num_nodes)
    
    report_lines = [
        "# CPP Engine Semantic Audit Report",
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Engine Status: {'NATIVE' if bridge.engine else 'DUMMY (FALLBACK)'}",
        f"Node Count: {num_nodes}",
        ""
    ]
    
    results = {}

    # 1. State Delta Test
    print("  Testing State Delta...")
    bridge.set_integrator_state(1.0)
    prev = bridge.get_node_outputs().copy()
    bridge.step(input_signal=1.0, control_pattern=1.0)
    current = bridge.get_node_outputs()
    stats = bridge.get_field_statistics(previous_outputs=prev)
    
    results["state_delta_test"] = "PASS" if stats["state_delta"] > 1e-7 else "FAIL"
    print(f"    Delta: {stats['state_delta']:.9f} -> {results['state_delta_test']}")

    # 2. Frozen Field Test
    print("  Testing for Frozen Field...")
    # Evolve for 100 steps with changing input
    means = []
    for t in range(100):
        bridge.step(input_signal=np.sin(t*0.1), control_pattern=1.0)
        s = bridge.get_field_statistics()
        means.append(s["mean"])
    
    # Check if mean is non-constant
    mean_change = np.std(means)
    results["frozen_field_test"] = "PASS" if mean_change > 1e-9 else "FAIL"
    print(f"    Mean StdDev: {mean_change:.9f} -> {results['frozen_field_test']}")

    # 3. Reaction Term Toggle Test
    print("  Testing Reaction Toggle...")
    bridge.set_integrator_state(5.0)
    bridge.set_reaction_enabled(True)
    out_with = bridge.step(input_signal=0.1, control_pattern=1.0)
    
    bridge.set_integrator_state(5.0)
    bridge.set_reaction_enabled(False)
    out_without = bridge.step(input_signal=0.1, control_pattern=1.0)
    
    diff = abs(out_with - out_without)
    results["reaction_toggle_test"] = "PASS" if diff > 1e-9 else "FAIL"
    print(f"    Toggle Diff: {diff:.9f} -> {results['reaction_toggle_test']}")

    # 4. Corridor Gate Test
    print("  Testing Corridor Gate...")
    bridge.set_integrator_state(10.0) # Should be gated if corridor enabled
    bridge.set_corridor_enabled(True)
    out_gated = bridge.step(input_signal=0.0, control_pattern=1.0)
    
    bridge.set_integrator_state(10.0)
    bridge.set_corridor_enabled(False)
    out_ungated = bridge.step(input_signal=0.0, control_pattern=1.0)
    
    gate_diff = abs(out_gated - out_ungated)
    results["corridor_gate_test"] = "PASS" if gate_diff > 1e-5 else "FAIL"
    print(f"    Gate Diff: {gate_diff:.9f} -> {results['corridor_gate_test']}")

    # 5. Scalar vs AVX2 (only if native)
    if bridge.engine:
        print("  Testing Scalar vs AVX2 consistency...")
        bridge.set_integrator_state(1.0)
        out_avx = bridge.step(0.5, 0.5)
        bridge.set_integrator_state(1.0)
        out_scalar = bridge.step_scalar(0.5, 0.5)
        simd_diff = abs(out_avx - out_scalar)
        results["simd_consistency_test"] = "PASS" if simd_diff < 1e-7 else "FAIL"
        print(f"    SIMD Diff: {simd_diff:.9f} -> {results['simd_consistency_test']}")
    else:
        results["simd_consistency_test"] = "SKIPPED (Dummy Mode)"

    # Final Summary
    report_lines.append("## Test Results")
    report_lines.append("| Test | Result |")
    report_lines.append("| :--- | :--- |")
    for k, v in results.items():
        report_lines.append(f"| {k} | {v} |")
    
    final_verdict = "VALID" if all(v in ["PASS", "SKIPPED (Dummy Mode)"] for v in results.values()) else "INVALID"
    report_lines.append(f"\n### Final Verdict: **{final_verdict}**")
    
    report_path = "CPP_ENGINE_SEMANTIC_AUDIT_REPORT.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
        
    print(f"\n✅ Audit Complete. Verdict: {final_verdict}. Report: {report_path}")

if __name__ == "__main__":
    run_audit()
