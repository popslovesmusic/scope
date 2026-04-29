import os
import json
import numpy as np
from datetime import datetime
from native_platform.run_native_platform import run_platform
from native_platform.phase_space import compute_plv, wrap_to_pi

def generate_refracted_signal(num_frames, freq=0.1, refraction_offset=0.8):
    """Generate signal and ground truth teacher theta as 8D unit vectors."""
    t = np.arange(num_frames)
    teacher_theta_scalar = (t * freq) % (2 * np.pi)
    
    # Create 8D unit vectors (using simple sin/cos projection for demo)
    # Each channel gets a different phase offset to be non-trivial
    offsets = np.linspace(0, 2*np.pi, 8, endpoint=False)
    teacher_phi = np.zeros((num_frames, 8))
    for i in range(8):
        # We model the 8D phase vector elements
        teacher_phi[:, i] = np.cos(teacher_theta_scalar + offsets[i])
    
    # Normalize to ensure they are unit vectors
    norms = np.linalg.norm(teacher_phi, axis=1, keepdims=True)
    teacher_phi = teacher_phi / (norms + 1e-9)

    # The signal seen by the features is refracted
    # For unit vectors, refraction is a rotation or systematic bias
    # Here we just use the scalar refracted theta to generate the input wave
    refracted_theta = wrap_to_pi(teacher_theta_scalar + refraction_offset)
    input_signals = np.sin(refracted_theta)
    
    return input_signals, teacher_phi

def run_validation():
    print("🔬 Starting Patch 29 Validation: Phase Refraction Effectiveness...")
    
    np.random.seed(42)
    num_frames_train = 100
    num_frames_disc = 40
    refraction_offset = 0.8
    
    # 1. Generate Training and Disconnect Data
    train_signals, train_teacher_phi = generate_refracted_signal(num_frames_train, refraction_offset=refraction_offset)
    disc_signals, disc_teacher_phi = generate_refracted_signal(num_frames_disc, refraction_offset=refraction_offset)
    
    # Ensure fresh log files
    for rid in ["P29_VAL_TRAIN", "P29_VAL_DISC"]:
        lp = f"logs/feedback_trace_{rid}.jsonl"
        if os.path.exists(lp):
            os.remove(lp)

    # 2. Execute Training Phase (Connected)
    print("\n--- Phase 1: Training Refraction Map ---")
    train_results = run_platform(
        num_frames=num_frames_train,
        input_signals=train_signals,
        connected=True,
        connected_state="train",
        teacher_theta=train_teacher_phi,
        run_id="P29_VAL_TRAIN"
    )
    
    ref_diag = train_results["refraction_diagnostics"]
    print(f"Refraction Confidence: {ref_diag['refraction_confidence']:.4f}")
    print(f"Refraction Classification: {ref_diag['medium_classification']}")

    # 3. Execute Disconnect Phase (Recursive)
    print("\n--- Phase 2: Disconnect (Recursive Continuation) ---")
    disc_results = run_platform(
        num_frames=num_frames_disc,
        input_signals=disc_signals, 
        connected=False,
        connected_state="disconnect",
        run_id="P29_VAL_DISC"
    )

    # 4. Analyze Results
    trace_path = disc_results["feedback_trace_path"]
    student_phi = []
    
    with open(trace_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            student_phi.append(data['phi_continued'])

    student_phi = np.array(student_phi) # (time, 8)
    teacher_phi = disc_teacher_phi # (time, 8)
    
    # Ensure we compare same number of frames
    n = min(len(student_phi), len(teacher_phi))
    student_phi = student_phi[:n]
    teacher_phi = teacher_phi[:n]
    
    # Compute Observed PLV (Mean Cosine Similarity)
    plv_observed = compute_plv(student_phi, teacher_phi)
    
    # Compute Unrefracted PLV
    delta_hat = np.array(ref_diag['delta_hat'])
    student_unrefracted = np.zeros_like(student_phi)
    for i in range(8):
        student_unrefracted[:, i] = student_phi[:, i] - delta_hat[i]
    
    norms = np.linalg.norm(student_unrefracted, axis=1, keepdims=True)
    student_unrefracted = student_unrefracted / (norms + 1e-9)
    
    plv_unrefracted = compute_plv(student_unrefracted, teacher_phi)

    print(f"\nResults:")
    print(f"  Observed Frame PLV:   {plv_observed:.4f}")
    print(f"  Unrefracted Frame PLV: {plv_unrefracted:.4f}")
    print(f"  Improvement:          {plv_unrefracted - plv_observed:.4f}")

    # 5. Generate Report
    report = [
        "# PATCH 29 PHASE REFRACTION REPORT",
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Baseline Comparison (Patch 27 Baseline vs Patch 29)",
        "| Metric | Patch 27 (Static) | Patch 29 (Refracted) |",
        "| :--- | :--- | :--- |",
        f"| S1_early_PLV | 0.028 | {plv_unrefracted:.4f} |",
        "",
        "## Refraction Analysis",
        f"**Medium Classification:** {ref_diag['medium_classification']}",
        f"**Refraction Confidence:** {ref_diag['refraction_confidence']:.4f}",
        f"**Refraction Variance:** {ref_diag['refraction_variance']:.4f}",
        "",
        "### Parameters per channel (delta_hat)",
        f"Values: {ref_diag['delta_hat']}",
        "",
        "## Performance Comparison",
        "| Frame | PLV |",
        "| :--- | :--- |",
        f"| Observed (Refracted) | {plv_observed:.4f} |",
        f"| Unrefracted (Stabilized) | {plv_unrefracted:.4f} |",
        "",
        f"### Final Verdict: {'PASS' if plv_unrefracted > 0.1 else 'FAIL'}",
        f"Status: {ref_diag['medium_classification']}"
    ]
    
    with open("PATCH_29_PHASE_REFRACTION_REPORT.md", "w") as f:
        f.write("\n".join(report))
        
    print(f"\n✅ Validation complete. Report saved to PATCH_29_PHASE_REFRACTION_REPORT.md")

if __name__ == "__main__":
    run_validation()
