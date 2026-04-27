import numpy as np

def normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v

def compute_phase_vector(W, C, E, V):
    # W: 3-vector, V: 3-vector
    # Concatenate all metrics into a single state vector
    phase = np.concatenate([W, [C, E], V])
    return normalize(phase)

def phase_mismatch(phi1, phi2):
    # 1 - cosine similarity
    dot = np.dot(phi1, phi2)
    return 1.0 - dot
