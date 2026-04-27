import numpy as np

def normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v

def compute_phase_vector(W, C, E, V):
    # W: 3-vector, V: 3-vector
    W = np.asarray(W, dtype=float)
    V = np.asarray(V, dtype=float)
    # Patch 16: Weight elements to increase sensitivity to change
    phase = np.array([
        2.0 * W[0],
        2.0 * W[1],
        2.0 * W[2],
        1.5 * C,
        1.5 * E,
        2.0 * V[0],
        2.0 * V[1],
        2.0 * V[2]
    ], dtype=float)
    return normalize(phase)

def phase_mismatch(phi1, phi2):
    # 1 - cosine similarity
    dot = float(np.dot(phi1, phi2))
    # Clip to prevent float precision errors
    dot = max(-1.0, min(1.0, dot))
    return 1.0 - dot
