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

def wrap_to_pi(v):
    return (v + np.pi) % (2 * np.pi) - np.pi

def compute_plv(phi1, phi2):
    """
    Computes Phase Locking Value (PLV).
    If inputs are 8D unit vectors (time, channels), computes mean cosine similarity.
    If inputs are scalar angles, computes mean length of resultant vector.
    """
    phi1 = np.asarray(phi1)
    phi2 = np.asarray(phi2)
    
    # Handle 8D vector case (time, channels)
    if len(phi1.shape) >= 2 and phi1.shape[-1] == 8:
        # Compute cosine similarity for each time step
        cos_sim = np.sum(phi1 * phi2, axis=-1) # Shape: (time,)
        return float(np.mean(cos_sim))
    else:
        # Classical PLV for scalar angles
        return float(np.abs(np.mean(np.exp(1j * (phi1 - phi2)))))

def compute_cosine_similarity(phi1, phi2):
    dot = np.sum(phi1 * phi2, axis=-1)
    return dot
