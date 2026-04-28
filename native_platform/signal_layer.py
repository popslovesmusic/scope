import numpy as np

def compute_x_channel(W_local, W_global):
    """
    Computes X channel as cross-view consistency / projection-loss proxy.
    X near 1.0 = high consistency.
    X near 0.0 = high disagreement / degeneracy.
    """
    # Use Euclidean distance between views
    dist = np.linalg.norm(W_local - W_global)
    # Map distance to consistency [0, 1]
    # Recalibrated Patch 23: more permissive decay
    x = np.exp(-0.25 * dist)
    return float(x)

def get_consistency_level(x):
    if x > 0.75: return "high"
    if x > 0.45: return "moderate"
    return "low"
