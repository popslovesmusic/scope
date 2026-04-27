
import numpy as np

def compute_W(node_outputs):
    """
    Computes W = [mean_energy, gradient, variance] normalized to sum=1.
    """
    if len(node_outputs) == 0:
        return np.array([1/3, 1/3, 1/3])
    
    outputs = np.array(node_outputs)
    mean_energy = np.mean(np.abs(outputs))
    variance = np.var(outputs)
    
    if len(outputs) > 1:
        gradient = np.mean(np.abs(np.diff(outputs)))
    else:
        gradient = 0.0
        
    W_raw = np.array([mean_energy, gradient, variance])
    
    # Normalize to sum=1
    S = np.sum(W_raw)
    if S > 1e-9:
        return W_raw / S
    return np.array([1/3, 1/3, 1/3])

def compute_metrics(W_prev, W):
    """
    Computes C (coupling), E (imbalance), V (velocity).
    """
    W = np.array(W)
    W_prev = np.array(W_prev)
    
    # C = min(W)/max(W)
    C = np.min(W) / (np.max(W) + 1e-12)
    
    # E = distance from [1/3, 1/3, 1/3]
    E = float(np.linalg.norm(W - np.array([1/3, 1/3, 1/3])))
    
    # V = W - W_prev
    V = W - W_prev
    
    return float(C), float(E), V

def compute_regional_W(node_outputs, regions=3):
    """
    Computes participation weights across spatial regions to avoid premature collapse.
    """
    outputs = np.asarray(node_outputs, dtype=float)
    if outputs.size == 0:
        return np.array([1/3, 1/3, 1/3])
    
    # Split the field into chunks
    chunks = np.array_split(outputs, regions)
    vals = []
    for chunk in chunks[:3]: # Ensure we only take first 3 regions for W1, W2, W3
        # Use mean energy of each region as its participation component
        vals.append(compute_W(chunk)[0])
        
    W_raw = np.array(vals, dtype=float)
    
    # Normalize to sum=1
    S = np.sum(W_raw)
    if S > 1e-9:
        return W_raw / S
    return np.array([1/3, 1/3, 1/3])
