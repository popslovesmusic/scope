
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
