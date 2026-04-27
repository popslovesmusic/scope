
import numpy as np

def project_to_12(W, C, E, V):
    """
    Maps W, C, E into a 12-slot signature.
    V is used to bias orientation.
    """
    signature = np.zeros(12)
    
    # 1. Map W into 3 quadrants of 4 slots each
    # Quadrant 1 (0-3): mean_energy
    # Quadrant 2 (4-7): gradient
    # Quadrant 3 (8-11): variance
    
    for i in range(3):
        base_idx = i * 4
        val = W[i]
        for j in range(4):
            # Spread value across the quadrant based on internal weight
            dist = abs(j/3.0 - 0.5) # simple spread
            signature[base_idx + j] = val * (1.0 - dist)

    # 2. Apply C (Coupling) and E (Imbalance) as global modulators
    signature *= (1.0 + C - E)
    
    # 3. Encode orientation bias from V
    # Inject ONLY into signature (input space)
    orientation_bias = np.tanh(np.mean(V))
    signature += orientation_bias * 0.1
    
    # Patch 11: Clamp to valid range
    signature = np.nan_to_num(signature, nan=0.0, posinf=1.0, neginf=0.0)
    signature = np.clip(signature, 0.0, 1.0)

    return signature, orientation_bias

def apply_operator_pressure(signature, pressure):
    """
    Shapes the 12-wheel signature based on operator pressure.
    Does not set the v14 operator directly; biases the input space.
    """
    # Map operator sectors to 12-wheel indices
    sector_map = {
        '++': [0, 1, 2],
        '--': [3, 4, 5],
        '+-': [6, 7, 8],
        '-+': [9, 10, 11]
    }
    
    sig_new = signature.copy()
    for op, idxs in sector_map.items():
        # Apply brightening or damping based on pressure
        # Max gain of 15% per sector
        p_val = pressure.get(op, 0.0)
        sig_new[idxs] *= (1.0 + p_val * 0.15)
        
    # Final clamp to stay within physical bounds
    sig_new = np.clip(sig_new, 0.0, 1.0)
    return sig_new
