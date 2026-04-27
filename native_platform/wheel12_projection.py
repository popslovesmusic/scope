
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
    
    return signature, orientation_bias
