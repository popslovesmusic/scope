import numpy as np
from .phase_space import normalize

def derive_residue_phase_bias(residue):
    if residue is None or not getattr(residue, 'is_committed', False):
        return np.zeros(8)
    
    # If the residue was committed, it represents a stable attractor.
    # We pull the prediction slightly toward the state that was validated as stable.
    # We expect 'phi' to be attached to the residue in the main loop.
    phi = getattr(residue, 'phi', None)
    if phi is not None:
        return 0.1 * np.array(phi)
    return np.zeros(8)

class ResiduePhasePredictor:
    def __init__(self, history_size=32):
        self.history = []
        self.history_size = history_size

    def update_history(self, phi):
        self.history.append(phi)
        if len(self.history) > self.history_size:
            self.history.pop(0)

    def predict_next(self, phi_current, residue=None):
        if not self.history:
            self.update_history(phi_current)
            return phi_current
        
        previous_phi = self.history[-1]
        
        # 1. Momentum: project forward along last direction
        trend = phi_current - previous_phi
        
        # 2. Residue Pull: bias toward committed stable states
        residue_pull = derive_residue_phase_bias(residue)
        
        # 3. Combine
        phi_pred = phi_current + trend + residue_pull
        
        # 4. Normalize to stay on hypersphere
        phi_pred = normalize(phi_pred)
        
        self.update_history(phi_current)
        return phi_pred
