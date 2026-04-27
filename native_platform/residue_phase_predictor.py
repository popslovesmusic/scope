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
        if len(self.history) < 2:
            self.update_history(phi_current)
            return phi_current

        # multi-step trajectory estimate
        phi_prev = self.history[-1]
        phi_prev2 = self.history[-2]

        v1 = phi_current - phi_prev
        v2 = phi_prev - phi_prev2

        # acceleration term (captures curvature)
        accel = v1 - v2

        # residue-driven pull (re-integrated from v1)
        residue_pull = derive_residue_phase_bias(residue)

        # forward projection (THIS is key)
        phi_pred = phi_current + v1 + 0.5 * accel + residue_pull

        # push prediction ahead (tunable)
        phi_pred = phi_current + 1.5 * (phi_pred - phi_current)

        # normalize
        phi_pred = phi_pred / (np.linalg.norm(phi_pred) + 1e-9)

        self.update_history(phi_current)

        return phi_pred
