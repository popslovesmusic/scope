import numpy as np

class PhasePredictor:
    def __init__(self):
        self.last_phi = None

    def predict_next(self, phi_current):
        if self.last_phi is None:
            self.last_phi = phi_current
            return phi_current

        # simple continuation: project forward along last direction
        # delta represents the 'momentum' of the state change
        delta = phi_current - self.last_phi
        phi_pred = phi_current + delta

        # normalize to keep bounded on the hypersphere
        phi_pred = phi_pred / (np.linalg.norm(phi_pred) + 1e-9)

        self.last_phi = phi_current
        return phi_pred
