import numpy as np

from .phase_space import normalize


class PhasePredictor:
    def __init__(self):
        self.last_phi = None

    def predict_next(self, phi_current):
        phi_current = normalize(np.asarray(phi_current, dtype=float))
        if self.last_phi is None:
            self.last_phi = phi_current
            return phi_current

        delta = phi_current - self.last_phi
        phi_pred = normalize(phi_current + delta)
        self.last_phi = phi_current
        return phi_pred
