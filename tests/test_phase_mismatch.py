import unittest
import numpy as np
from native_platform.phase_space import compute_phase_vector, phase_mismatch
from native_platform.phase_predictor import PhasePredictor

class TestPhaseMismatch(unittest.TestCase):
    def test_phase_vector_norm(self):
        W = np.array([0.5, 0.3, 0.2])
        V = np.array([0.1, -0.1, 0.0])
        phi = compute_phase_vector(W, 0.8, 0.2, V)
        norm = np.linalg.norm(phi)
        self.assertAlmostEqual(norm, 1.0, places=6)

    def test_phase_mismatch_range(self):
        phi1 = np.array([1, 0, 0, 0, 0, 0, 0, 0])
        phi2 = np.array([-1, 0, 0, 0, 0, 0, 0, 0]) # Opposite
        phi3 = np.array([1, 0, 0, 0, 0, 0, 0, 0])  # Same
        
        self.assertAlmostEqual(phase_mismatch(phi1, phi3), 0.0)
        self.assertAlmostEqual(phase_mismatch(phi1, phi2), 2.0)

    def test_predictor_stable(self):
        predictor = PhasePredictor()
        phi1 = np.array([1, 0, 0, 0, 0, 0, 0, 0])
        phi2 = np.array([1, 0, 0, 0, 0, 0, 0, 0])
        
        predictor.predict_next(phi1)
        phi_pred = predictor.predict_next(phi2)
        
        # If input is stable, prediction should be identical
        self.assertAlmostEqual(phase_mismatch(phi_pred, phi2), 0.0)

    def test_predictor_momentum(self):
        predictor = PhasePredictor()
        phi1 = np.array([0, 1, 0, 0, 0, 0, 0, 0])
        phi2 = np.array([0.1, 0.9, 0, 0, 0, 0, 0, 0]) # Moving toward x (8 elements)
        phi2 = phi2 / np.linalg.norm(phi2)
        
        predictor.predict_next(phi1)
        phi_pred = predictor.predict_next(phi2)
        
        # Prediction should have continued in the same direction
        # so it should be even 'further' from phi1 than phi2 was
        dist1 = phase_mismatch(phi1, phi2)
        dist2 = phase_mismatch(phi1, phi_pred)
        self.assertGreater(dist2, dist1)

if __name__ == '__main__':
    unittest.main()
