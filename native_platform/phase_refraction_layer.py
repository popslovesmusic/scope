import numpy as np
from .phase_space import normalize

def wrap_to_pi(v):
    return (v + np.pi) % (2 * np.pi) - np.pi

class BandRefractionModel:
    def __init__(self, band_id, channels=8):
        self.band_id = band_id
        self.channels = channels
        
        # Parameters to learn
        self.delta_hat = np.zeros(channels) # constant phase offset
        self.eta_hat = np.ones(channels)   # velocity scale
        
        # Fourier coefficients for nonlinear warp (order 1)
        self.a1 = np.zeros(channels)
        self.b1 = np.zeros(channels)
        
        # Statistics for diagnostics
        self.delta_history = [] # list of arrays
        self.eta_history = []
        self.variance = 1.0
        self.confidence = 0.0
        self.drift_rate = 0.0

    def learn_step(self, theta_teacher, theta_obs, omega_teacher, omega_obs, signal_x):
        """Estimate refraction during training."""
        delta = wrap_to_pi(theta_obs - theta_teacher)
        self.delta_history.append(delta)
        
        # Velocity scale estimation (safe ratio)
        mask = np.abs(omega_teacher) > 0.005
        if np.any(mask):
            eta = np.ones(self.channels)
            eta[mask] = omega_obs[mask] / (omega_teacher[mask] + 1e-9)
            self.eta_history.append(eta)
            
        if len(self.delta_history) > 100:
            self.delta_history.pop(0)
        if len(self.eta_history) > 100:
            self.eta_history.pop(0)
            
        # Update estimates (Circular Mean for delta)
        history_arr = np.array(self.delta_history)
        c_mean = np.angle(np.mean(np.exp(1j * history_arr), axis=0))
        self.delta_hat = c_mean
        
        # Circular variance = 1 - |mean_resultant_vector|
        c_var = 1.0 - np.abs(np.mean(np.exp(1j * history_arr), axis=0))
        self.variance = float(np.mean(c_var))
        
        if self.eta_history:
            self.eta_hat = np.median(np.array(self.eta_history), axis=0)
            
        # Refraction confidence
        self.confidence = signal_x * (1.0 - self.variance)
        
        # Estimate drift rate (rough)
        if len(self.delta_history) > 20:
            early = np.angle(np.mean(np.exp(1j * history_arr[:10]), axis=0))
            late = np.angle(np.mean(np.exp(1j * history_arr[-10:]), axis=0))
            self.drift_rate = float(np.mean(np.abs(wrap_to_pi(late - early))))

    def apply_inverse(self, theta_obs):
        """Map observed/student phase back to unrefracted frame."""
        return wrap_to_pi(theta_obs - self.delta_hat)

    def apply_forward(self, theta_true):
        """Map true/model phase to observed frame."""
        return wrap_to_pi(theta_true + self.delta_hat)

    def classify_medium(self):
        if self.variance <= 0.15 and self.drift_rate <= 0.05:
            return "constant_medium"
        if self.variance <= 0.35:
            return "weakly_drifting_medium"
        return "unstable_medium"

class PhaseRefractionLayer:
    def __init__(self, channels=8):
        self.bands = {
            "theta": BandRefractionModel("theta", channels),
            "alpha": BandRefractionModel("alpha", channels),
            "beta": BandRefractionModel("beta", channels),
            "mixed": BandRefractionModel("mixed", channels)
        }
        self.active_band_id = "alpha"

    def update_train(self, theta_teacher, theta_obs, omega_teacher, omega_obs, signal_x):
        band = self.bands[self.active_band_id]
        band.learn_step(theta_teacher, theta_obs, omega_teacher, omega_obs, signal_x)

    def unrefract(self, theta_obs):
        return self.bands[self.active_band_id].apply_inverse(theta_obs)

    def refract(self, theta_unrefracted):
        return self.bands[self.active_band_id].apply_forward(theta_unrefracted)

    def get_diagnostics(self):
        band = self.bands[self.active_band_id]
        return {
            "band_id": self.active_band_id,
            "refraction_confidence": band.confidence,
            "refraction_variance": band.variance,
            "refraction_drift_rate": band.drift_rate,
            "medium_classification": band.classify_medium(),
            "delta_hat": band.delta_hat.tolist(),
            "eta_hat": band.eta_hat.tolist()
        }
