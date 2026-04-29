import numpy as np
from .phase_space import normalize

def wrap_to_pi(v):
    return (v + np.pi) % (2 * np.pi) - np.pi

class BandRefractionModel:
    def __init__(self, band_id, channels=8):
        self.band_id = band_id
        self.channels = channels
        
        # Static parameters (learned in Phase 1)
        self.delta_hat = np.zeros(channels) # constant phase offset
        self.eta_hat = np.ones(channels)   # velocity scale
        
        # Adaptive parameters (tracked in Phase 2/3)
        self.adaptive_delta_hat = np.zeros(channels)
        self.alpha_drift = 0.05 # slow drift estimator rate
        
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
        # We use the resultant vector to update static estimate
        resultant = np.mean(np.exp(1j * history_arr), axis=0)
        self.delta_hat = np.angle(resultant)
        
        # Circular variance = 1 - |mean_resultant_vector|
        c_var = 1.0 - np.abs(resultant)
        self.variance = float(np.mean(c_var))
        
        if self.eta_history:
            self.eta_hat = np.median(np.array(self.eta_history), axis=0)
            
        # Refraction confidence
        self.confidence = signal_x * (1.0 - self.variance)
        
        # Initial adaptive state matches static
        self.adaptive_delta_hat = self.delta_hat.copy()
        
        # Estimate drift rate
        if len(self.delta_history) > 20:
            early = np.angle(np.mean(np.exp(1j * history_arr[:10]), axis=0))
            late = np.angle(np.mean(np.exp(1j * history_arr[-10:]), axis=0))
            self.drift_rate = float(np.mean(np.abs(wrap_to_pi(late - early))))

    def update_adaptive(self, theta_teacher, theta_obs, signal_x):
        """Slowly update adaptive delta_hat when connected, based on confidence."""
        if signal_x < 0.3: return # Skip if signal is too weak/inconsistent
        
        # delta = wrap(obs - teacher)
        delta_current = wrap_to_pi(theta_obs - theta_teacher)
        
        # Confidence-weighted update
        weight = self.alpha_drift * signal_x
        
        # Update using complex exponential for circular consistency
        z_adaptive = np.exp(1j * self.adaptive_delta_hat)
        z_current = np.exp(1j * delta_current)
        
        z_next = (1.0 - weight) * z_adaptive + weight * z_current
        self.adaptive_delta_hat = np.angle(z_next)

    def apply_inverse(self, theta_obs, adaptive=False):
        """Map observed/student phase back to unrefracted frame."""
        offset = self.adaptive_delta_hat if adaptive else self.delta_hat
        return wrap_to_pi(theta_obs - offset)

    def apply_forward(self, theta_true, adaptive=False):
        """Map true/model phase to observed frame."""
        offset = self.adaptive_delta_hat if adaptive else self.delta_hat
        return wrap_to_pi(theta_true + offset)

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
        self.use_adaptive = True

    def update_train(self, theta_teacher, theta_obs, omega_teacher, omega_obs, signal_x):
        band = self.bands[self.active_band_id]
        band.learn_step(theta_teacher, theta_obs, omega_teacher, omega_obs, signal_x)

    def update_adaptive(self, theta_teacher, theta_obs, signal_x):
        """Called only when connected but after initial training."""
        band = self.bands[self.active_band_id]
        band.update_adaptive(theta_teacher, theta_obs, signal_x)

    def unrefract(self, theta_obs):
        return self.bands[self.active_band_id].apply_inverse(theta_obs, adaptive=self.use_adaptive)

    def refract(self, theta_unrefracted):
        return self.bands[self.active_band_id].apply_forward(theta_unrefracted, adaptive=self.use_adaptive)

    def get_diagnostics(self):
        band = self.bands[self.active_band_id]
        return {
            "band_id": self.active_band_id,
            "refraction_confidence": band.confidence,
            "refraction_variance": band.variance,
            "refraction_drift_rate": band.drift_rate,
            "medium_classification": band.classify_medium(),
            "delta_hat": band.delta_hat.tolist(),
            "adaptive_delta_hat": band.adaptive_delta_hat.tolist(),
            "eta_hat": band.eta_hat.tolist(),
            "use_adaptive": self.use_adaptive
        }
