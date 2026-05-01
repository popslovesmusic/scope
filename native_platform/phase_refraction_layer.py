import numpy as np
from .phase_space import normalize

def wrap_to_pi(v):
    return (v + np.pi) % (2 * np.pi) - np.pi

class BandRefractionModel:
    def __init__(self, band_id, channels=8):
        self.band_id = band_id
        self.channels = channels
        # For 8 channels, we treat them as 4 complex pairs for rotation-based refraction
        self.num_pairs = channels // 2
        
        # Static parameters
        self.delta_hat = np.zeros(self.num_pairs) # phase offset per pair
        
        # Adaptive parameters
        self.adaptive_delta_hat = np.zeros(self.num_pairs)
        self.alpha_drift = 0.05 # slow drift estimator rate
        
        # Statistics for diagnostics
        self.delta_history = [] 
        self.variance = 1.0
        self.confidence = 0.0
        self.drift_rate = 0.0
        self.drift_vector = np.zeros(self.num_pairs)
        self.steps_since_adaptive_update = 0

    def _to_complex(self, v):
        """Map 8D real vector to 4D complex vector."""
        return v[0::2] + 1j * v[1::2]

    def _to_real(self, c):
        """Map 4D complex vector back to 8D real vector."""
        res = np.zeros(self.channels)
        res[0::2] = c.real
        res[1::2] = c.imag
        return res

    def learn_step(self, phi_teacher, phi_obs, omega_teacher, omega_obs, signal_x):
        """Estimate refraction using complex phase difference across 4 planes."""
        if signal_x < 0.2: return
        
        c_teacher = self._to_complex(phi_teacher)
        c_obs = self._to_complex(phi_obs)
        
        # delta = angle(obs) - angle(teacher) for each pair
        delta = wrap_to_pi(np.angle(c_obs + 1e-12) - np.angle(c_teacher + 1e-12))
        
        # Store delta and confidence
        self.delta_history.append((delta, signal_x))
        
        if len(self.delta_history) > 200:
            self.delta_history.pop(0)
            
        # Update estimates (Weighted Circular Mean per pair)
        hist = self.delta_history
        deltas = np.array([h[0] for h in hist])
        weights = np.array([h[1] for h in hist])
        
        # Weighted resultant vector
        resultant = np.average(np.exp(1j * deltas), weights=weights, axis=0)
        self.delta_hat = np.angle(resultant)
        
        # Circular variance = 1 - |mean_resultant_vector|
        c_var = 1.0 - np.abs(resultant)
        self.variance = float(np.mean(c_var))
        
        # Refraction confidence
        self.confidence = signal_x * (1.0 - self.variance)
        
        # Initial adaptive state matches static
        self.adaptive_delta_hat = self.delta_hat.copy()
        
        # Estimate drift rate (Signed Vector)
        if len(self.delta_history) > 40:
            early_d = np.array([h[0] for h in hist[:20]])
            early_w = np.array([h[1] for h in hist[:20]])
            late_d = np.array([h[0] for h in hist[-20:]])
            late_w = np.array([h[1] for h in hist[-20:]])
            
            early = np.angle(np.average(np.exp(1j * early_d), weights=early_w, axis=0))
            late = np.angle(np.average(np.exp(1j * late_d), weights=late_w, axis=0))
            
            # Signed drift per step over the window (approx 40 steps)
            self.drift_vector = wrap_to_pi(late - early) / 40.0
            self.drift_rate = float(np.mean(np.abs(self.drift_vector)))

    def update_adaptive(self, phi_teacher, phi_obs, signal_x):
        """Slowly update adaptive delta_hat when connected, using complex rotation."""
        if signal_x < 0.3: return 
        
        c_teacher = self._to_complex(phi_teacher)
        c_obs = self._to_complex(phi_obs)
        delta_current = wrap_to_pi(np.angle(c_obs + 1e-12) - np.angle(c_teacher + 1e-12))
        
        # Confidence-weighted update
        weight = self.alpha_drift * signal_x
        
        # Update using complex exponential for circular consistency
        z_adaptive = np.exp(1j * self.adaptive_delta_hat)
        z_current = np.exp(1j * delta_current)
        
        z_next = (1.0 - weight) * z_adaptive + weight * z_current
        self.adaptive_delta_hat = np.angle(z_next)
        
        # Reset extrapolation counter when we have a real update
        self.steps_since_adaptive_update = 0

    def step(self, connected=True):
        """Called once per frame to handle internal state evolution."""
        if not connected:
            self.steps_since_adaptive_update += 1
            # Extrapolate drift during disconnect
            self.adaptive_delta_hat = wrap_to_pi(self.adaptive_delta_hat + self.drift_vector)

    def apply_inverse(self, phi_obs, adaptive=False):
        """Map observed student phase (8D) back to unrefracted frame."""
        offset = self.adaptive_delta_hat if adaptive else self.delta_hat
        c_obs = self._to_complex(phi_obs)
        c_unrefracted = c_obs * np.exp(-1j * offset)
        return normalize(self._to_real(c_unrefracted))

    def apply_forward(self, phi_true, adaptive=False):
        """Map true/model phase (8D) to observed frame."""
        offset = self.adaptive_delta_hat if adaptive else self.delta_hat
        c_true = self._to_complex(phi_true)
        c_refracted = c_true * np.exp(1j * offset)
        return normalize(self._to_real(c_refracted))

    def to_dict(self):
        return {
            "delta_hat": self.delta_hat.tolist(),
            "adaptive_delta_hat": self.adaptive_delta_hat.tolist(),
            "confidence": self.confidence,
            "variance": self.variance,
            "drift_rate": self.drift_rate,
            "drift_vector": self.drift_vector.tolist(),
            "delta_history": [[h[0].tolist(), float(h[1])] for h in self.delta_history]
        }

    @classmethod
    def from_dict(cls, band_id, d, channels=8):
        obj = cls(band_id, channels=channels)
        num_pairs = channels // 2
        if d:
            obj.delta_hat = np.array(d.get("delta_hat", np.zeros(num_pairs)))
            obj.adaptive_delta_hat = np.array(d.get("adaptive_delta_hat", obj.delta_hat.copy()))
            obj.confidence = float(d.get("confidence", 0.0))
            obj.variance = float(d.get("variance", 1.0))
            obj.drift_rate = float(d.get("drift_rate", 0.0))
            obj.drift_vector = np.array(d.get("drift_vector", np.zeros(num_pairs)))
            
            hist_data = d.get("delta_history", [])
            obj.delta_history = [(np.array(h[0]), h[1]) for h in hist_data]
        return obj

    def classify_medium(self):
        if self.variance <= 0.15 and self.drift_rate <= 0.05:
            return "constant_medium"
        if self.variance <= 0.35:
            return "weakly_drifting_medium"
        return "unstable_medium"
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

    def update_train(self, phi_teacher, phi_obs, omega_teacher, omega_obs, signal_x):
        band = self.bands[self.active_band_id]
        band.learn_step(phi_teacher, phi_obs, omega_teacher, omega_obs, signal_x)

    def update_adaptive(self, phi_teacher, phi_obs, signal_x):
        """Called only when connected but after initial training."""
        band = self.bands[self.active_band_id]
        band.update_adaptive(phi_teacher, phi_obs, signal_x)

    def step(self, connected=True):
        """Propagate step to all bands."""
        for band in self.bands.values():
            band.step(connected)

    def unrefract(self, phi_obs):
        return self.bands[self.active_band_id].apply_inverse(phi_obs, adaptive=self.use_adaptive)

    def refract(self, phi_unrefracted):
        return self.bands[self.active_band_id].apply_forward(phi_unrefracted, adaptive=self.use_adaptive)

    def to_dict(self):
        return {
            "active_band_id": self.active_band_id,
            "use_adaptive": self.use_adaptive,
            "bands": {k: b.to_dict() for k, b in self.bands.items()}
        }

    @classmethod
    def from_dict(cls, d, channels=8):
        obj = cls(channels=channels)
        if d:
            obj.active_band_id = d.get("active_band_id", "alpha")
            obj.use_adaptive = d.get("use_adaptive", True)
            bands_data = d.get("bands", {})
            for k, b_data in bands_data.items():
                if k in obj.bands:
                    obj.bands[k] = BandRefractionModel.from_dict(k, b_data, channels=channels)
        return obj

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
            "use_adaptive": self.use_adaptive
        }

