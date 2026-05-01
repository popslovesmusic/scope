import numpy as np
from .phase_space import normalize

def wrap_to_pi(v):
    return (v + np.pi) % (2 * np.pi) - np.pi

def circular_mean(angles):
    if len(angles) == 0:
        return 0.0
    return np.angle(np.mean(np.exp(1j * np.asarray(angles))))

class BandOscillator:
    def __init__(self, band_id, channels=8):
        self.band_id = band_id
        self.channels = channels
        self.num_pairs = channels // 2
        
        self.theta = np.zeros(self.num_pairs)
        self.omega = np.zeros(self.num_pairs)
        self.alpha = np.zeros(self.num_pairs)
        self.A = np.zeros(self.num_pairs)
        
        self.omega_history = []
        self.alpha_history = []
        self.history_len = 32
        
        self.prev_theta_teacher = None
        self.prev_omega_teacher = None
        
        self.confidence = 1.0
        
        # Patch 27 Defaults - Tuned for rigidity
        self.decay_omega = 0.995 
        self.decay_alpha = 0.95 
        self.k_alpha = 0.40     
        self.k_restore = 0.25   
        self.omega_floor = 0.001
        self.omega_max = 0.50
        self.alpha_max = 0.15
        self.gain_alpha = 0.08

    def _to_complex(self, v):
        return v[0::2] + 1j * v[1::2]

    def _to_real(self, c):
        res = np.zeros(self.channels)
        res[0::2] = c.real
        res[1::2] = c.imag
        return res

    def update_train(self, phi_in, A_in):
        # Convert 8D phi_in to 4 angles
        c_in = self._to_complex(phi_in)
        theta_in = np.angle(c_in + 1e-12)

        omega_in = wrap_to_pi(theta_in - self.theta)

        if self.prev_theta_teacher is not None:
            omega_teacher = wrap_to_pi(theta_in - self.prev_theta_teacher)
            self.omega_history.append(omega_teacher)
            if len(self.omega_history) > self.history_len:
                self.omega_history.pop(0)

            if self.prev_omega_teacher is not None:
                alpha_teacher = wrap_to_pi(omega_teacher - self.prev_omega_teacher)
                self.alpha_history.append(alpha_teacher)
                if len(self.alpha_history) > self.history_len:
                    self.alpha_history.pop(0)
            self.prev_omega_teacher = omega_teacher

        self.prev_theta_teacher = theta_in.copy()

        # Follow the teacher with a coupling constant during train
        k_follow = 0.20
        self.theta = wrap_to_pi(self.theta + k_follow * wrap_to_pi(theta_in - self.theta))
        self.omega = omega_in

        if np.isscalar(A_in):
            self.A = np.full_like(self.theta, A_in)
        else:
            self.A = np.asarray(A_in[0::2]).copy()
        self.confidence = 1.0

    def step_recursive(self, k_L=0.12, L=None):
        if L is None:
            L = np.zeros_like(self.theta)
            
        if len(self.omega_history) < 4:
            c = np.exp(1j * self.theta)
            return normalize(self._to_real(c))

        omega_hat = np.array([circular_mean([h[c] for h in self.omega_history]) for c in range(len(self.theta))])
        if len(self.omega_history) > 1:
            diffs = np.diff(np.array(self.omega_history), axis=0)
            alpha_hat = np.array([circular_mean(diffs[:, c]) for c in range(len(self.theta))])
        else:
            alpha_hat = np.zeros_like(self.theta)

        # 2. Omega Next
        omega_next = self.decay_omega * self.omega + self.k_alpha * alpha_hat + k_L * L + self.k_restore * (omega_hat - self.omega)
        omega_next = np.clip(omega_next, -self.omega_max, self.omega_max)
        
        moving_mask = np.abs(omega_hat) > self.omega_floor
        frozen_mask = np.abs(omega_next) < self.omega_floor
        restore_mask = np.logical_and(moving_mask, frozen_mask)
        omega_next[restore_mask] = omega_hat[restore_mask] * self.confidence

        # 3. Theta Next
        self.theta = wrap_to_pi(self.theta + omega_next)
        
        # 4. Alpha Next
        self.alpha = self.decay_alpha * self.alpha + self.gain_alpha * (omega_next - self.omega)
        self.alpha = np.clip(self.alpha, -self.alpha_max, self.alpha_max)
        
        self.omega = omega_next
        self.A *= 0.98
        self.confidence *= 0.99
        
        c = np.exp(1j * self.theta)
        return normalize(self._to_real(c))

    def to_dict(self):
        return {
            "band_id": self.band_id,
            "theta": self.theta.tolist(),
            "omega": self.omega.tolist(),
            "alpha": self.alpha.tolist(),
            "A": self.A.tolist(),
            "omega_history": [h.tolist() for h in self.omega_history],
            "alpha_history": [h.tolist() for h in self.alpha_history],
            "confidence": self.confidence,
            "prev_theta_teacher": self.prev_theta_teacher.tolist() if self.prev_theta_teacher is not None else None,
            "prev_omega_teacher": self.prev_omega_teacher.tolist() if self.prev_omega_teacher is not None else None
        }

    @staticmethod
    def from_dict(d, channels=8):
        b = BandOscillator(d["band_id"], channels=channels)
        num_pairs = channels // 2
        b.theta = np.array(d.get("theta", np.zeros(num_pairs)))
        b.omega = np.array(d.get("omega", np.zeros(num_pairs)))
        b.alpha = np.array(d.get("alpha", np.zeros(num_pairs)))
        b.A = np.array(d.get("A", np.zeros(num_pairs)))
        b.omega_history = [np.array(h) for h in d.get("omega_history", [])]
        b.alpha_history = [np.array(h) for h in d.get("alpha_history", [])]
        b.confidence = d.get("confidence", 1.0)
        b.prev_theta_teacher = np.array(d["prev_theta_teacher"]) if d.get("prev_theta_teacher") is not None else None
        b.prev_omega_teacher = np.array(d["prev_omega_teacher"]) if d.get("prev_omega_teacher") is not None else None
        return b

class RecursiveMotionAnchor:
    def __init__(self, channels=8):
        self.bands = {
            "theta": BandOscillator("theta", channels),
            "alpha": BandOscillator("alpha", channels),
            "beta": BandOscillator("beta", channels),
            "mixed": BandOscillator("mixed", channels)
        }
        self.active_band_id = "alpha" 
        self.channels = channels

    def update(self, phi_input, A_input, signal_x, connected=True, active_groove_id=None, L=None):
        band = self.bands[self.active_band_id]
        if connected:
            band.update_train(phi_input, A_input)
            return phi_input
        else:
            return band.step_recursive(L=L)

    def get_state(self):
        band = self.bands[self.active_band_id]
        return {
            "theta": band.theta.tolist(),
            "omega": band.omega.tolist(),
            "alpha": band.alpha.tolist(),
            "A": band.A.tolist(),
            "confidence": band.confidence
        }

    def to_dict(self):
        return {
            "bands": {bid: b.to_dict() for bid, b in self.bands.items()},
            "active_band_id": self.active_band_id,
            "channels": self.channels
        }

    @staticmethod
    def from_dict(d):
        if not d:
            return RecursiveMotionAnchor()
        channels = d.get("channels", 8)
        ra = RecursiveMotionAnchor(channels=channels)
        ra.active_band_id = d.get("active_band_id", "alpha")
        ra.bands = {bid: BandOscillator.from_dict(bd, channels=channels) for bid, bd in d.get("bands", {}).items()}
        return ra
