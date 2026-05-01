import numpy as np
from .phase_space import normalize

def wrap(v):
    """Wraps values to [-pi, pi] range."""
    return (v + np.pi) % (2 * np.pi) - np.pi

class InductiveTransformerLayer:
    def __init__(self, channels=8, sample_rate=128):
        self.channels = channels
        self.num_pairs = channels // 2
        self.sample_rate = sample_rate
        
        # Internal State (Student) - Using 4 angles for stability
        self.theta = np.zeros(self.num_pairs) 
        self.A = np.zeros(self.num_pairs)     
        self.omega = np.zeros(self.num_pairs) 
        self.L = np.zeros(self.num_pairs)     
        
        # Teacher Tracking (Last Input)
        self.teacher_theta = np.zeros(self.num_pairs)
        self.teacher_omega = np.zeros(self.num_pairs)
        
        # Parameters
        self.decay_L = 0.96
        self.gain_L = 0.08
        self.k_input_base = 0.45 
        self.k_L = 0.22
        self.max_phase_correction = 0.5 
        self.decay_A = 0.97
        self.k_A = 0.1
        
        self.prev_theta_input = None

    def _to_complex(self, v):
        return v[0::2] + 1j * v[1::2]

    def _to_real(self, c):
        res = np.zeros(self.channels)
        res[0::2] = c.real
        res[1::2] = c.imag
        return res

    def update(self, phi_input, A_input, signal_x, connected=True):
        if phi_input is None: return self.get_state_vector()
        
        # Convert 8D input to 4 angles
        c_input = self._to_complex(phi_input)
        theta_input = np.angle(c_input + 1e-12)

        # Teacher Update
        if self.prev_theta_input is not None:
            self.teacher_omega = wrap(theta_input - self.prev_theta_input)
        self.teacher_theta = theta_input

        # 1. Dynamic Coupling
        k_input = self.k_input_base * signal_x if connected else 0.0
        
        # 2. Inductive Memory Update
        self.L = self.decay_L * self.L + self.gain_L * self.omega
        
        # 3. Phase Update (Theta)
        if connected:
            correction = wrap(theta_input - self.theta)
            correction = np.clip(correction, -self.max_phase_correction, self.max_phase_correction)
            # Update internal velocity towards input
            self.omega = wrap(self.omega + k_input * correction)

        # Apply update using current velocity + inductive bias
        # 💡 CRITICAL: Don't update self.omega from this combined delta_theta 
        # to avoid positive feedback runaway.
        delta_theta = self.omega + self.k_L * self.L
        new_theta = wrap(self.theta + delta_theta)
        self.theta = new_theta

        # 4. Amplitude Update
        target_A = np.full(self.num_pairs, A_input) if np.isscalar(A_input) else A_input[0::2]
        if connected:
            self.A = self.decay_A * self.A + self.k_A * target_A
        else:
            self.A *= self.decay_A

        self.prev_theta_input = theta_input.copy()
        
        return self.get_state_vector()

    def to_dict(self):
        return {
            "theta": self.theta.tolist(),
            "omega": self.omega.tolist(),
            "L": self.L.tolist(),
            "A": self.A.tolist()
        }

    @classmethod
    def from_dict(cls, d, channels=8):
        obj = cls(channels=channels)
        num_pairs = channels // 2
        if d:
            obj.theta = np.array(d.get("theta", np.zeros(num_pairs)))
            obj.omega = np.array(d.get("omega", np.zeros(num_pairs)))
            obj.L = np.array(d.get("L", np.zeros(num_pairs)))
            obj.A = np.array(d.get("A", np.zeros(num_pairs)))
        return obj

    def get_state_vector(self):
        """Returns the current state as a normalized 8D unit vector."""
        c = np.exp(1j * self.theta)
        return normalize(self._to_real(c))

    def get_raw_geometry(self):
        return {
            "teacher_theta": self.teacher_theta.tolist(),
            "student_theta": self.theta.tolist(),
            "student_amp": self.A.tolist(),
            "student_omega": self.omega.tolist()
        }
