import numpy as np
from .phase_space import normalize

def wrap(v):
    """Wraps values to [-pi, pi] range."""
    return (v + np.pi) % (2 * np.pi) - np.pi

class InductiveTransformerLayer:
    def __init__(self, channels=8, sample_rate=128):
        self.channels = channels
        self.sample_rate = sample_rate
        
        # Internal State (Student)
        self.theta = np.zeros(channels) # 'Phase' of each channel
        self.A = np.zeros(channels)     # Amplitude
        self.omega = np.zeros(channels) # Phase velocity
        self.L = np.zeros(channels)     # Inductive motion memory
        
        # Teacher Tracking (Last Input)
        self.teacher_theta = np.zeros(channels)
        self.teacher_omega = np.zeros(channels)
        self.teacher_amp = np.zeros(channels)
        
        # Parameters (Defaults from Patch 24/25)
        self.decay_L = 0.96
        self.gain_L = 0.08
        self.k_input_base = 0.18
        self.k_L = 0.22
        self.k_neighbor = 0.04
        self.decay_A = 0.97
        self.k_A = 0.1
        self.max_phase_correction = 0.12
        
        self.prev_theta_input = None

    def update(self, theta_input, A_input, signal_x, connected=True):
        """
        Updates the inductive transformer state.
        theta_input: Target phase (8-element vector)
        A_input: Target amplitude (scalar or vector)
        signal_x: Consistency proxy [0, 1]
        connected: Boolean, if False, runs in internal-only mode (disconnect)
        """
        # Teacher Update (for metric tracking)
        if theta_input is not None:
            if self.prev_theta_input is not None:
                self.teacher_omega = wrap(theta_input - self.prev_theta_input)
            self.teacher_theta = theta_input
            self.teacher_amp = np.full(self.channels, A_input) if np.isscalar(A_input) else A_input

        # 1. Dynamic Coupling k_input scales with Signal X
        k_input = self.k_input_base * signal_x if connected else 0.0
        
        # 2. Inductive Memory Update (L_t = decay_L * L_prev + gain_L * omega_t)
        self.L = self.decay_L * self.L + self.gain_L * self.omega
        
        # 3. Phase Update (Theta)
        if connected:
            # External driving + Inductive Bias
            correction = wrap(theta_input - self.theta)
            correction = np.clip(correction, -self.max_phase_correction, self.max_phase_correction)
            delta_theta = self.omega + k_input * correction + self.k_L * self.L
        else:
            # Internal purely: Momentum + Inductive Bias
            delta_theta = self.omega + self.k_L * self.L
            
        # Neighbor coupling (Transformer-style coupling)
        if self.channels > 1:
            mean_theta = np.mean(self.theta)
            neighbor_pull = self.k_neighbor * wrap(mean_theta - self.theta)
            delta_theta += neighbor_pull

        # Apply update
        new_theta = self.theta + delta_theta
        
        # Update velocity (Internal / Student)
        self.omega = wrap(new_theta - self.theta)
        
        # Update state
        self.theta = new_theta 
        
        # 4. Amplitude Update
        if connected:
            target_A = np.full(self.channels, A_input) if np.isscalar(A_input) else A_input
            self.A = self.decay_A * self.A + self.k_A * target_A
        else:
            self.A = self.decay_A * self.A

        self.prev_theta_input = theta_input.copy() if theta_input is not None else None
        
        return self.get_state_vector()

    def get_state_vector(self):
        """Returns the current state as a normalized vector for integration."""
        return normalize(self.theta)

    def get_raw_geometry(self):
        """Exposes raw geometry for Patch 25 logging."""
        return {
            "teacher_theta": self.teacher_theta.tolist(),
            "student_theta": self.theta.tolist(),
            "teacher_amp": self.teacher_amp.tolist(),
            "student_amp": self.A.tolist(),
            "teacher_omega": self.teacher_omega.tolist(),
            "student_omega": self.omega.tolist()
        }
