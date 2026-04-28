import numpy as np
from .phase_space import normalize

def wrap(v):
    """Wraps values to [-pi, pi] range."""
    return (v + np.pi) % (2 * np.pi) - np.pi

class InductiveTransformerLayer:
    def __init__(self, channels=8, sample_rate=128):
        self.channels = channels
        self.sample_rate = sample_rate
        
        # State
        self.theta = np.zeros(channels) # 'Phase' of each channel
        self.A = np.zeros(channels)     # Amplitude
        self.omega = np.zeros(channels) # Phase velocity
        self.L = np.zeros(channels)     # Inductive motion memory
        
        # Parameters (Defaults from Patch 24)
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
        # 1. Dynamic Coupling k_input scales with Signal X
        k_input = self.k_input_base * signal_x if connected else 0.0
        
        # 2. Phase Velocity (Omega)
        # If we have a new input, we can derive velocity from it
        if self.prev_theta_input is not None:
            # We don't use theta_input directly to drive omega, 
            # we use the internal theta's progress to maintain continuity.
            pass 

        # 3. Inductive Memory Update (L_t = decay_L * L_prev + gain_L * omega_t)
        # Omega is our internal tracking of velocity
        self.L = self.decay_L * self.L + self.gain_L * self.omega
        
        # 4. Phase Update (Theta)
        # theta_next = theta + omega + k_input * wrap(theta_input - theta) + k_L * L_t
        
        if connected:
            # External driving + Inductive Bias
            correction = wrap(theta_input - self.theta)
            # Clip correction to max_phase_correction to prevent shock
            correction = np.clip(correction, -self.max_phase_correction, self.max_phase_correction)
            
            delta_theta = self.omega + k_input * correction + self.k_L * self.L
        else:
            # Internal purely: Momentum + Inductive Bias
            delta_theta = self.omega + self.k_L * self.L
            
        # Neighbor coupling (Transformer-style coupling)
        # Simple all-to-all mean coupling for now
        if self.channels > 1:
            mean_theta = np.mean(self.theta)
            neighbor_pull = self.k_neighbor * wrap(mean_theta - self.theta)
            delta_theta += neighbor_pull

        # Apply update
        new_theta = self.theta + delta_theta
        
        # Update velocity (Internal)
        self.omega = wrap(new_theta - self.theta)
        
        # Update state
        self.theta = new_theta # Note: for hypersphere we might normalize, 
                               # but the spec treats these as wrapping phases.
        
        # 5. Amplitude Update
        if connected:
            self.A = self.decay_A * self.A + self.k_A * A_input
        else:
            self.A = self.decay_A * self.A

        self.prev_theta_input = theta_input.copy() if theta_input is not None else None
        
        return self.get_state_vector()

    def get_state_vector(self):
        """Returns the current state as a normalized vector for integration."""
        # For our 8-element phase space, we can map theta back to a normalized vector
        # Since theta can grow, we use sin(theta) or similar if they are true phases,
        # but here they are likely offsets. Let's just normalize the result.
        return normalize(self.theta)
