import numpy as np
import colorsys
from typing import List, Dict, Any, Optional

class SignalScopeWindow:
    def __init__(self, window_id: str, level: str, fallback_policy: str = "decay_to_center"):
        self.window_id = window_id
        self.level = level
        self.fallback_policy = fallback_policy
        self.W = np.array([1/3, 1/3, 1/3])
        self.V = np.zeros(3)
        self.V_prev = np.zeros(3)
        self.R_scope = np.array([1/3, 1/3, 1/3])  # Residue memory trace
        self.decay_rate = 0.05
        self.floor = 1e-6
        self.ema_alpha = 0.1
        
        # State history for event detection
        self.history_C = []
        self.history_W = []
        self.decoupled = False

    def normalize(self, raw: np.ndarray) -> np.ndarray:
        # Clamp or rectify raw values to non-negative domain
        X = np.maximum(raw, 0)
        S = np.sum(X)
        if S > self.floor:
            return X / S
        
        # Fallback policies
        if self.fallback_policy == "neutral":
            return np.array([1/3, 1/3, 1/3])
        elif self.fallback_policy == "hold_last":
            return self.W
        elif self.fallback_policy == "decay_to_center":
            return (1 - self.decay_rate) * self.W + self.decay_rate * np.array([1/3, 1/3, 1/3])
        return np.array([1/3, 1/3, 1/3])

    def update(self, raw: np.ndarray):
        W_new = self.normalize(raw)
        self.V_prev = self.V
        self.V = W_new - self.W
        self.W = W_new
        
        # Update Residue: memory trace of prior state evolution
        # R_scope(t) = EMA(W, V, C, E_scope)
        self.R_scope = (1 - self.ema_alpha) * self.R_scope + self.ema_alpha * self.W
        
        # Update history
        coupling = np.min(self.W) / (np.max(self.W) + 1e-12)
        self.history_C.append(coupling)
        self.history_W.append(self.W.copy())
        if len(self.history_C) > 20:
            self.history_C.pop(0)
            self.history_W.pop(0)

    def get_observables(self) -> Dict[str, Any]:
        W = self.W
        V = self.V
        V_prev = self.V_prev
        
        coupling = float(np.min(W) / (np.max(W) + 1e-12))
        imbalance = float(np.linalg.norm(W - np.array([1/3, 1/3, 1/3])))
        dominance = float(np.max(W))
        speed = float(np.linalg.norm(V))
        
        # Curvature: angle between V(t) and V(t-1)
        if np.linalg.norm(V) > 1e-12 and np.linalg.norm(V_prev) > 1e-12:
            cos_theta = np.dot(V, V_prev) / (np.linalg.norm(V) * np.linalg.norm(V_prev))
            curvature = float(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
        else:
            curvature = 0.0
            
        # Projection to 2D for 8-way direction
        dx = (W[0] - W[1]) / np.sqrt(2)
        dy = (W[0] + W[1] - 2*W[2]) / np.sqrt(6)
        
        dvx = (V[0] - V[1]) / np.sqrt(2)
        dvy = (V[0] + V[1] - 2*V[2]) / np.sqrt(6)
        direction = self.compute_8way(dvx, dvy)
        
        # HSV Projection
        hue_angle = np.degrees(np.arctan2(dy, dx)) % 360
        hue = hue_angle / 360.0
        saturation = float(np.clip(imbalance / (np.sqrt(2/3)), 0, 1))
        brightness = coupling
        
        # Event Detection
        events = []
        if coupling > 0.7 and imbalance < 0.15:
            events.append("balanced_lock")
        if dominance > 0.8 and coupling < 0.2:
            events.append("single_channel_collapse")
        if speed > 0.2:
            events.append("transition_burst")
        if curvature > 2.5: # Near 180 degree flip
            events.append("chaotic_flicker")
            
        # Decoupling / Recoupling
        coupling_floor = 0.1
        recoupling_threshold = 0.3
        if coupling < coupling_floor:
            if not self.decoupled:
                events.append("decoupling")
                self.decoupled = True
        elif coupling > recoupling_threshold:
            if self.decoupled:
                events.append("recoupling")
                self.decoupled = False
                
        # Phase lock spike
        if len(self.history_C) > 2:
            c_delta = coupling - self.history_C[-2]
            if c_delta > 0.3 and speed < 0.05:
                events.append("phase_lock_spike")
                
        # Stable attractor
        if len(self.history_W) >= 10:
            var = np.var(self.history_W, axis=0)
            if np.sum(var) < 0.001:
                events.append("stable_attractor")

        return {
            "window_id": self.window_id,
            "level": self.level,
            "W": W.tolist(),
            "C": coupling,
            "E_scope": imbalance,
            "D": dominance,
            "V": V.tolist(),
            "speed": speed,
            "curvature": curvature,
            "R_scope": self.R_scope.tolist(),
            "hue": hue,
            "saturation": saturation,
            "brightness": brightness,
            "direction_8way": direction,
            "events": events
        }

    def compute_8way(self, dx, dy, threshold=1e-6) -> str:
        if np.sqrt(dx**2 + dy**2) < threshold:
            return "NC"
        angle = np.degrees(np.arctan2(dy, dx))
        if -22.5 <= angle < 22.5: return 'E'
        if 22.5 <= angle < 67.5: return 'NE'
        if 67.5 <= angle < 112.5: return 'N'
        if 112.5 <= angle < 157.5: return 'NW'
        if angle >= 157.5 or angle < -157.5: return 'W'
        if -157.5 <= angle < -112.5: return 'SW'
        if -112.5 <= angle < -67.5: return 'S'
        if -67.5 <= angle < -22.5: return 'SE'
        return 'NC'

class SignalScope:
    def __init__(self, beta: float = 0.05):
        self.local = SignalScopeWindow("L0", "local")
        self.global_window = SignalScopeWindow("G0", "global")
        self.meta_window = SignalScopeWindow("M0", "meta")
        
        # For temporal lifting
        self.local_buffer = []
        self.global_buffer = []
        self.beta = beta # top-down feedback strength

    def update(self, raw_features: np.ndarray) -> Dict[str, Any]:
        """
        Updates the hierarchy with new local features.
        raw_features: [f1, f2, f3]
        """
        # 1. Update Local
        self.local.update(raw_features)
        self.local_buffer.append(self.local.W.copy())
        if len(self.local_buffer) > 3:
            self.local_buffer.pop(0)
            
        # 2. Lift to Global (Aggregate of 3 locals)
        if len(self.local_buffer) == 3:
            g_raw = np.mean(self.local_buffer, axis=0)
            self.global_window.update(g_raw)
            self.global_buffer.append(self.global_window.W.copy())
            if len(self.global_buffer) > 3:
                self.global_buffer.pop(0)
        else:
            # Fallback if buffer not full
            self.global_window.update(self.local.W)
            
        # 3. Lift to Meta (Aggregate of 3 globals)
        if len(self.global_buffer) == 3:
            m_raw = np.mean(self.global_buffer, axis=0)
            self.meta_window.update(m_raw)
        else:
            self.meta_window.update(self.global_window.W)
            
        # 4. Top-down Feedback (Optional but in spec)
        # W_i'(t+1) = Normalize((1 - beta) * W_i(t+1) + beta * G(t))
        if self.beta > 0:
            feedback_W = (1 - self.beta) * self.local.W + self.beta * self.global_window.W
            self.local.W = self.local.normalize(feedback_W)
            
        return {
            "local": self.local.get_observables(),
            "global": self.global_window.get_observables(),
            "meta": self.meta_window.get_observables()
        }
