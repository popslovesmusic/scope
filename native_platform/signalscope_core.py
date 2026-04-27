
import numpy as np
from .scope_adapter import compute_W, compute_metrics

class ScopeWindow:
    def __init__(self, window_id):
        self.window_id = window_id
        self.W = np.array([1/3, 1/3, 1/3])
        self.C = 1.0
        self.E = 0.0
        self.V = np.zeros(3)

    def update(self, raw_features):
        W_new = compute_W(raw_features)
        self.C, self.E, self.V = compute_metrics(self.W, W_new)
        self.W = W_new

class SignalScope:
    def __init__(self):
        self.local = ScopeWindow("local")
        self.global_window = ScopeWindow("global")
        self.meta = ScopeWindow("meta")
        
        self.local_history = []
        self.global_history = []

    def update(self, node_outputs):
        # 1. Local update
        self.local.update(node_outputs)
        self.local_history.append(self.local.W.copy())
        if len(self.local_history) > 10:
            self.local_history.pop(0)

        # 2. Recursive Lifting to Global
        if len(self.local_history) >= 3:
            global_raw = np.mean(self.local_history[-3:], axis=0)
            self.global_window.update(global_raw)
            self.global_history.append(self.global_window.W.copy())
            if len(self.global_history) > 10:
                self.global_history.pop(0)
        else:
            self.global_window.update(self.local.W)

        # 3. Recursive Lifting to Meta
        if len(self.global_history) >= 3:
            meta_raw = np.mean(self.global_history[-3:], axis=0)
            self.meta.update(meta_raw)
        else:
            self.meta.update(self.global_window.W)

        return {
            "W_local": self.local.W,
            "W_global": self.global_window.W,
            "W_meta": self.meta.W,
            "C": self.local.C,
            "E": self.local.E,
            "V": self.local.V
        }
