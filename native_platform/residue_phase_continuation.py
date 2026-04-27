import numpy as np
from .phase_space import normalize, phase_mismatch

class ResiduePhaseContinuation:
    def __init__(self, history_size=64, trace_size=128, base_groove_gain=0.15, max_groove_gain=0.75):
        self.history = []
        self.trace_buffer = []
        self.history_size = int(history_size)
        self.trace_size = int(trace_size)
        self.base_groove_gain = float(base_groove_gain)
        self.max_groove_gain = float(max_groove_gain)
        self.traversal_count = 0

    def update_history(self, phi):
        phi = np.asarray(phi, dtype=float)
        self.history.append(phi)
        if len(self.history) > self.history_size:
            self.history.pop(0)

    def reinforce_trace(self, phi, mismatch, threshold=0.02):
        if mismatch <= threshold:
            self.trace_buffer.append(np.asarray(phi, dtype=float).copy())
            if len(self.trace_buffer) > self.trace_size:
                self.trace_buffer.pop(0)

    def groove_gain(self):
        # gain grows with repeated traversal but remains bounded
        growth = 1.0 - np.exp(-0.15 * float(self.traversal_count))
        return self.base_groove_gain + (self.max_groove_gain - self.base_groove_gain) * growth

    def trace_groove_vector(self):
        if not self.trace_buffer:
            return None
        return normalize(np.mean(np.stack(self.trace_buffer, axis=0), axis=0))

    def continue_next(self, phi_current, last_mismatch=None):
        phi_current = normalize(np.asarray(phi_current, dtype=float))

        if len(self.history) < 2:
            self.update_history(phi_current)
            return phi_current

        # local phase-flow continuation
        phi_prev = self.history[-1]
        phi_prev2 = self.history[-2]
        v1 = phi_current - phi_prev
        v2 = phi_prev - phi_prev2
        curvature = v1 - v2

        local_continuation = normalize(phi_current + 1.15 * v1 + 0.35 * curvature)

        # trace-feedback groove continuation
        groove = self.trace_groove_vector()
        if groove is not None:
            beta = self.groove_gain()
            phi_continued = normalize((1.0 - beta) * local_continuation + beta * groove)
        else:
            phi_continued = local_continuation

        self.update_history(phi_current)
        return phi_continued

    def mark_traversal_complete(self):
        self.traversal_count += 1
