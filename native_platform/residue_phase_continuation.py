import numpy as np
from .phase_space import normalize, phase_mismatch

class ResiduePhaseContinuation:
    def __init__(self, history_size=64, trace_size=128, base_groove_gain=0.15, max_groove_gain=0.25, successful_traversals=0):
        self.history = []
        self.trace_buffer = []
        self.history_size = int(history_size)
        self.trace_size = int(trace_size)
        self.base_groove_gain = float(base_groove_gain)
        self.max_groove_gain = float(max_groove_gain)
        self.traversal_count = 0
        
        # Patch 20: Trace feedback state
        self.trace_segments = []
        self.trace_feedback_gain = 0.10
        self.max_trace_feedback_gain = 0.65
        self.successful_traversals = int(successful_traversals)

    def update_history(self, phi):
        phi = np.asarray(phi, dtype=float)
        self.history.append(phi)
        if len(self.history) > self.history_size:
            self.history.pop(0)

    def reinforce_trace(self, phi, mismatch, threshold=0.01):
        phi = np.asarray(phi, dtype=float)
        if mismatch < threshold:
            self.trace_buffer.append(phi.copy())
            if len(self.trace_buffer) > self.trace_size:
                self.trace_buffer.pop(0)
        else:
            # decay groove on divergence
            self.trace_buffer = self.trace_buffer[-10:]

    def store_trace_segment(self, phi_prev, phi_current, mismatch, threshold=0.020):
        if phi_prev is None:
            return
        if mismatch < threshold:
            segment = normalize(np.asarray(phi_current) - np.asarray(phi_prev))
            self.trace_segments.append(segment)
            if len(self.trace_segments) > self.trace_size:
                self.trace_segments.pop(0)
        else:
            # on divergence, preserve only recent trace context
            self.trace_segments = self.trace_segments[-10:]

    def trace_feedback_vector(self):
        if len(self.trace_segments) < 8:
            return None
        recent = self.trace_segments[-min(len(self.trace_segments), 16):]
        return normalize(np.mean(np.stack(recent, axis=0), axis=0))

    def groove_gain(self):
        # Patch 20: gain grows with repeated successful traversal
        growth = 1.0 - np.exp(-0.20 * float(self.successful_traversals))
        potential_gain = self.trace_feedback_gain + (self.max_trace_feedback_gain - self.trace_feedback_gain) * growth
        
        # Buffer-level saturation: only use full gain if we have enough segments
        # Ramp up gain as we collect more segments in the current run
        buffer_fill = min(1.0, len(self.trace_segments) / 32.0)
        return potential_gain * buffer_fill

    def trace_groove_vector(self):
        if len(self.trace_buffer) < 3:
            return None
        direction = self.trace_buffer[-1] - self.trace_buffer[-2]
        return normalize(direction)

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

        # Patch 20: trace-feedback groove continuation using segments
        feedback_vec = self.trace_feedback_vector()
        if feedback_vec is not None:
            beta = self.groove_gain()
            phi_continued = normalize(local_continuation + beta * feedback_vec)
        else:
            # Fallback to older trace_groove_vector if available
            groove = self.trace_groove_vector()
            if groove is not None:
                beta = self.groove_gain()
                phi_continued = normalize(local_continuation + beta * groove)
            else:
                phi_continued = local_continuation

        self.update_history(phi_current)
        return phi_continued

    def mark_traversal_complete(self):
        self.traversal_count += 1

    def mark_successful_traversal(self):
        self.successful_traversals += 1

    def mark_failed_traversal(self):
        self.successful_traversals = max(0, self.successful_traversals - 1)
        self.trace_segments = self.trace_segments[-10:]
