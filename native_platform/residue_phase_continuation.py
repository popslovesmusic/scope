import numpy as np
from .phase_space import normalize, phase_mismatch
from .signal_layer import get_consistency_level

class ResiduePhaseContinuation:
    def __init__(self, history_size=64, trace_size=128, base_groove_gain=0.15, max_groove_gain=0.25, successful_traversals=0):
        self.history = []
        self.trace_buffer = []
        self.history_size = int(history_size)
        self.trace_size = int(trace_size)
        
        # Patch 23: Scalars are preserved only for potential small bias in 'hold' mode
        self.trace_feedback_gain = 0.10
        self.max_trace_feedback_gain = 0.65
        self.successful_traversals = int(successful_traversals)
        
        self.trace_segments = []
        self.traversal_count = 0
        self.running_mismatch_mean = 0.015
        self.mismatch_history = []

    def update_history(self, phi):
        phi = np.asarray(phi, dtype=float)
        self.history.append(phi)
        if len(self.history) > self.history_size:
            self.history.pop(0)

    def evaluate_survivability(self, phi_candidate, mismatch, op_star, signal_x):
        """
        Elimination-based survivability gating (Patch 23).
        Returns: 'reinforce', 'hold', or 'reject'
        """
        # 1. Update mismatch trend
        self.mismatch_history.append(mismatch)
        if len(self.mismatch_history) > 8:
            self.mismatch_history.pop(0)
            
        mismatch_smooth = np.mean(self.mismatch_history)
        self.running_mismatch_mean = 0.95 * self.running_mismatch_mean + 0.05 * mismatch_smooth
        
        failed_tests = []
        
        # Test A: Admissibility (Implicitly passed if O* exists)
        if op_star is None: failed_tests.append("admissibility")
        
        # Test B: Persistence
        # Using smoothed mismatch for gating to avoid noise spikes
        if mismatch_smooth > 3.5 * self.running_mismatch_mean:
            failed_tests.append("persistence_hard")
        elif mismatch_smooth > 1.2 * self.running_mismatch_mean:
            failed_tests.append("persistence_soft")
            
        # Test C: Cross-view consistency (Signal X)
        x_level = get_consistency_level(signal_x)
        if x_level == "low":
            failed_tests.append("consistency_hard")
        elif x_level == "moderate":
            # Consistency is soft if below a higher bar
            if signal_x < 0.6:
                failed_tests.append("consistency_soft")
            
        # Test D: Trace coherence
        trace_vec = self.trace_feedback_vector()
        if trace_vec is not None:
            if len(self.history) > 0:
                seg = normalize(phi_candidate - self.history[-1])
                dist = np.linalg.norm(seg - trace_vec)
                # Permissive coherence for EEG features
                if dist > 0.7: 
                    failed_tests.append("coherence")

        # Decision Logic
        if any(t in failed_tests for t in ["persistence_hard", "consistency_hard"]):
            return "reject", failed_tests
            
        if len(failed_tests) == 0:
            return "reinforce", failed_tests
            
        return "hold", failed_tests

    def store_trace_segment(self, phi_prev, phi_current, mismatch, decision):
        """
        Update trace buffer based on survivability decision.
        """
        if phi_prev is None or decision == "reject":
            return
            
        # On 'reinforce', store fully. On 'hold', preserve context but don't deeply reinforce.
        if decision == "reinforce":
            segment = normalize(np.asarray(phi_current) - np.asarray(phi_prev))
            self.trace_segments.append(segment)
            if len(self.trace_segments) > self.trace_size:
                self.trace_segments.pop(0)
        elif decision == "hold":
            # Just keep recent context on hold
            if len(self.trace_segments) > 10:
                self.trace_segments = self.trace_segments[-10:]

    def trace_feedback_vector(self):
        if len(self.trace_segments) < 16:
            return None
        recent = self.trace_segments[-min(len(self.trace_segments), 32):]
        return normalize(np.mean(np.stack(recent, axis=0), axis=0))

    def groove_gain(self):
        growth = 1.0 - np.exp(-0.20 * float(self.successful_traversals))
        potential_gain = self.trace_feedback_gain + (self.max_trace_feedback_gain - self.trace_feedback_gain) * growth
        buffer_fill = min(1.0, len(self.trace_segments) / 32.0)
        return potential_gain * buffer_fill

    def continue_next(self, phi_current, decision, external_feedback_vec=None):
        phi_current = normalize(np.asarray(phi_current, dtype=float))

        if len(self.history) < 2:
            self.update_history(phi_current)
            return phi_current

        phi_prev = self.history[-1]
        phi_prev2 = self.history[-2]
        v1 = phi_current - phi_prev
        v2 = phi_prev - phi_prev2
        curvature = v1 - v2
        local_continuation = normalize(phi_current + 1.15 * v1 + 0.35 * curvature)

        trace_vec = self.trace_feedback_vector()
        
        # Patch 23: Selection-based continuation (no weights when reinforced)
        if decision == "reinforce":
            # High confidence: additive integration of all available signals
            combined = local_continuation.copy()
            if trace_vec is not None: combined += trace_vec
            if external_feedback_vec is not None: combined += external_feedback_vec
            phi_continued = normalize(combined)
        elif decision == "hold":
            # Moderate confidence: local momentum + small bias from trace
            combined = local_continuation.copy()
            if trace_vec is not None: combined += 0.1 * trace_vec
            phi_continued = normalize(combined)
        else:
            # Low confidence/Reject: pure local momentum
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

    def reinforce_trace(self, phi, mismatch, threshold=0.01):
        # Legacy positional reinforcement - keep for now but could be integrated
        phi = np.asarray(phi, dtype=float)
        if mismatch < threshold:
            self.trace_buffer.append(phi.copy())
            if len(self.trace_buffer) > self.trace_size:
                self.trace_buffer.pop(0)
        else:
            self.trace_buffer = self.trace_buffer[-10:]
