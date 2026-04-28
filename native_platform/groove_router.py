import numpy as np
from .phase_space import normalize, phase_mismatch

class Groove:
    def __init__(self, groove_id, segment_initial, op_initial, trace_size=64):
        self.groove_id = groove_id
        self.trace_size = trace_size
        self.trace_segments = []
        
        # Identity features: use velocity (segment) instead of static phase
        self.centroid_segment = np.asarray(segment_initial, dtype=float).copy() if segment_initial is not None else np.zeros(8)
        self.op_histogram = {'++': 0.0, '--': 0.0, '+-': 0.0, '-+': 0.0}
        self.op_histogram[op_initial] = 1.0
        
        # Metrics
        self.use_count = 1
        self.success_count = 0
        self.last_seen = 0
        
    def score(self, current_segment, op_star):
        # 1. Velocity/Flow similarity
        if current_segment is None or np.all(self.centroid_segment == 0):
            return 0.0
            
        dist = np.linalg.norm(current_segment - self.centroid_segment)
        # Convert distance to a similarity score [0, 1]
        flow_similarity = np.exp(-5.0 * dist) 
        
        # 2. Operator match
        total_ops = sum(self.op_histogram.values())
        operator_match = self.op_histogram.get(op_star, 0.0) / total_ops if total_ops > 0 else 0.0
        
        return 0.85 * flow_similarity + 0.15 * operator_match

    def reinforce(self, segment, phi_oriented, op_star, decision, threshold=0.015):
        """
        Update identity only if survivability decision allows.
        """
        if decision == "reject":
            return
            
        self.use_count += 1
        self.last_seen = 0
        
        if decision == "reinforce":
            # Update identity features with momentum
            alpha = 0.01 
            if segment is not None:
                self.centroid_segment = normalize((1.0 - alpha) * self.centroid_segment + alpha * segment)
                
            self.op_histogram[op_star] = self.op_histogram.get(op_star, 0.0) + 1.0
            
            # Store directional segment
            self.success_count += 1
            self.trace_segments.append(np.asarray(segment, dtype=float).copy())
            if len(self.trace_segments) > self.trace_size:
                self.trace_segments.pop(0)
        elif decision == "hold":
            # Just keep recent context on hold
            if len(self.trace_segments) > 10:
                self.trace_segments = self.trace_segments[-10:]

    def feedback_vector(self):
        if len(self.trace_segments) < 8:
            return None
        recent = self.trace_segments[-min(len(self.trace_segments), 16):]
        return normalize(np.mean(np.stack(recent, axis=0), axis=0))

    def to_dict(self):
        return {
            "groove_id": self.groove_id,
            "trace_size": self.trace_size,
            "trace_segments": [s.tolist() for s in self.trace_segments],
            "centroid_segment": self.centroid_segment.tolist(),
            "op_histogram": self.op_histogram,
            "use_count": self.use_count,
            "success_count": self.success_count,
            "last_seen": self.last_seen
        }

    @staticmethod
    def from_dict(d):
        # Fallback for old memory files
        centroid = d.get("centroid_segment", d.get("centroid_phi", np.zeros(8)))
        g = Groove(d["groove_id"], centroid, "++", d.get("trace_size", 64))
        g.trace_segments = [np.array(s) for s in d.get("trace_segments", [])]
        g.centroid_segment = np.array(centroid)
        g.op_histogram = d.get("op_histogram", {'++': 0.0, '--': 0.0, '+-': 0.0, '-+': 0.0})
        g.use_count = d.get("use_count", 0)
        g.success_count = d.get("success_count", 0)
        g.last_seen = d.get("last_seen", 0)
        return g

class GrooveRouter:
    def __init__(self, max_grooves=8, create_threshold=0.85, switch_margin=0.05):
        self.grooves = {}
        self.active_groove_id = None
        self.max_grooves = max_grooves
        self.create_threshold = create_threshold
        self.switch_margin = switch_margin
        self.next_id = 0
        
        # identity_lock_v1
        self.locked_groove_id = None
        self.lock_duration = 0
        self.min_lock_frames = 25
        self.running_mismatch_mean = 0.015 # seed with safe default

    def route(self, phi_prev, phi_oriented, op_star):
        best_id = None
        best_score = -1.0
        
        current_segment = normalize(np.asarray(phi_oriented) - np.asarray(phi_prev)) if phi_prev is not None else None
        
        if current_segment is None:
            return self.grooves.get(self.active_groove_id), 0.0

        # if currently locked, stay locked unless strong evidence
        if self.locked_groove_id is not None:
            if self.lock_duration < self.min_lock_frames:
                self.lock_duration += 1
                self.active_groove_id = self.locked_groove_id
                return self.grooves.get(self.active_groove_id), 1.0 # assume perfect lock score

        # 1. Score all existing grooves
        scores = {gid: g.score(current_segment, op_star) for gid, g in self.grooves.items()}
        
        if scores:
            best_id = max(scores, key=scores.get)
            best_score = scores[best_id]
        
        # 2. Selection / Creation logic
        if (best_score < self.create_threshold or best_id is None) and len(self.grooves) < self.max_grooves:
            # Create new groove
            new_groove = self.create_groove(current_segment, op_star)
            self.active_groove_id = new_groove.groove_id
            self.locked_groove_id = self.active_groove_id
            self.lock_duration = 0
        elif best_id is not None:
            # Check if we should switch to best_id
            if self.active_groove_id is None:
                self.active_groove_id = best_id
                self.locked_groove_id = best_id
                self.lock_duration = 0
            else:
                current_score = scores.get(self.active_groove_id, 0.0)
                # stronger margin for identity_lock_v1
                if best_score > current_score + 0.15:
                    self.active_groove_id = best_id
                    self.locked_groove_id = best_id
                    self.lock_duration = 0
                else:
                    # stay with current active/locked
                    pass
        
        return self.grooves.get(self.active_groove_id), (best_score if best_id else 0.0)

    def create_groove(self, segment, op_star):
        gid = f"groove_{self.next_id}"
        self.next_id += 1
        g = Groove(gid, segment, op_star)
        self.grooves[gid] = g
        return g

    def reinforce_active(self, phi_prev, phi_current, op_star, decision, threshold=0.015):
        if self.active_groove_id is None or phi_prev is None:
            return
        
        # Break lock on reject (sharp divergence)
        if decision == "reject":
            self.locked_groove_id = None
            self.lock_duration = 0
        
        g = self.grooves[self.active_groove_id]
        segment = normalize(np.asarray(phi_current) - np.asarray(phi_prev))
        g.reinforce(segment, phi_current, op_star, decision, threshold=threshold)

    def active_feedback_vector(self):
        if self.active_groove_id is None:
            return None
        return self.grooves[self.active_groove_id].feedback_vector()

    def to_dict(self):
        return {
            "grooves": {gid: g.to_dict() for gid, g in self.grooves.items()},
            "next_id": self.next_id,
            "max_grooves": self.max_grooves,
            "create_threshold": self.create_threshold,
            "switch_margin": self.switch_margin,
            "locked_groove_id": self.locked_groove_id,
            "lock_duration": self.lock_duration,
            "running_mismatch_mean": self.running_mismatch_mean
        }

    @staticmethod
    def from_dict(d):
        if not d:
            return GrooveRouter()
        
        gr = GrooveRouter(
            max_grooves=d.get("max_grooves", 8),
            create_threshold=d.get("create_threshold", 0.85),
            switch_margin=d.get("switch_margin", 0.05)
        )
        gr.next_id = d.get("next_id", 0)
        gr.grooves = {gid: Groove.from_dict(gd) for gid, gd in d.get("grooves", {}).items()}
        gr.locked_groove_id = d.get("locked_groove_id")
        gr.lock_duration = d.get("lock_duration", 0)
        gr.running_mismatch_mean = d.get("running_mismatch_mean", 0.015)
        return gr

    def summary(self):
        return {
            "gro_count": len(self.grooves),
            "act_gid": self.active_groove_id,
            "locked": self.locked_groove_id,
            "lock_dur": self.lock_duration,
            "gro_metrics": {
                gid: {
                    "use": g.use_count,
                    "ok": g.success_count,
                    "segs": len(g.trace_segments)
                } for gid, g in self.grooves.items()
            }
        }
