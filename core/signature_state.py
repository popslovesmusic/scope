
import numpy as np

class SignatureState:
    def __init__(self, size):
        self.size = int(size)
        self.step = 0
        self.input_trace = []

        # Patch3: signed_field is the primary branch-bearing surface.
        self.signed_field = np.zeros(self.size, dtype=float)

        # amplitude is derived from signed_field (abs) and kept for compatibility.
        self.amplitude = np.zeros(self.size, dtype=float)
        self.phase = np.zeros(self.size, dtype=float)
        self.stability = np.ones(self.size, dtype=float)

        self.peaks = []
        self.bands = []

        # v14 patch2: polarized signature overlays (derived fields).
        self.polarity = np.zeros(self.size, dtype=int)
        self.zero_crossings = []
        self.mode_index = np.zeros(self.size, dtype=int)
        self.component_summary = {}

        # Patch4: promoted components (operator-addressable).
        self.components = []
        self.component_history = []
        self.active_component_id = None

        # Patch5: trace salience + caution pressure.
        self.trace_salience = []
        self.raw_caution_scalar = 0.0
        self.caution_scalar = 0.0
        self.caution_release_scalar = 0.0
        self.caution_field = np.zeros_like(self.signed_field, dtype=float)
        self.hold_state = False

        # Phase7: recovery + recontextualization signals (control-only; do not erase trace memory).
        self.recovery_scalar = 0.0
        self.recontextualization_score = 0.0
        self.hold_release_counter = 0
        self.recovery_field = np.zeros_like(self.signed_field, dtype=float)

    def derive_amplitude_from_signed(self, lo=0.0, hi=1.0):
        self.amplitude = np.clip(np.abs(self.signed_field), float(lo), float(hi))

    def apply_update(self, gated_delta, lam, kappa, eta, boundary_penalty):
        gated_delta = np.asarray(gated_delta, dtype=float)
        boundary_penalty = np.asarray(boundary_penalty, dtype=float)
        if gated_delta.shape != self.signed_field.shape:
            raise ValueError("gated_delta shape mismatch")
        if boundary_penalty.shape != self.signed_field.shape:
            raise ValueError("boundary_penalty shape mismatch")

        # v14 conceptual update:
        # signature_next = (1 - ?) * signature_current + ? * gated_projection - ? * boundary_penalty
        # Patch3: apply it to signed_field; amplitude is derived.
        self.signed_field = (1.0 - float(lam)) * self.signed_field + float(kappa) * gated_delta - float(eta) * boundary_penalty
        self.signed_field = np.clip(self.signed_field, -1.0, 1.0)
        self.derive_amplitude_from_signed(0.0, 1.0)

    def get_peaks(self, threshold):
        return np.where(self.amplitude > threshold)[0]

    def top_peaks(self, k=5):
        if self.amplitude.size == 0:
            return []
        k = max(0, min(int(k), int(self.amplitude.size)))
        if k == 0:
            return []
        idx = np.argsort(-self.amplitude)[:k]
        return [{"family": int(i), "amplitude": float(self.amplitude[i])} for i in idx]

    def apply_phase_shift(self, delta_phase, mix=0.25):
        delta_phase = float(delta_phase)
        mix = float(mix)
        self.phase = (1.0 - mix) * self.phase + mix * (self.phase + delta_phase)

    def update_stability(self, writeback_delta, alpha=0.2):
        alpha = float(alpha)
        writeback_delta = np.asarray(writeback_delta, dtype=float)
        # Smaller updates imply higher stability.
        inst = 1.0 - np.tanh(np.abs(writeback_delta))
        self.stability = (1.0 - alpha) * self.stability + alpha * inst

    def clip_amplitude(self, lo=0.0, hi=1.0):
        # Compatibility: clip derived amplitude and also keep signed_field within bounds.
        self.amplitude = np.clip(self.amplitude, float(lo), float(hi))
        self.signed_field = np.clip(self.signed_field, -float(hi), float(hi))

    def recompute_polarized_summary(
        self,
        *,
        signed_source,
        polarity_threshold: float = 0.01,
        zero_crossing_threshold: float = 0.02,
        wraparound_lattice: bool = True,
        enable_summary: bool = True,
    ):
        from .polarity_ops import (
            assign_polarity,
            compute_mode_index,
            detect_zero_crossings,
            summarize_dominant_component,
        )

        pol = assign_polarity(signed_source, polarity_threshold=polarity_threshold)
        self.polarity = pol
        self.zero_crossings = detect_zero_crossings(
            signed_source,
            pol,
            zero_crossing_threshold=zero_crossing_threshold,
            wraparound_lattice=wraparound_lattice,
        )
        self.mode_index = compute_mode_index(pol, wraparound_lattice=wraparound_lattice)
        if enable_summary:
            self.component_summary = summarize_dominant_component(
                amplitude=self.amplitude,
                phase=self.phase,
                signed_source=signed_source,
                polarity=self.polarity,
                mode_index=self.mode_index,
                zero_crossings=self.zero_crossings,
            )
        else:
            self.component_summary = {}
