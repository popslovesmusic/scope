
import numpy as np

from core.signature_state import SignatureState
from core.reasoning_loop import run_reasoning
from core.peak_detector import detect_peaks_and_bands
from core.corridor_gate import build_dynamic_corridor
from core.readout_heads import binary_readout


def test_binary_readout_produces_valid_class_and_confidence():
    config = {
        "signature_size": 12,
        "lambda": 0.2,
        "kappa": 0.6,
        "eta": 0.1,
        "phases": 2,
        "peak_threshold": 0.1,
        "family_diffusion": 0.1,
        "polarity_threshold": 0.01,
        "zero_crossing_threshold": 0.02,
        "output_model": {
            "classifier": {
                "mode": "binary",
                "decision_rule": "max_corridor_supported_energy",
                "class_heads": [
                    {"name": "class_0", "readout_family": [0, 1, 2]},
                    {"name": "class_1", "readout_family": [9, 10, 11]},
                ],
            }
        },
    }

    state = SignatureState(config["signature_size"])
    # Patch3: signed_field is primary; amplitude is derived.
    state.signed_field = np.linspace(-1.0, 1.0, config["signature_size"])
    state.derive_amplitude_from_signed(0.0, 1.0)
    trace = run_reasoning(state, config)
    assert len(trace) == 2

    peaks, bands = detect_peaks_and_bands(state.amplitude, min_height=0.05)
    corridor = build_dynamic_corridor(size=12, peaks=peaks, bands=bands, stability=state.stability)
    scores, selected, conf = binary_readout(amplitude=state.amplitude, corridor_window=corridor.window, class_heads=config["output_model"]["classifier"]["class_heads"])
    assert selected in (0, 1)
    assert 0.0 <= conf <= 1.0
    assert len(scores) == 2
