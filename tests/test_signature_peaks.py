
import numpy as np

from core.peak_detector import detect_peaks_and_bands


def test_detects_local_maxima_and_bands():
    amp = np.array([0.0, 0.1, 0.9, 0.2, 0.0, 0.7, 0.05, 0.0, 0.0, 0.6, 0.1, 0.0])
    peaks, bands = detect_peaks_and_bands(amp, min_height=0.2, min_distance=1, merge_radius=1, band_rel_threshold=0.5)
    peak_idx = sorted([p.family for p in peaks])
    assert peak_idx == [2, 5, 9]
    assert all(b.width >= 1 for b in bands)
