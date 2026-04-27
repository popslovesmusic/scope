import numpy as np

from core.polarity_ops import assign_polarity, detect_zero_crossings, dominant_polarity, polarity_shift_metric


def test_assign_polarity_three_state():
    x = np.array([-0.1, -0.001, 0.0, 0.002, 0.2])
    pol = assign_polarity(x, polarity_threshold=0.01)
    assert pol.tolist() == [-1, 0, 0, 0, 1]


def test_detect_zero_crossings_adjacent_opposites():
    x = np.array([0.1, -0.2, 0.0, 0.3])
    pol = assign_polarity(x, polarity_threshold=0.01)
    zc = detect_zero_crossings(x, pol, zero_crossing_threshold=0.02, wraparound_lattice=False)
    assert len(zc) == 1
    assert zc[0]["left_index"] == 0
    assert zc[0]["right_index"] == 1


def test_polarity_shift_metric_bounds():
    a = np.array([0, 1, 1, -1])
    b = np.array([0, 1, -1, -1])
    s = polarity_shift_metric(a, b)
    assert 0.0 <= s <= 1.0


def test_dominant_polarity_counts():
    pol = np.array([1, 1, 0, -1])
    assert dominant_polarity(pol) == 1
