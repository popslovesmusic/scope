
import numpy as np

from core.orientation_ops import OPERATORS_BY_NAME


def test_read_then_write_is_inverse_for_stage1_ops():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    for name in ["++", "--", "+-", "-+"]:
        op = OPERATORS_BY_NAME[name]
        y = op.read(x)
        z = op.write(y)
        assert np.allclose(z, x)


def test_ops_are_distinct_on_asymmetric_input():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    pp = OPERATORS_BY_NAME["++"].read(x)
    mm = OPERATORS_BY_NAME["--"].read(x)
    assert not np.allclose(pp, mm)
