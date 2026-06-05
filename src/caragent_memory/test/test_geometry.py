import math

from caragent_memory.geometry import yaw_difference_deg, yaw_difference_rad


def test_yaw_difference_wraps_across_pi():
    a = math.radians(179.0)
    b = math.radians(-179.0)
    assert math.isclose(yaw_difference_deg(a, b), 2.0, abs_tol=1e-6)


def test_yaw_difference_is_absolute():
    assert math.isclose(yaw_difference_rad(0.0, math.pi / 2.0), math.pi / 2.0)
