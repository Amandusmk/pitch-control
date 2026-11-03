"""Tests for pitchcontrol.coords: the conversions that would silently corrupt results."""
import numpy as np
import pytest

from pitchcontrol.coords import rotate_half_turn, to_metres


def test_centre_and_corners_map_to_metres_with_y_up():
    # Metrica (0,0) is the top-left of the image; after conversion it must be
    # (-52.5, +34): left edge, TOP of the pitch with y pointing up.
    x, y = to_metres(np.array([0.5, 0.0, 1.0, 0.0, 1.0]),
                     np.array([0.5, 0.0, 0.0, 1.0, 1.0]))
    assert x == pytest.approx([0.0, -52.5, 52.5, -52.5, 52.5])
    assert y == pytest.approx([0.0, 34.0, 34.0, -34.0, -34.0])


def test_missing_values_stay_nan():
    x, y = to_metres(np.array([np.nan, 0.5]), np.array([0.5, np.nan]))
    assert np.isnan(x[0]) and np.isnan(y[1])
    assert x[1] == 0.0 and y[0] == 0.0


def test_rotate_half_turn_negates_both_axes():
    x, y = rotate_half_turn(np.array([10.0, -3.0]), np.array([5.0, 20.0]))
    assert x == pytest.approx([-10.0, 3.0])
    assert y == pytest.approx([-5.0, -20.0])
