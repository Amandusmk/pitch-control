"""Tests for pitchcontrol.intercept: the time-to-intercept formula and its units."""
import numpy as np
import pytest

from pitchcontrol.intercept import time_to_intercept

V_MAX = 5.0      # m/s, illustrative test value only
T_REACT = 0.7    # s, illustrative test value only


def test_time_to_intercept_projects_forward_by_reaction_time_then_runs_at_top_speed():
    # Player at the origin running at 5 m/s along +x. After 0.7 s of reaction time
    # they have drifted to (3.5, 0).
    player = np.array([[0.0, 0.0]])
    velocity = np.array([[5.0, 0.0]])
    targets = np.array([[13.5, 0.0],     # ahead: 10 m left at 5 m/s -> 0.7 + 2.0
                        [-10.0, 0.0],    # behind: 13.5 m at 5 m/s -> 0.7 + 2.7
                        [3.5, 0.0]])     # exactly the drifted position -> just the reaction time
    tti = time_to_intercept(player, velocity, targets, v_max=V_MAX, t_react=T_REACT)
    assert tti.shape == (1, 3)
    assert tti[0] == pytest.approx([2.7, 3.4, 0.7])


def test_missing_velocity_is_treated_as_standing_still():
    player = np.array([[0.0, 0.0]])
    targets = np.array([[10.0, 0.0]])
    nan_v = time_to_intercept(player, np.array([[np.nan, np.nan]]), targets, V_MAX, T_REACT)
    zero_v = time_to_intercept(player, np.array([[0.0, 0.0]]), targets, V_MAX, T_REACT)
    assert nan_v == pytest.approx(zero_v)
    assert nan_v[0, 0] == pytest.approx(0.7 + 10.0 / 5.0)
