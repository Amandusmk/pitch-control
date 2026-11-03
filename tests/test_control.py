"""Tests for pitchcontrol.control: invariants a pitch control model must satisfy."""
import numpy as np
import pytest

from pitchcontrol.control import baseline_control, spearman_control

# Illustrative values for testing only; the project's real parameters are chosen separately.
SIGMA_S = 0.45
LAMBDA_PER_S = 4.3


def _run(arrival, is_a, ball_time=0.0):
    lambdas = np.full(len(is_a), LAMBDA_PER_S)
    return spearman_control(np.array(arrival, float), np.array(is_a), SIGMA_S, lambdas, ball_time)


def test_equal_arrival_times_split_control_evenly_and_it_sums_to_about_one():
    # One player per team, both arrive at 2 s at point 0; at point 1 team A is 1.5 s earlier.
    p_a, p_b = _run([[2.0, 1.0], [2.0, 2.5]], [True, False])
    assert p_a[0] == pytest.approx(p_b[0])
    assert p_a[1] > 0.9 and p_b[1] < 0.1
    assert (p_a + p_b) == pytest.approx([1.0, 1.0], abs=0.02)


def test_a_team_with_no_rival_gets_the_whole_point_and_the_other_gets_nothing():
    p_a, p_b = _run([[3.0], [4.0]], [True, True])
    assert p_b[0] == 0.0
    assert p_a[0] == pytest.approx(1.0, abs=0.02)


def test_unstable_time_step_is_rejected():
    # dt * lambda >= 1 would make the Euler update overshoot past full control.
    with pytest.raises(ValueError):
        spearman_control(np.array([[1.0], [1.0]]), np.array([True, False]), SIGMA_S,
                         np.array([30.0, 30.0]), 0.0, dt=0.04)


def test_baseline_is_even_when_tied_and_decisive_when_one_team_is_far_faster():
    arrival = np.array([[2.0, 2.0, 1.0], [5.0, 9.0, 11.0], [2.0, 12.0, 11.0]])
    p_a, p_b = baseline_control(arrival, np.array([True, False, False]), s=1.0)
    assert p_a[0] == pytest.approx(0.5, abs=1e-9) and p_b[0] == pytest.approx(0.5, abs=1e-9)
    assert p_a[2] > 0.99
    assert (p_a + p_b) == pytest.approx(1.0)
