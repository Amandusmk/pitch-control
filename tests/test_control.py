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


def test_full_teams_never_overshoot_full_control():
    # 22 players all arrive together: total rate is 22 * lambda, so dt * rate is far above 1.
    # A plain Euler step overshoots and blows up here; the exponential step must not.
    is_a = np.array([True] * 11 + [False] * 11)
    arrival = np.full((22, 3), 2.0)
    arrival[11:, 2] = 1.0                          # point 2: team B clearly earlier
    lambdas = np.full(22, LAMBDA_PER_S)
    p_a, p_b = spearman_control(arrival, is_a, SIGMA_S, lambdas, 0.0)
    assert ((p_a >= 0) & (p_b >= 0) & (p_a + p_b <= 1.0 + 1e-9)).all()
    assert p_a[0] == pytest.approx(p_b[0], abs=1e-9)   # tie: teams equal by symmetry
    assert p_a[0] + p_b[0] > 0.99                       # integration stops at 99% controlled
    assert p_b[2] > 0.9


def test_baseline_is_even_when_tied_and_decisive_when_one_team_is_far_faster():
    arrival = np.array([[2.0, 2.0, 1.0], [5.0, 9.0, 11.0], [2.0, 12.0, 11.0]])
    p_a, p_b = baseline_control(arrival, np.array([True, False, False]), s=1.0)
    assert p_a[0] == pytest.approx(0.5, abs=1e-9) and p_b[0] == pytest.approx(0.5, abs=1e-9)
    assert p_a[2] > 0.99
    assert (p_a + p_b) == pytest.approx(1.0)


def test_a_later_ball_arrival_lets_a_slower_team_contest_the_point():
    # Team A arrives at 2 s, team B at 5 s. If the ball is there at 0 s, A takes the point
    # before B arrives; if the ball only arrives at 5 s, B has arrived too and shares it.
    arrival = [[2.0], [5.0]]
    early_a, early_b = _run(arrival, [True, False], ball_time=0.0)
    late_a, late_b = _run(arrival, [True, False], ball_time=5.0)
    assert early_b[0] < 0.05
    assert late_b[0] > early_b[0] + 0.2 and late_a[0] > late_b[0]
    assert late_a[0] + late_b[0] > 0.99


def test_nan_inputs_are_rejected_instead_of_returning_zero_control():
    arrival = np.array([[2.0], [3.0]])
    lambdas = np.full(2, LAMBDA_PER_S)
    with pytest.raises(ValueError):
        spearman_control(arrival, np.array([True, False]), SIGMA_S, lambdas, float("nan"))
    with pytest.raises(ValueError):
        baseline_control(np.array([[np.nan], [3.0]]), np.array([True, False]), s=1.0)
