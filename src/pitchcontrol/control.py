"""Pitch control: probability that each team controls each point of the pitch.

Inputs are arrival times tau[i, g] in seconds: how long player i needs to reach
grid point g (see pitchcontrol.intercept). Only players on the pitch must be
passed: arrival times must be finite.

spearman_control (the project's model, Spearman 2018, "Beyond Expected Goals"):
    Player i has arrived at point g by time t with probability
        f_i(t) = 1 / (1 + exp(-pi / (sqrt(3) * sigma) * (t - tau_i)))
    (a logistic with standard deviation sigma seconds). Once arrived, a player
    gains control of a still-uncontrolled point at rate lambda_i per second:
        dP_i/dt = (1 - sum_j P_j) * f_i(t) * lambda_i
    Integration starts when the ball arrives (t = ball_time) and runs until
    almost all of the point is controlled. A team's control is the sum of its
    players' P_i.

baseline_control (the simple comparison model): logistic in the difference
    between each team's fastest arrival times.

Numerical assumption: first-order Euler integration with step dt seconds.
No default values are given for sigma, lambda or ball_time: they are modelling
choices made elsewhere.
"""
import numpy as np
from scipy.special import expit

DEFAULT_DT_S = 0.04            # numerical step, equal to one tracking frame
RESIDUAL_TOLERANCE = 0.01      # stop once every point is at least 99% controlled
SETTLE_TIME_S = 10.0           # extra integration time after the slowest point is first reached


def _arrival_probability(t, arrival_times, sigma):
    """Probability that each player has arrived by time t (s): logistic with std sigma (s)."""
    return expit(np.pi / (np.sqrt(3.0) * sigma) * (t - arrival_times))


def _validate(arrival_times, is_team_a, sigma, lambdas, dt):
    """Raise ValueError for inputs that would make the integration meaningless or unstable."""
    if arrival_times.ndim != 2 or len(is_team_a) != arrival_times.shape[0]:
        raise ValueError("arrival_times must be (players, points) matching is_team_a")
    if not np.isfinite(arrival_times).all():
        raise ValueError("arrival_times must be finite; drop players who are off the pitch")
    if sigma <= 0 or dt <= 0 or np.any(lambdas <= 0):
        raise ValueError("sigma, dt and lambdas must be positive")
    if dt * lambdas.max() >= 1.0:
        raise ValueError("dt * lambda >= 1: the Euler step is unstable, reduce dt")


def spearman_control(arrival_times, is_team_a, sigma, lambdas, ball_time, dt=DEFAULT_DT_S):
    """Time-integrated pitch control for two teams at every grid point.

    Args:
        arrival_times: (N, G) seconds for each of N players to reach each of G points.
        is_team_a: (N,) boolean, True for team A players, False for team B.
        sigma: standard deviation of the arrival-time uncertainty, in seconds.
        lambdas: (N,) control rate per player, in 1/s.
        ball_time: time in seconds at which the ball reaches the points (0 for a
            static picture without a pass).
        dt: Euler step in seconds.

    Returns:
        (control_a, control_b): two (G,) arrays in [0, 1]. They sum to at most 1;
        the shortfall is the part of the point not yet controlled when integration stopped.
    """
    tau = np.asarray(arrival_times, dtype=float)
    is_team_a = np.asarray(is_team_a, dtype=bool)
    lambdas = np.asarray(lambdas, dtype=float)
    _validate(tau, is_team_a, sigma, lambdas, dt)
    rate = lambdas[:, None]
    control = np.zeros_like(tau)
    t = ball_time
    t_end = max(ball_time, tau.min(axis=0).max()) + SETTLE_TIME_S
    while t < t_end:
        residual = 1.0 - control.sum(axis=0)
        if residual.max() < RESIDUAL_TOLERANCE:
            break
        control = control + dt * residual * _arrival_probability(t, tau, sigma) * rate
        t += dt
    return control[is_team_a].sum(axis=0), control[~is_team_a].sum(axis=0)


def baseline_control(arrival_times, is_team_a, s):
    """Baseline: P(team A) = logistic((t_B - t_A) / s), with t = each team's fastest arrival.

    Args:
        arrival_times: (N, G) seconds for each player to reach each point.
        is_team_a: (N,) boolean, True for team A players.
        s: logistic scale in seconds (smaller = more decisive).

    Returns:
        (control_a, control_b): two (G,) arrays that sum to exactly 1. A team with
        no players has infinite arrival time and so loses every point.
    """
    tau = np.asarray(arrival_times, dtype=float)
    is_team_a = np.asarray(is_team_a, dtype=bool)
    fastest_a = np.min(tau[is_team_a], axis=0, initial=np.inf)
    fastest_b = np.min(tau[~is_team_a], axis=0, initial=np.inf)
    control_a = expit((fastest_b - fastest_a) / s)
    return control_a, 1.0 - control_a
