"""Glue: turn one tracking frame into pitch control values on a grid of points.

Pipeline for one frame: players on the pitch -> time to intercept for each grid
point (pitchcontrol.intercept) -> Spearman time-integrated control
(pitchcontrol.control). All parameters are required arguments; none are defaulted.
"""
from types import MappingProxyType

import numpy as np

from pitchcontrol.control import baseline_control, spearman_control
from pitchcontrol.coords import PITCH_LENGTH_M, PITCH_WIDTH_M
from pitchcontrol.intercept import time_to_intercept
from pitchcontrol.pitch import team_positions

# Parameters of Spearman (2018) as set in `default_model_params` of the public tutorial
# implementation (Friends-of-Tracking-Data-FoTD/LaurieOnTracking, Metrica_PitchControl.py),
# checked against that code. Adopted by the project owner as the fixed, a-priori values:
# NOT tuned on this project's data.
#   v_max 5 m/s, t_react 0.7 s, sigma 0.45 s, lambda 4.3 per s (one value for all players).
# Differences from the tutorial: it gives goalkeepers 3x lambda, and it assumes a 15 m/s
# ball; here lambda is shared by all players and the real pass flight time is used.
PUBLISHED_PARAMETERS = MappingProxyType(
    {"v_max": 5.0, "t_react": 0.7, "sigma": 0.45, "lambda_per_s": 4.3})


def make_grid(step_m):
    """Cell-centred grid covering the 105 x 68 m pitch.

    Args:
        step_m: grid spacing in metres. It must divide both 105 and 68 exactly, so
            in practice 1.0 or 0.5 (105 and 68 share no larger common divisor).

    Returns:
        (xs, ys, points): 1-D x and y centres in metres, and (len(ys) * len(xs), 2)
        points ordered row by row, so values reshape to (len(ys), len(xs)).

    Raises:
        ValueError: if step_m does not divide the pitch exactly (cells would be dropped).
    """
    counts = []
    for length in (PITCH_LENGTH_M, PITCH_WIDTH_M):
        cells = length / step_m
        if step_m <= 0 or not np.isclose(cells, round(cells)):
            raise ValueError(f"step_m={step_m} does not divide the {length} m side exactly")
        counts.append(int(round(cells)))
    xs = -PITCH_LENGTH_M / 2 + step_m / 2 + step_m * np.arange(counts[0])
    ys = -PITCH_WIDTH_M / 2 + step_m / 2 + step_m * np.arange(counts[1])
    grid_x, grid_y = np.meshgrid(xs, ys)
    return xs, ys, np.column_stack([grid_x.ravel(), grid_y.ravel()])


def _players_on_pitch(row):
    """Positions (N, 2) m, velocities (N, 2) m/s and a home flag (N,) for players in this frame.

    Players with NaN position are left out; NaN velocity stays NaN (treated as
    standing still by time_to_intercept).
    """
    positions, velocities, is_home = [], [], []
    for team in ("Home", "Away"):
        for name, xy in team_positions(row, team).items():
            positions.append(xy)
            velocities.append((row[f"{team}_{name}_vx"], row[f"{team}_{name}_vy"]))
            is_home.append(team == "Home")
    return np.array(positions), np.array(velocities, dtype=float), np.array(is_home)


def frame_control(row, points, v_max, t_react, sigma, lambda_per_s, ball_time):
    """Home and away control at each point for one frame.

    Args:
        row: one row of the table from kinematics.add_kinematics (metres, m/s).
        points: (G, 2) target points in metres.
        v_max: shared top speed, m/s.
        t_react: shared reaction time, s.
        sigma: arrival-time uncertainty, s.
        lambda_per_s: control rate shared by all players, 1/s.
        ball_time: time in seconds for the ball to reach the points (0 = static picture).

    Returns:
        (home_control, away_control): (G,) arrays in [0, 1].
    """
    arrival, is_home = _frame_arrivals(row, points, v_max, t_react)
    lambdas = np.full(len(is_home), lambda_per_s)
    return spearman_control(arrival, is_home, sigma, lambdas, ball_time)


def frame_baseline(row, points, v_max, t_react, s):
    """Baseline home and away control at each point for one frame (no ball time).

    Uses the same time-to-intercept as frame_control, then the logistic of the
    difference between each team's fastest arrival (see control.baseline_control).

    Args:
        row, points, v_max, t_react: as in frame_control.
        s: logistic scale in seconds.

    Returns:
        (home_control, away_control): (G,) arrays that sum to 1.
    """
    arrival, is_home = _frame_arrivals(row, points, v_max, t_react)
    return baseline_control(arrival, is_home, s)


def _frame_arrivals(row, points, v_max, t_react):
    """Arrival times (N, G) in seconds and a home flag (N,) for the players in this frame."""
    positions, velocities, is_home = _players_on_pitch(row)
    if len(is_home) == 0:
        raise ValueError("no players on the pitch in this frame")
    return time_to_intercept(positions, velocities, points, v_max, t_react), is_home
