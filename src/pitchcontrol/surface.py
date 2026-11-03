"""Glue: turn one tracking frame into pitch control values on a grid of points.

Pipeline for one frame: players on the pitch -> time to intercept for each grid
point (pitchcontrol.intercept) -> Spearman time-integrated control
(pitchcontrol.control). All parameters are required arguments; none are defaulted.
"""
import numpy as np

from pitchcontrol.control import spearman_control
from pitchcontrol.coords import PITCH_LENGTH_M, PITCH_WIDTH_M
from pitchcontrol.intercept import time_to_intercept
from pitchcontrol.pitch import team_positions


def make_grid(step_m):
    """Cell-centred grid covering the 105 x 68 m pitch.

    Args:
        step_m: grid spacing in metres (must divide the pitch reasonably evenly).

    Returns:
        (xs, ys, points): 1-D x and y centres in metres, and (len(ys) * len(xs), 2)
        points ordered row by row, so values reshape to (len(ys), len(xs)).
    """
    xs = np.arange(-PITCH_LENGTH_M / 2 + step_m / 2, PITCH_LENGTH_M / 2, step_m)
    ys = np.arange(-PITCH_WIDTH_M / 2 + step_m / 2, PITCH_WIDTH_M / 2, step_m)
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
            velocities.append((row.get(f"{team}_{name}_vx", np.nan),
                               row.get(f"{team}_{name}_vy", np.nan)))
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
    positions, velocities, is_home = _players_on_pitch(row)
    arrival = time_to_intercept(positions, velocities, points, v_max, t_react)
    lambdas = np.full(len(is_home), lambda_per_s)
    return spearman_control(arrival, is_home, sigma, lambdas, ball_time)
