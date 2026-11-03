"""Time for a player to reach a point on the pitch (Spearman 2018, option C).

Model, for a player at position p (m) with velocity v (m/s) and a target point r (m):
    1. During the reaction time t_react the player keeps moving at constant velocity:
           p' = p + v * t_react
    2. After that they run in a straight line to r at the constant top speed v_max:
           TTI = t_react + |r - p'| / v_max

Assumptions (the project's modelling choices, to be defended in the write-up):
    * every player, goalkeepers included, shares one v_max and one t_react;
    * no acceleration limit: top speed is reached instantly after the reaction time;
    * straight-line running, no obstacles.
The numerical values of v_max and t_react are deliberately NOT defaulted here.
"""
import numpy as np


def time_to_intercept(positions, velocities, targets, v_max, t_react):
    """Time in seconds for each player to reach each target point.

    Args:
        positions: (N, 2) player positions in metres.
        velocities: (N, 2) player velocities in m/s. NaN (unknown) is treated as
            standing still, so that player's projection p' equals p.
        targets: (G, 2) target points in metres.
        v_max: shared top speed in m/s (must be positive).
        t_react: shared reaction time in seconds (must be non-negative).

    Returns:
        (N, G) array of times in seconds.
    """
    if v_max <= 0 or t_react < 0:
        raise ValueError("v_max must be positive and t_react non-negative")
    positions = np.asarray(positions, dtype=float)
    velocities = np.nan_to_num(np.asarray(velocities, dtype=float), nan=0.0)
    targets = np.asarray(targets, dtype=float)
    projected = positions + velocities * t_react                       # (N, 2) m
    distance = np.linalg.norm(targets[None, :, :] - projected[:, None, :], axis=2)  # (N, G) m
    return t_react + distance / v_max
