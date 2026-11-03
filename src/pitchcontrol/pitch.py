"""Draw a football pitch and a tracking frame, in metres.

Pitch frame (see pitchcontrol.coords): 105 x 68 m, origin at the centre spot,
x along the length, y up. Markings follow IFAB Laws of the Game dimensions.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Circle, Rectangle

from pitchcontrol.coords import PITCH_LENGTH_M, PITCH_WIDTH_M

HALF_L = PITCH_LENGTH_M / 2
HALF_W = PITCH_WIDTH_M / 2
CENTRE_CIRCLE_RADIUS_M = 9.15
PENALTY_AREA_M = (16.5, 40.32)   # depth from goal line, width
GOAL_AREA_M = (5.5, 18.32)       # depth from goal line, width
PENALTY_SPOT_M = 11.0            # distance from goal line
GOAL_WIDTH_M = 7.32
LINE_STYLE = {"color": "0.35", "linewidth": 1.2, "fill": False}
TEAM_COLOURS = {"Home": "tab:red", "Away": "tab:blue"}


def _box(ax, x0, y0, width, height):
    """Add an unfilled rectangle with lower-left corner (x0, y0), sizes in metres."""
    ax.add_patch(Rectangle((x0, y0), width, height, **LINE_STYLE))


def _draw_end(ax, sign):
    """Draw boxes, spot and arc for the goal at x = sign * 52.5 m (sign = +1 or -1)."""
    goal_line = sign * HALF_L
    for depth, width in (PENALTY_AREA_M, GOAL_AREA_M):
        _box(ax, goal_line - depth if sign > 0 else goal_line, -width / 2, depth, width)
    _box(ax, goal_line if sign > 0 else goal_line - 2.0, -GOAL_WIDTH_M / 2, 2.0, GOAL_WIDTH_M)
    spot_x = goal_line - sign * PENALTY_SPOT_M
    ax.plot(spot_x, 0, marker="o", markersize=2, color=LINE_STYLE["color"])
    # Penalty arc: part of a 9.15 m circle around the spot lying outside the box.
    half_angle = np.degrees(np.arccos((PENALTY_AREA_M[0] - PENALTY_SPOT_M) / CENTRE_CIRCLE_RADIUS_M))
    centre_angle = 180.0 if sign > 0 else 0.0
    ax.add_patch(Arc((spot_x, 0), 2 * CENTRE_CIRCLE_RADIUS_M, 2 * CENTRE_CIRCLE_RADIUS_M,
                     theta1=centre_angle - half_angle, theta2=centre_angle + half_angle,
                     color=LINE_STYLE["color"], linewidth=LINE_STYLE["linewidth"]))


def draw_pitch(ax=None):
    """Draw a 105 x 68 m pitch centred at the origin and return the axes.

    Args:
        ax: matplotlib axes to draw on; a new figure is created if None.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10.5, 6.8))
    _box(ax, -HALF_L, -HALF_W, PITCH_LENGTH_M, PITCH_WIDTH_M)
    ax.plot([0, 0], [-HALF_W, HALF_W], color=LINE_STYLE["color"], linewidth=LINE_STYLE["linewidth"])
    ax.add_patch(Circle((0, 0), CENTRE_CIRCLE_RADIUS_M, **LINE_STYLE))
    for sign in (-1, 1):
        _draw_end(ax, sign)
    ax.set_xlim(-HALF_L - 3, HALF_L + 3)
    ax.set_ylim(-HALF_W - 3, HALF_W + 3)
    ax.set_aspect("equal")
    ax.axis("off")
    return ax


def team_positions(row, team):
    """Return {player_name: (x, y)} in metres for players present in this frame.

    Players with NaN position (substitutes, not yet on the pitch) are omitted.

    Args:
        row: one row of the table from io_metrica.load_tracking.
        team: 'Home' or 'Away'.
    """
    prefix = f"{team}_"
    positions = {}
    for col in row.index:
        if col.startswith(prefix) and col.endswith("_x"):
            name = col[len(prefix):-2]
            x, y = row[col], row[col[:-1] + "y"]
            if not (np.isnan(x) or np.isnan(y)):
                positions[name] = (x, y)
    return positions


def plot_frame(row, ax=None):
    """Draw one tracking frame: home (red), away (blue), ball (black), in metres."""
    ax = draw_pitch(ax)
    for team, colour in TEAM_COLOURS.items():
        for name, (x, y) in team_positions(row, team).items():
            ax.scatter(x, y, s=140, color=colour, edgecolor="white", zorder=3)
            ax.annotate(name.replace("Player", ""), (x, y), ha="center", va="center",
                        fontsize=7, color="white", zorder=4)
    if not (np.isnan(row["ball_x"]) or np.isnan(row["ball_y"])):
        ax.scatter(row["ball_x"], row["ball_y"], s=50, color="black", edgecolor="white", zorder=5)
    return ax
