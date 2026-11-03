"""Tests for pitchcontrol.pitch: geometry and absent-player handling."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pitchcontrol.pitch import draw_pitch, team_positions


def test_draw_pitch_covers_the_105_by_68_pitch_with_equal_aspect():
    ax = draw_pitch()
    left, right = ax.get_xlim()
    bottom, top = ax.get_ylim()
    assert left <= -52.5 and right >= 52.5 and bottom <= -34.0 and top >= 34.0
    assert ax.get_aspect() == 1.0
    plt.close(ax.figure)


def test_team_positions_skips_players_who_are_not_on_the_pitch():
    row = pd.Series({"Home_Player1_x": 1.0, "Home_Player1_y": 2.0,
                     "Home_Player12_x": np.nan, "Home_Player12_y": np.nan,
                     "Away_Player25_x": 5.0, "Away_Player25_y": 6.0})
    assert team_positions(row, "Home") == {"Player1": (1.0, 2.0)}
    assert team_positions(row, "Away") == {"Player25": (5.0, 6.0)}
