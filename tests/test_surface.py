"""Tests for pitchcontrol.surface: turning one tracking frame into control values."""
import numpy as np
import pandas as pd
import pytest

from pitchcontrol.surface import PUBLISHED_PARAMETERS, frame_control, make_grid

PARAMS = {**PUBLISHED_PARAMETERS, "ball_time": 0.0}


def test_grid_is_cell_centred_and_covers_the_pitch():
    xs, ys, points = make_grid(step_m=1.0)
    assert xs[0] == pytest.approx(-52.0) and xs[-1] == pytest.approx(52.0)
    assert ys[0] == pytest.approx(-33.5) and ys[-1] == pytest.approx(33.5)
    assert points.shape == (len(xs) * len(ys), 2)


def test_control_favours_the_nearer_team_is_symmetric_and_ignores_absent_players():
    row = pd.Series({
        "Home_Player1_x": -10.0, "Home_Player1_y": 0.0, "Home_Player1_vx": 0.0, "Home_Player1_vy": 0.0,
        "Away_Player2_x": 10.0, "Away_Player2_y": 0.0, "Away_Player2_vx": 0.0, "Away_Player2_vy": 0.0,
        "Home_Player3_x": np.nan, "Home_Player3_y": np.nan,       # substitute, not on the pitch
        "Home_Player3_vx": np.nan, "Home_Player3_vy": np.nan,
    })
    points = np.array([[-20.0, 0.0], [0.0, 0.0], [20.0, 0.0]])
    home, away = frame_control(row, points, **PARAMS)
    assert home[0] > 0.9 and away[2] > 0.9
    assert home[1] == pytest.approx(0.5, abs=0.01) and away[1] == pytest.approx(0.5, abs=0.01)


def test_grid_rejects_a_step_that_would_silently_drop_part_of_the_pitch():
    # 2 m does not divide 105 m: the last 1 m strip would be missing.
    with pytest.raises(ValueError):
        make_grid(step_m=2.0)
