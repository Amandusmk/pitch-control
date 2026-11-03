"""Tests for pitchcontrol.validation: pass selection, metrics and passing-team orientation."""
import numpy as np
import pandas as pd
import pytest

from pitchcontrol.surface import PUBLISHED_PARAMETERS
from pitchcontrol.validation import (
    auc,
    bootstrap_ci,
    brier,
    build_pass_table,
    passing_team_control,
)


def _events():
    rows = [
        # team, type, subtype, period, start_frame, end_frame, end_x, end_y
        ("Home", "PASS", None, 1, 100, 130, 10.0, 0.0),                 # completed open play
        ("Home", "PASS", "GOAL KICK", 1, 200, 260, 30.0, 0.0),          # excluded: not open play
        ("Home", "PASS", "CLEARANCE", 1, 300, 330, 30.0, 0.0),          # excluded: not open play
        ("Away", "BALL LOST", "INTERCEPTION", 1, 400, 430, -5.0, 3.0),  # failed pass
        ("Away", "BALL LOST", "CROSS-INTERCEPTION", 1, 500, 540, -5.0, 3.0),  # failed pass
        ("Away", "BALL LOST", "THEFT", 1, 600, 600, 0.0, 0.0),          # not a pass
        ("Away", "BALL LOST", None, 1, 700, 730, 5.0, 5.0),             # plain loss: optional
        ("Home", "PASS", None, 1, 800, 800, 10.0, 0.0),                 # excluded: zero flight time
    ]
    cols = ["team", "type", "subtype", "period", "start_frame", "end_frame", "end_x", "end_y"]
    return pd.DataFrame(rows, columns=cols)


def test_pass_table_keeps_open_play_passes_and_interception_losses_only():
    table = build_pass_table(_events())
    assert list(table.start_frame) == [100, 400, 500]
    assert list(table.completed) == [True, False, False]
    assert list(table.team) == ["Home", "Away", "Away"]
    assert table.flight_s.tolist() == pytest.approx([1.2, 1.2, 1.6])
    with_plain = build_pass_table(_events(), include_plain_losses=True)
    assert list(with_plain.start_frame) == [100, 400, 500, 700]


def test_auc_and_brier_on_known_values():
    labels = np.array([1, 1, 0, 0])
    assert auc(np.array([0.9, 0.8, 0.2, 0.1]), labels) == pytest.approx(1.0)
    assert auc(np.array([0.1, 0.2, 0.8, 0.9]), labels) == pytest.approx(0.0)
    assert auc(np.array([0.5, 0.5, 0.5, 0.5]), labels) == pytest.approx(0.5)
    assert brier(np.array([0.9, 0.1]), np.array([1, 0])) == pytest.approx(0.01)


def test_bootstrap_interval_collapses_when_the_classes_are_perfectly_separated():
    prob = np.array([0.9, 0.8, 0.7, 0.2, 0.1, 0.05])
    labels = np.array([1, 1, 1, 0, 0, 0])
    low, high = bootstrap_ci(auc, prob, labels, n_boot=200, seed=0)
    assert low == pytest.approx(1.0) and high == pytest.approx(1.0)


def test_control_is_read_for_the_passing_team():
    # Home player stands at the target, Away player is far away; both stationary.
    tracking = pd.DataFrame([{
        "frame": 10, "period": 1,
        "Home_Player1_x": 9.0, "Home_Player1_y": 0.0, "Home_Player1_vx": 0.0, "Home_Player1_vy": 0.0,
        "Away_Player2_x": -30.0, "Away_Player2_y": 0.0, "Away_Player2_vx": 0.0, "Away_Player2_vy": 0.0,
    }]).set_index("frame")
    passes = pd.DataFrame({"team": ["Home", "Away"], "start_frame": [10, 10],
                           "end_x": [10.0, 10.0], "end_y": [0.0, 0.0], "flight_s": [1.0, 1.0],
                           "completed": [True, False]})
    for method in ("spearman", "baseline"):
        control = passing_team_control(tracking, passes, PUBLISHED_PARAMETERS, method)
        assert control.iloc[0] > 0.95       # Home passes to a point Home controls
        assert control.iloc[1] < 0.05       # the same point is nearly lost for an Away pass


def test_inputs_that_would_silently_misattribute_passes_are_rejected():
    tracking = pd.DataFrame({"frame": [1, 1], "period": [1, 1]}).set_index("frame")
    passes = pd.DataFrame({"team": ["Home"], "start_frame": [1], "end_x": [0.0],
                           "end_y": [0.0], "flight_s": [1.0], "completed": [True]})
    with pytest.raises(ValueError):               # duplicate frames: .loc would return two rows
        passing_team_control(tracking, passes, PUBLISHED_PARAMETERS)
    unique = tracking.iloc[:1]
    with pytest.raises(ValueError):               # a typo in the team must not become "Away"
        passing_team_control(unique, passes.assign(team="home"), PUBLISHED_PARAMETERS)


def test_a_pass_whose_frame_has_no_players_is_skipped_not_fatal():
    tracking = pd.DataFrame({"frame": [1], "period": [1]}).set_index("frame")   # no player columns
    passes = pd.DataFrame({"team": ["Home"], "start_frame": [1], "end_x": [0.0],
                           "end_y": [0.0], "flight_s": [1.0], "completed": [True]})
    scores = passing_team_control(tracking, passes, PUBLISHED_PARAMETERS)
    assert scores.isna().all()


def test_bootstrap_of_a_single_class_sample_is_nan_not_an_error():
    low, high = bootstrap_ci(auc, np.array([0.9, 0.8]), np.array([1, 1]), n_boot=20)
    assert np.isnan(low) and np.isnan(high)
