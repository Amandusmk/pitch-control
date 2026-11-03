"""Tests for pitchcontrol.io_metrica: orientation and conversion that must not drift."""
from pathlib import Path

import pandas as pd
import pytest

from pitchcontrol.io_metrica import (
    load_game,
    load_tracking,
    orient_home_attacks_right,
    periods_to_rotate,
)

RAW = Path(__file__).parents[1] / "data" / "raw"


def _game(name, *extra):
    """pytest.param for a downloaded game directory, skipped when its data is missing."""
    skip = pytest.mark.skipif(not (RAW / name).exists(), reason=f"{name} not downloaded")
    return pytest.param(RAW / name, *extra, id=name, marks=skip)


def test_orient_rotates_only_the_given_periods():
    df = pd.DataFrame({"period": [1, 2], "Home_Player1_x": [10.0, 10.0],
                       "Home_Player1_y": [5.0, 5.0]})
    out = orient_home_attacks_right(df, rotate_periods=[2])
    assert out.loc[0, "Home_Player1_x"] == 10.0 and out.loc[0, "Home_Player1_y"] == 5.0
    assert out.loc[1, "Home_Player1_x"] == -10.0 and out.loc[1, "Home_Player1_y"] == -5.0
    assert df.loc[1, "Home_Player1_x"] == 10.0  # input not mutated


def test_periods_to_rotate_finds_the_period_where_home_defends_the_right_goal():
    # Home goalkeeper (largest |mean x|) is at +40 in period 1 and at -40 in period 2.
    df = pd.DataFrame({"period": [1, 1, 2, 2],
                       "Home_Player11_x": [40.0, 38.0, -40.0, -39.0],
                       "Home_Player8_x": [5.0, 6.0, -3.0, -4.0]})
    assert periods_to_rotate(df) == [1]


@pytest.mark.parametrize("game_dir", [_game("game1")])
def test_frame_one_of_game_one_is_converted_to_metres(game_dir):
    # Raw frame 1, home Player11: (0.00082, 0.48238) -> x=-52.414 m, y=+1.198 m (y up).
    row = load_tracking(game_dir).iloc[0]
    assert row["Home_Player11_x"] == pytest.approx(-52.4139, abs=1e-3)
    assert row["Home_Player11_y"] == pytest.approx(1.1982, abs=1e-3)


@pytest.mark.parametrize("game_dir", [_game("game1"), _game("game2")])
def test_home_goalkeeper_is_on_the_left_in_both_halves(game_dir):
    # Core invariant in every game (game 2 is mirrored relative to game 1): after
    # orienting, the home goalkeeper defends x < 0 in both halves.
    df = load_tracking(game_dir)
    for period in (1, 2):
        assert df.loc[df["period"] == period, "Home_Player11_x"].mean() < -30


@pytest.mark.parametrize("game_dir", [_game("game1"), _game("game2")])
def test_events_are_in_metres_and_within_the_pitch(game_dir):
    _, ev = load_game(game_dir)
    passes = ev[ev["type"] == "PASS"].dropna(subset=["start_x", "end_x"])
    # Manual annotations at the line can sit slightly off-pitch (raw values from -0.03 to
    # 1.02, i.e. up to 2.04 m past a touchline), so allow 2.5 m; a wrong scale or a missing
    # flip would break this by far more.
    margin = 2.5
    assert passes["start_x"].abs().max() <= 52.5 + margin
    assert passes["start_y"].abs().max() <= 34.0 + margin


@pytest.mark.parametrize("game_dir, rotated, kept", [_game("game1", 2, 1), _game("game2", 1, 2)])
def test_events_are_rotated_in_the_same_period_as_the_tracking(game_dir, rotated, kept):
    # Game 1: home defends left in period 1, so period 2 is rotated. Game 2 is mirrored:
    # period 1 is rotated. Sign check against the raw file for one event in each period.
    raw = pd.read_csv(next(game_dir.glob("*RawEventsData.csv")))
    _, ev = load_game(game_dir)
    for period, sign in ((rotated, -1.0), (kept, 1.0)):
        i = raw.index[(raw["Period"] == period) & raw["Start X"].notna()][0]
        assert ev.loc[i, "start_x"] == pytest.approx(sign * (raw.loc[i, "Start X"] - 0.5) * 105.0)
        assert ev.loc[i, "start_y"] == pytest.approx(sign * (0.5 - raw.loc[i, "Start Y"]) * 68.0)
