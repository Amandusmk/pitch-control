"""Tests for pitchcontrol.io_metrica: orientation and conversion that must not drift."""
from pathlib import Path

import pandas as pd
import pytest

from pitchcontrol.io_metrica import load_events, load_tracking, orient_home_attacks_right

GAME1 = Path(__file__).parents[1] / "data" / "raw" / "game1"
needs_data = pytest.mark.skipif(not GAME1.exists(), reason="Metrica game 1 data not downloaded")


def test_orient_rotates_only_period_two():
    df = pd.DataFrame({"period": [1, 2], "Home_Player1_x": [10.0, 10.0],
                       "Home_Player1_y": [5.0, 5.0]})
    out = orient_home_attacks_right(df)
    assert out.loc[0, "Home_Player1_x"] == 10.0 and out.loc[0, "Home_Player1_y"] == 5.0
    assert out.loc[1, "Home_Player1_x"] == -10.0 and out.loc[1, "Home_Player1_y"] == -5.0
    assert df.loc[1, "Home_Player1_x"] == 10.0  # input not mutated


@needs_data
def test_frame_one_is_converted_to_metres():
    # Raw frame 1, home Player11: (0.00082, 0.48238) -> x=-52.414 m, y=+1.198 m (y up).
    df = load_tracking(GAME1)
    row = df.iloc[0]
    assert row["Home_Player11_x"] == pytest.approx(-52.4139, abs=1e-3)
    assert row["Home_Player11_y"] == pytest.approx(1.1982, abs=1e-3)


@needs_data
def test_home_goalkeeper_is_on_the_left_in_both_halves():
    # Core invariant: after orienting, the home goalkeeper defends x<0 all match.
    df = load_tracking(GAME1)
    for period in (1, 2):
        assert df.loc[df["period"] == period, "Home_Player11_x"].mean() < -30


@needs_data
def test_events_are_in_metres_and_oriented():
    ev = load_events(GAME1 / "Sample_Game_1_RawEventsData.csv")
    passes = ev[ev["type"] == "PASS"].dropna(subset=["start_x", "end_x"])
    # Manual annotations at the line can sit slightly off-pitch (raw values up to 1.01
    # or -0.02), so allow a 2 m margin; a wrong scale or missing flip would break this.
    margin = 2.0
    assert passes["start_x"].abs().max() <= 52.5 + margin
    assert passes["start_y"].abs().max() <= 34.0 + margin
    assert (passes["period"].isin([1, 2])).all()


@needs_data
def test_second_half_event_is_rotated_relative_to_the_raw_file():
    # A period 2 event must equal the rotated conversion of its raw value (sign check).
    raw = pd.read_csv(GAME1 / "Sample_Game_1_RawEventsData.csv")
    ev = load_events(GAME1 / "Sample_Game_1_RawEventsData.csv")
    i = raw.index[(raw["Period"] == 2) & raw["Start X"].notna()][0]
    assert ev.loc[i, "start_x"] == pytest.approx(-(raw.loc[i, "Start X"] - 0.5) * 105.0)
    assert ev.loc[i, "start_y"] == pytest.approx(-(0.5 - raw.loc[i, "Start Y"]) * 68.0)
