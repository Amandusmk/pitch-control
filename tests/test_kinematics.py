"""Tests for pitchcontrol.kinematics: velocity units, glitch removal and gap rules."""
import numpy as np
import pytest

from pitchcontrol.kinematics import (
    drop_implausible_steps,
    fill_short_gaps,
    smooth_and_differentiate,
)

DT = 0.04  # s per frame at 25 Hz


def test_constant_velocity_is_recovered_exactly_in_metres_per_second():
    # x advances 0.08 m/frame = 2.0 m/s; y falls 0.04 m/frame = -1.0 m/s.
    i = np.arange(60)
    x_s, y_s, vx, vy = smooth_and_differentiate(0.08 * i, -0.04 * i, window=7, polyorder=2, dt=DT)
    assert vx == pytest.approx(2.0)
    assert vy == pytest.approx(-1.0)
    assert x_s == pytest.approx(0.08 * i)


def test_smoothing_does_not_bridge_a_gap_and_skips_short_segments():
    x = np.arange(40, dtype=float)
    x[15:20] = np.nan          # gap 1 splits the series
    x[26:28] = np.nan          # gap 2 leaves a 6-frame fragment [20:26], shorter than window 7
    y = x.copy()
    _, _, vx, _ = smooth_and_differentiate(x, y, window=7, polyorder=2, dt=DT)
    assert np.isnan(vx[15:28]).all()          # gaps stay NaN and the short fragment is dropped
    assert not np.isnan(vx[:15]).any() and not np.isnan(vx[28:]).any()


def test_drop_implausible_steps_removes_a_teleport_but_keeps_normal_running():
    x = np.cumsum(np.full(30, 0.2))            # 5 m/s running
    y = np.zeros(30)
    # Ramp glitch like those in the data: 5 frames moving 1 m/frame = 25 m/s, then back on track.
    x[10:14] = x[9] + 1.0 * np.arange(1, 5)
    xc, yc = drop_implausible_steps(x, y, max_speed=12.0, dt=DT)
    assert np.isnan(xc[9:15]).all() and np.isnan(yc[9:15]).all()
    assert not np.isnan(xc[:9]).any() and not np.isnan(xc[15:]).any()


def test_fill_short_gaps_only_fills_gaps_inside_data_up_to_the_limit():
    v = np.array([np.nan, 0.1, np.nan, np.nan, 0.4, np.nan, np.nan, np.nan, np.nan, 0.9, np.nan])
    y = np.where(np.isnan(v), np.nan, 0.0)
    out, _ = fill_short_gaps(v, y, max_gap=2)
    assert out[2:4] == pytest.approx([0.2, 0.3])         # 2-frame gap interpolated linearly
    assert np.isnan(out[5:9]).all()                      # 4-frame gap longer than limit
    assert np.isnan(out[0]) and np.isnan(out[10])        # edges are never extrapolated


def test_fill_short_gaps_refuses_to_draw_a_line_across_a_jump():
    # 20 m in 3 frames (0.12 s) would be ~167 m/s: the player teleported, do not bridge.
    x = np.array([0.0, np.nan, np.nan, 20.0])
    out, _ = fill_short_gaps(x, np.where(np.isnan(x), np.nan, 0.0), max_gap=5)
    assert np.isnan(out[1:3]).all()
