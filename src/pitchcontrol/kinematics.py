"""Clean player positions and estimate velocities from 25 Hz tracking.

Pipeline for one player over one half (all positions in metres, time in seconds):
    1. drop_implausible_steps  - discard samples next to physically impossible jumps.
    2. fill_short_gaps         - linearly interpolate short gaps if the bridge is plausible.
    3. smooth_and_differentiate - Savitzky-Golay filter on each continuous stretch.

Do NOT apply step 1 to the ball: passes and shots legitimately exceed 12 m/s.
"""
import numpy as np
from scipy.signal import savgol_filter

FRAME_PERIOD_S = 0.04          # s between frames at 25 Hz
MAX_PLAYER_SPEED_MS = 12.0     # m/s; faster steps between frames are treated as tracking errors


def _true_runs(mask):
    """Return (starts, ends) of consecutive True runs in a boolean mask (end exclusive)."""
    padded = np.concatenate(([False], mask, [False])).astype(int)
    edges = np.flatnonzero(np.diff(padded))
    return edges[::2], edges[1::2]


def drop_implausible_steps(x, y, max_speed=MAX_PLAYER_SPEED_MS, dt=FRAME_PERIOD_S):
    """Set to NaN every sample adjacent to a step faster than max_speed.

    Args:
        x, y: positions in metres, one value per frame (NaN = missing).
        max_speed: largest believable player speed in m/s.
        dt: seconds between frames.

    Returns:
        New (x, y) arrays; inputs are not modified.

    Assumption: a step above max_speed means one of its two end samples is wrong
    and we cannot tell which, so both are discarded. A run of consecutive bad
    steps (e.g. interpolated across a jump) removes the whole run plus its ends.
    Limitation: a short displaced plateau (jump away, then jump back) keeps its
    middle samples, because only the two end steps are flagged.
    """
    step_speed = np.hypot(np.diff(x), np.diff(y)) / dt  # m/s between frame i and i+1
    bad_step = np.nan_to_num(step_speed, nan=0.0) > max_speed
    bad_sample = np.zeros(len(x), dtype=bool)
    bad_sample[:-1] |= bad_step
    bad_sample[1:] |= bad_step
    return np.where(bad_sample, np.nan, x), np.where(bad_sample, np.nan, y)


def fill_short_gaps(x, y, max_gap, max_speed=MAX_PLAYER_SPEED_MS, dt=FRAME_PERIOD_S):
    """Linearly interpolate short gaps in a player's path, if the bridge is plausible.

    Args:
        x, y: positions in metres, one value per frame (NaN = missing, in both).
        max_gap: longest gap to fill, in frames (1 frame = 0.04 s).
        max_speed: a gap is only bridged if the straight-line speed between its
            two bounding samples is at most this, in m/s.
        dt: seconds between frames.

    Returns:
        New (x, y) arrays. Gaps longer than max_gap, gaps whose bridge would
        need more than max_speed (the player "jumped"), and NaNs at either end
        stay NaN: we never extrapolate and never draw a line across a jump.
    """
    out_x, out_y = np.array(x, dtype=float), np.array(y, dtype=float)
    starts, ends = _true_runs(np.isnan(out_x) | np.isnan(out_y))
    for start, end in zip(starts, ends):
        if start == 0 or end == len(out_x) or end - start > max_gap:
            continue
        distance = np.hypot(out_x[end] - out_x[start - 1], out_y[end] - out_y[start - 1])
        if distance / ((end - start + 1) * dt) > max_speed:
            continue
        frames = np.arange(start, end)
        out_x[start:end] = np.interp(frames, [start - 1, end], [out_x[start - 1], out_x[end]])
        out_y[start:end] = np.interp(frames, [start - 1, end], [out_y[start - 1], out_y[end]])
    return out_x, out_y


def smooth_and_differentiate(x, y, window, polyorder, dt=FRAME_PERIOD_S):
    """Savitzky-Golay smoothed positions and velocities, per continuous stretch.

    A least-squares polynomial of degree polyorder is fitted to each window of
    `window` frames; the smoothed position is the polynomial's value at the
    window centre and the velocity is its derivative there.

    Args:
        x, y: positions in metres (NaN = missing).
        window: odd number of frames in the fitting window (7 frames = 0.28 s).
        polyorder: polynomial degree, less than window.
        dt: seconds between frames.

    Returns:
        (x_smooth, y_smooth, vx, vy): metres and metres per second, NaN where a
        sample is missing or lies in a stretch shorter than `window`. The filter
        never crosses a gap. Near each stretch's ends scipy fits the polynomial
        to the first/last window (mode='interp'), which is less reliable.
    """
    n = len(x)
    results = [np.full(n, np.nan) for _ in range(4)]
    present = ~(np.isnan(x) | np.isnan(y))
    for start, end in zip(*_true_runs(present)):
        if end - start < window:
            continue
        for coord, (pos_out, vel_out) in zip((x, y), ((results[0], results[2]),
                                                      (results[1], results[3]))):
            segment = coord[start:end]
            pos_out[start:end] = savgol_filter(segment, window, polyorder)
            vel_out[start:end] = savgol_filter(segment, window, polyorder, deriv=1, delta=dt)
    return tuple(results)
