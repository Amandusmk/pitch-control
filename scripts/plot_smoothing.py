"""Compare Savitzky-Golay windows on one player's speed and report cleaning statistics.

Usage: python scripts/plot_smoothing.py
Writes outputs/figures/smoothing_comparison.png (git-ignored).
Data: Metrica Sports sample data, game 1.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pitchcontrol.io_metrica import load_tracking  # noqa: E402
from pitchcontrol.kinematics import (  # noqa: E402
    FRAME_PERIOD_S,
    MAX_GAP_FRAMES,
    SMOOTHING_POLYORDER,
    SMOOTHING_WINDOW_FRAMES,
    drop_implausible_steps,
    fill_short_gaps,
    smooth_and_differentiate,
)

WINDOWS = (7, 13, 25)          # frames; 0.28 s, 0.52 s, 1.00 s
POLYORDER = SMOOTHING_POLYORDER
GAP_LIMITS = (5, 12, 25)
HALF_SPAN_S = 10.0
ZOOM_S = 3.0


def player_columns(tracking):
    """Return names like 'Home_Player8' for every player column pair."""
    return [c[:-2] for c in tracking.columns if c.endswith("_x") and c.startswith(("Home_", "Away_"))]


def cleaned_xy(period_df, player, max_gap):
    """Positions (m) after dropping implausible steps and filling gaps up to max_gap frames."""
    x, y = drop_implausible_steps(period_df[player + "_x"].to_numpy(),
                                  period_df[player + "_y"].to_numpy())
    return fill_short_gaps(x, y, max_gap)


def report_cleaning(tracking):
    """Print how much data the step rule removes and how many frames each gap limit fills."""
    dropped = valid = 0
    filled = {g: 0 for g in GAP_LIMITS}
    for player in player_columns(tracking):
        for _, half in tracking.groupby("period"):
            raw = half[player + "_x"].to_numpy()
            xc, yc = drop_implausible_steps(raw, half[player + "_y"].to_numpy())
            valid += int((~np.isnan(raw)).sum())
            dropped += int((np.isnan(xc) & ~np.isnan(raw)).sum())
            for gap in GAP_LIMITS:
                filled[gap] += int((np.isnan(xc) & ~np.isnan(fill_short_gaps(xc, yc, gap)[0])).sum())
    print(f"valid player-frames: {valid}; dropped by 12 m/s step rule: {dropped} "
          f"({100 * dropped / valid:.2f}%)")
    for gap, count in filled.items():
        print(f"  gap limit {gap:>2} frames ({gap * FRAME_PERIOD_S:.2f} s): fills {count} frames")


def find_fast_stretch(tracking):
    """Return (player, row index of peak) for the fastest window-13 smoothed speed, period 1."""
    half = tracking[tracking["period"] == 1]
    best = (None, 0, 0.0)
    for player in player_columns(tracking):
        if not player.startswith("Home"):
            continue
        x, y = cleaned_xy(half, player, MAX_GAP_FRAMES)
        _, _, vx, vy = smooth_and_differentiate(x, y, SMOOTHING_WINDOW_FRAMES, POLYORDER,
                                                trim_edges=True)
        speed = np.hypot(vx, vy)
        if np.isfinite(speed).any() and np.nanmax(speed) > best[2]:
            best = (player, int(np.nanargmax(speed)), float(np.nanmax(speed)))
    return best[0], best[1]


def main():
    """Draw speed versus time for raw differences and three SG windows."""
    tracking = load_tracking(ROOT / "data" / "raw" / "game1")
    report_cleaning(tracking)
    player, peak = find_fast_stretch(tracking)
    half = tracking[tracking["period"] == 1]
    t = half["time_s"].to_numpy()
    x, y = cleaned_xy(half, player, MAX_GAP_FRAMES)
    raw_speed = np.hypot(np.diff(x), np.diff(y)) / FRAME_PERIOD_S

    fig, axes = plt.subplots(2, 1, figsize=(11, 8.5))
    for ax, span in zip(axes, (HALF_SPAN_S, ZOOM_S / 2)):
        ax.plot(t[1:], raw_speed, color="0.6", linewidth=1, label="raw differences")
        for window in WINDOWS:
            _, _, vx, vy = smooth_and_differentiate(x, y, window, POLYORDER, trim_edges=True)
            ax.plot(t, np.hypot(vx, vy), linewidth=1.6,
                    label=f"Savitzky-Golay, window {window} frames ({window * FRAME_PERIOD_S:.2f} s)")
        ax.set_xlim(t[peak] - span, t[peak] + span)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("speed (m/s)")
        ax.grid(alpha=0.3)
    axes[0].set_title(f"{player}, game 1 period 1: speed estimates (polynomial order {POLYORDER})")
    axes[1].set_title("zoom around the peak")
    axes[0].legend(loc="upper left", fontsize=8)
    out = ROOT / "outputs" / "figures" / "smoothing_comparison.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main()
