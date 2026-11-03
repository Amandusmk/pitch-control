"""Plot two game 1 frames (kick-off and one in the second half) to check orientation.

Usage: python scripts/plot_frame.py [second_half_frame]
Writes outputs/figures/frames_check.png (git-ignored). Data: Metrica Sports sample data.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pitchcontrol.io_metrica import load_tracking  # noqa: E402
from pitchcontrol.pitch import plot_frame  # noqa: E402

DEFAULT_SECOND_HALF_FRAME = 100000


def main(second_half_frame):
    """Save a two-panel figure: frame 1 (period 1) and a period 2 frame."""
    tracking = load_tracking(ROOT / "data" / "raw" / "game1").set_index("frame")
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 13.6))
    for ax, frame in zip(axes, (1, second_half_frame)):
        row = tracking.loc[frame]
        plot_frame(row, ax)
        ax.set_title(f"Game 1, frame {frame} (period {int(row['period'])}, t={row['time_s']:.2f} s)"
                     " - home red, away blue; home attacks right")
    out = ROOT / "outputs" / "figures" / "frames_check.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SECOND_HALF_FRAME)
