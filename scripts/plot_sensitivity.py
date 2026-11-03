"""Show how pitch control on one real frame changes with each parameter.

Usage: python scripts/plot_sensitivity.py [frame]
Writes outputs/figures/sensitivity.png (git-ignored). Red = home control, blue = away.
The first panel uses the values published with Spearman (2018) as a REFERENCE ONLY;
each other panel changes one thing. Data: Metrica Sports sample data, game 1.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pitchcontrol.io_metrica import load_tracking  # noqa: E402
from pitchcontrol.kinematics import add_kinematics  # noqa: E402
from pitchcontrol.pitch import plot_frame  # noqa: E402
from pitchcontrol.surface import frame_control, make_grid  # noqa: E402

DEFAULT_FRAME = 20000
GRID_STEP_M = 2.0
REFERENCE = {"v_max": 5.0, "t_react": 0.7, "sigma": 0.45, "lambda_per_s": 4.3, "ball_time": 0.0}
VARIATIONS = [
    ("reference (Spearman 2018)", {}),
    ("faster players: v_max 7", {"v_max": 7.0}),
    ("quicker reaction: t_react 0.3", {"t_react": 0.3}),
    ("more arrival uncertainty: sigma 0.9", {"sigma": 0.9}),
    ("slower control: lambda 1.5", {"lambda_per_s": 1.5}),
    ("ball takes 2 s to arrive", {"ball_time": 2.0}),
]


def draw_panel(ax, row, xs, ys, points, overrides, title):
    """Draw one control surface with the players on top; return the home share (0-1)."""
    params = {**REFERENCE, **overrides}
    home, _ = frame_control(row, points, **params)
    surface = home.reshape(len(ys), len(xs))
    ax.imshow(surface, origin="lower", extent=[xs[0] - GRID_STEP_M / 2, xs[-1] + GRID_STEP_M / 2,
                                               ys[0] - GRID_STEP_M / 2, ys[-1] + GRID_STEP_M / 2],
              cmap="RdBu_r", vmin=0, vmax=1, alpha=0.75, zorder=0)
    plot_frame(row, ax)
    share = float(surface.mean())
    ax.set_title(f"{title}\nhome controls {100 * share:.0f}% of the pitch", fontsize=9)
    return share


def main(frame):
    """Save a 3 x 2 figure of control surfaces for one frame under varied parameters."""
    tracking = add_kinematics(load_tracking(ROOT / "data" / "raw" / "game1")).set_index("frame")
    row = tracking.loc[frame]
    xs, ys, points = make_grid(GRID_STEP_M)
    fig, axes = plt.subplots(3, 2, figsize=(14, 13))
    for ax, (title, overrides) in zip(axes.ravel(), VARIATIONS):
        print(f"{title}: home share {draw_panel(ax, row, xs, ys, points, overrides, title):.3f}")
    fig.suptitle(f"Game 1, frame {frame} (period {int(row['period'])}, t={row['time_s']:.1f} s): "
                 "red = home control, blue = away; home attacks right", fontsize=11)
    out = ROOT / "outputs" / "figures" / "sensitivity.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_FRAME)
