"""Render a short clip of one real pass with the pitch control surface underneath.

Usage: python scripts/make_clip.py [game_dir] [--start-frame N]
Default game_dir is data/raw/game1. Without --start-frame, a failed open-play pass is
chosen automatically: a long enough flight (1.4-2.6 s) whose passing team had a low but
not hopeless control share at the end point when the ball was kicked.
Writes outputs/clips/pass_<game>_<frame>.gif and outputs/figures/pass_<game>_<frame>_kick.png
(both git-ignored). Red = home control, blue = away. Data: Metrica Sports sample data.
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pitchcontrol.io_metrica import load_game  # noqa: E402
from pitchcontrol.kinematics import FRAME_PERIOD_S, add_kinematics  # noqa: E402
from pitchcontrol.pitch import plot_frame  # noqa: E402
from pitchcontrol.surface import PUBLISHED_PARAMETERS, frame_control, make_grid  # noqa: E402
from pitchcontrol.validation import build_pass_table, passing_team_control  # noqa: E402

GRID_STEP_M = 1.0
PAD_FRAMES = 12            # 0.5 s shown before the kick and after the ball arrives
DRAW_EVERY = 2             # draw every 2nd frame: 12.5 pictures per second
FPS = 12
FLIGHT_RANGE_S = (1.4, 2.6)
SCORE_RANGE = (0.15, 0.35)


def pick_failed_pass(tracking, passes):
    """Return the failed pass (a table row) with a mid-range score inside the ranges above."""
    failed = passes[~passes["completed"]].copy()
    failed["score"] = passing_team_control(tracking, failed, PUBLISHED_PARAMETERS)
    ok = failed[failed["flight_s"].between(*FLIGHT_RANGE_S) & failed["score"].between(*SCORE_RANGE)]
    if ok.empty:
        raise SystemExit("no pass matches the selection ranges; pass --start-frame instead")
    closest = (ok["score"] - np.mean(SCORE_RANGE)).abs().to_numpy().argmin()
    return ok.iloc[closest]


def risk_at(tracking, p, frame):
    """Passing team's control at the end point if the ball were still in flight at this frame."""
    remaining = max(int(p["end_frame"]) - frame, 1) * FRAME_PERIOD_S
    now = pd.DataFrame([{"team": p["team"], "start_frame": frame, "end_x": p["end_x"],
                         "end_y": p["end_y"], "flight_s": remaining}])
    return float(passing_team_control(tracking, now, PUBLISHED_PARAMETERS).iloc[0])


def draw(ax, tracking, p, frame, grid):
    """Draw one picture: control surface, players, ball, the pass and a live read-out."""
    xs, ys, points = grid
    row = tracking.loc[frame]
    home, away = frame_control(row, points, **PUBLISHED_PARAMETERS, ball_time=0.0)
    ax.clear()
    half = GRID_STEP_M / 2
    ax.imshow((home / (home + away)).reshape(len(ys), len(xs)), origin="lower", cmap="RdBu_r",
              vmin=0, vmax=1, alpha=0.75, zorder=0,
              extent=[xs[0] - half, xs[-1] + half, ys[0] - half, ys[-1] + half])
    plot_frame(row, ax)
    kick = tracking.loc[int(p["start_frame"])]
    ax.annotate("", xy=(p["end_x"], p["end_y"]), xytext=(kick["ball_x"], kick["ball_y"]),
                arrowprops={"arrowstyle": "->", "color": "black", "lw": 1.6, "ls": "--"}, zorder=6)
    # White with a black edge: gold gets approximated away by the GIF's 256-colour palette.
    ax.scatter([p["end_x"]], [p["end_y"]], marker="*", s=280, color="white",
               edgecolor="black", linewidth=1.3, zorder=7)
    seconds = (frame - int(p["start_frame"])) * FRAME_PERIOD_S
    outcome = "completed" if p["completed"] else "intercepted"
    ax.set_title(f"{p['team']} pass, {outcome}  |  t = {seconds:+.2f} s from the kick  |  "
                 f"passing team's control at the star: {risk_at(tracking, p, frame):.2f}",
                 fontsize=9)
    ax.text(0, -36.2, "red = home control, blue = away control   |   dashed arrow = the pass, "
            "star = where the ball ended up", ha="center", va="center", fontsize=8, zorder=8)


def main(game_dir, start_frame):
    """Build the clip and the kick-moment still for one pass."""
    game_dir = Path(game_dir)
    tracking, events = load_game(game_dir)
    tracking = add_kinematics(tracking).set_index("frame")
    passes = build_pass_table(events)
    if start_frame is None:
        p = pick_failed_pass(tracking, passes)
    else:
        p = passes[passes["start_frame"] == start_frame].iloc[0]
    kick_frame, end_frame = int(p["start_frame"]), int(p["end_frame"])
    print(f"{game_dir.name}: {p['team']} pass at frame {kick_frame}, flight {p['flight_s']:.2f} s, "
          f"{'completed' if p['completed'] else 'failed'}")
    grid = make_grid(GRID_STEP_M)
    frames = list(range(kick_frame - PAD_FRAMES, end_frame + PAD_FRAMES + 1, DRAW_EVERY))
    tag = f"{game_dir.name}_{kick_frame}"
    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    draw(ax, tracking, p, kick_frame, grid)
    still = ROOT / "outputs" / "figures" / f"pass_{tag}_kick.png"
    still.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(still, dpi=110, bbox_inches="tight")
    clip = ROOT / "outputs" / "clips" / f"pass_{tag}.gif"
    clip.parent.mkdir(parents=True, exist_ok=True)
    FuncAnimation(fig, lambda f: draw(ax, tracking, p, f, grid), frames=frames).save(
        clip, writer=PillowWriter(fps=FPS), dpi=75)
    print(f"{len(frames)} pictures\n{still}\n{clip}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("game_dir", nargs="?", default=str(ROOT / "data" / "raw" / "game1"))
    parser.add_argument("--start-frame", type=int, default=None)
    args = parser.parse_args()
    main(args.game_dir, args.start_frame)
