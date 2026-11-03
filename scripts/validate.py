"""Validate pitch control against real passes in one game.

Usage: python scripts/validate.py [game_dir]      (default: data/raw/game1)
Prints the main result, the sensitivity grid and the plain-loss secondary check, and
writes outputs/figures/validation_<game>.png (git-ignored).
Data: Metrica Sports sample data.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pitchcontrol.io_metrica import load_game  # noqa: E402
from pitchcontrol.kinematics import FRAME_PERIOD_S, add_kinematics  # noqa: E402
from pitchcontrol.surface import PUBLISHED_PARAMETERS  # noqa: E402
from pitchcontrol.validation import (  # noqa: E402
    auc,
    bootstrap_ci,
    brier,
    build_pass_table,
    calibration_table,
    passing_team_control,
)

LOOK_BACK_FRAMES = 6   # half the smoothing window (13): the velocity then uses no frame after the kick
CAVEAT = ("NOTE: a pass's end point is where the ball ended up, so it partly reveals the outcome\n"
          "(for failed passes it is where the opponent took the ball, not the intended target).\n"
          "AUC here is therefore optimistic; compare models and settings, not absolute values.")
V_MAX_GRID = (5.0, 6.0, 7.0)
T_REACT_GRID = (0.3, 0.7)
METHODS = ("spearman", "baseline")


def score_passes(tracking, passes, params):
    """Return {method: (scores, labels)} for passes whose start frame exists in the tracking."""
    results = {}
    for method in METHODS:
        scores = passing_team_control(tracking, passes, params, method)
        ok = scores.notna()
        results[method] = (scores[ok].to_numpy(), passes["completed"][ok].to_numpy().astype(int))
    return results


def summarise(scores, labels):
    """Return a one-line text summary: AUC and Brier with 95% bootstrap intervals."""
    a_lo, a_hi = bootstrap_ci(auc, scores, labels)
    b_lo, b_hi = bootstrap_ci(brier, scores, labels)
    return (f"AUC {auc(scores, labels):.3f} [{a_lo:.3f}, {a_hi:.3f}]   "
            f"Brier {brier(scores, labels):.3f} [{b_lo:.3f}, {b_hi:.3f}]")


def print_main(results, passes):
    """Print sample sizes, the constant-rate Brier reference and both models' metrics."""
    labels = results["spearman"][1]
    rate = labels.mean()
    print(f"passes scored: {len(labels)} of {len(passes)} ({int(labels.sum())} completed, "
          f"{int((1 - labels).sum())} failed; completion rate {rate:.3f})")
    print(f"Brier of always predicting the completion rate: {rate * (1 - rate):.3f}")
    for method in METHODS:
        print(f"  {method:9s} {summarise(*results[method])}")


def look_back(passes, frames):
    """Evaluate each pass `frames` earlier: state from before the kick, flight time lengthened."""
    return passes.assign(start_frame=passes["start_frame"] - frames,
                         flight_s=passes["flight_s"] + frames * FRAME_PERIOD_S)


def print_sensitivity(tracking, passes):
    """Print AUC and Brier over the v_max x t_react grid for both models (no intervals)."""
    print("\nSensitivity (AUC / Brier):   v_max  t_react   spearman        baseline")
    for v_max in V_MAX_GRID:
        for t_react in T_REACT_GRID:
            params = {**PUBLISHED_PARAMETERS, "v_max": v_max, "t_react": t_react}
            res = score_passes(tracking, passes, params)
            cells = [f"{auc(*res[m]):.3f} / {brier(*res[m]):.3f}" for m in METHODS]
            print(f"                              {v_max:>4.1f}   {t_react:>5.1f}   "
                  f"{cells[0]}   {cells[1]}")


def plot_results(results, out):
    """Save score histograms (completed vs failed) and a calibration plot for both models."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    scores, labels = results["spearman"]
    bins = np.linspace(0, 1, 21)
    axes[0].hist(scores[labels == 1], bins, alpha=0.6, density=True, label="completed", color="tab:green")
    axes[0].hist(scores[labels == 0], bins, alpha=0.6, density=True, label="failed", color="tab:red")
    axes[0].set(xlabel="passing team's control at the pass end point (Spearman model)",
                ylabel="density", title="Does control separate completed from failed passes?")
    axes[0].legend()
    axes[1].plot([0, 1], [0, 1], "k--", linewidth=1, label="perfectly calibrated")
    for method, colour in zip(METHODS, ("tab:blue", "tab:orange")):
        table = calibration_table(*results[method], n_bins=10)
        axes[1].plot(table["mean_predicted"], table["observed_rate"], "o-", color=colour,
                     label=method)
    axes[1].set(xlabel="predicted probability (passing team's control)",
                ylabel="observed completion rate", title="Calibration (10 equal-width bins)")
    axes[1].legend()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(out)


def main(game_dir):
    """Run the whole validation for one game directory."""
    game_dir = Path(game_dir)
    tracking, events = load_game(game_dir)
    tracking = add_kinematics(tracking).set_index("frame")
    passes = build_pass_table(events)
    print(f"== {game_dir.name}: main definition (open-play passes vs interception losses) ==")
    results = score_passes(tracking, passes, PUBLISHED_PARAMETERS)
    print_main(results, passes)
    print_sensitivity(tracking, passes)
    print(f"\n== no look-ahead: state taken {LOOK_BACK_FRAMES} frames before the kick ==")
    print_main(score_passes(tracking, look_back(passes, LOOK_BACK_FRAMES), PUBLISHED_PARAMETERS), passes)
    print("\n== secondary: plain ball losses also counted as failed passes ==")
    with_plain = build_pass_table(events, include_plain_losses=True)
    print_main(score_passes(tracking, with_plain, PUBLISHED_PARAMETERS), with_plain)
    print("\n" + CAVEAT)
    plot_results(results, ROOT / "outputs" / "figures" / f"validation_{game_dir.name}.png")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ROOT / "data" / "raw" / "game1")
