"""Validate pitch control against real passes (Metrica events + tracking).

Design (see docs/modelling_decisions.md):
    * Completed passes: events of type PASS in open play (goal kicks and clearances excluded).
    * Failed passes: BALL LOST events whose subtype contains INTERCEPTION (goal-kick
      interceptions excluded). Optionally also plain BALL LOST events with no subtype.
    * For each pass the model is evaluated at the tracking frame where the pass is
      hit (start_frame), at the pass end point, with ball_time = real flight time.
      The score is the passing team's control share there: mine / (mine + theirs).
    * Metrics: AUC (does a higher score mean a completed pass?), Brier score and
      a calibration table, with bootstrap confidence intervals.

Limitation: a failed pass's end point is where the opponent got the ball, not the
intended target, which is not recorded. The test is therefore partly circular for
failed passes (see the write-up).
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from pitchcontrol.kinematics import FRAME_PERIOD_S
from pitchcontrol.surface import frame_baseline, frame_control

NOT_OPEN_PLAY = ("GOAL KICK", "CLEARANCE", "HEAD-CLEARANCE")
_PASS_COLUMNS = ["team", "period", "start_frame", "end_frame", "end_x", "end_y"]


def build_pass_table(events, include_plain_losses=False):
    """Select completed and failed open-play passes from the events table.

    Args:
        events: table from io_metrica.load_events (positions in metres).
        include_plain_losses: also count BALL LOST events with no subtype as failed passes
            (noisier labels, closer to a typical real failure rate).

    Returns:
        New table with team, period, start_frame, end_frame, end_x/end_y (m),
        flight_s (s, frames * 0.04) and completed (bool). Rows with a non-positive
        flight time or missing end position are dropped.
    """
    subtype = events["subtype"].fillna("")
    completed = (events["type"] == "PASS") & ~subtype.isin(NOT_OPEN_PLAY)
    lost = events["type"] == "BALL LOST"
    failed = lost & subtype.str.contains("INTERCEPTION") & ~subtype.str.contains("GOAL KICK")
    if include_plain_losses:
        failed = failed | (lost & (subtype == ""))
    table = events.loc[completed | failed, _PASS_COLUMNS].copy()
    table["completed"] = completed[completed | failed]
    table["flight_s"] = (table["end_frame"] - table["start_frame"]) * FRAME_PERIOD_S
    usable = (table["flight_s"] > 0) & table[["end_x", "end_y"]].notna().all(axis=1)
    return table[usable].reset_index(drop=True)


def logistic_scale(sigma):
    """Logistic scale s (s) whose distribution has standard deviation sigma (s): sigma*sqrt(3)/pi.

    Links the baseline's one parameter to the Spearman model's sigma, so the baseline
    has no free parameter of its own.
    """
    return sigma * np.sqrt(3.0) / np.pi


def passing_team_control(tracking, passes, params, method="spearman"):
    """Passing team's control share at each pass end point, at the frame the pass is hit.

    Args:
        tracking: table from kinematics.add_kinematics indexed by 'frame'.
        passes: table from build_pass_table.
        params: dict with v_max, t_react, sigma, lambda_per_s (see surface.PUBLISHED_PARAMETERS).
        method: 'spearman' (uses ball_time = flight_s) or 'baseline' (no ball time).

    Returns:
        Series aligned with passes: mine / (mine + theirs) in [0, 1]; NaN if the frame is missing.
    """
    if method not in ("spearman", "baseline"):
        raise ValueError("method must be 'spearman' or 'baseline'")
    scores = np.full(len(passes), np.nan)
    for i, p in enumerate(passes.itertuples()):
        if p.start_frame not in tracking.index:
            continue
        row, point = tracking.loc[p.start_frame], np.array([[p.end_x, p.end_y]])
        if method == "spearman":
            home, away = frame_control(row, point, params["v_max"], params["t_react"],
                                       params["sigma"], params["lambda_per_s"], p.flight_s)
        else:
            home, away = frame_baseline(row, point, params["v_max"], params["t_react"],
                                        logistic_scale(params["sigma"]))
        mine, theirs = (home, away) if p.team == "Home" else (away, home)
        scores[i] = (mine / (mine + theirs))[0]
    return pd.Series(scores, index=passes.index)


def auc(scores, labels):
    """Area under the ROC curve: chance a random completed pass scores above a random failed one.

    Ties count half. Returns NaN if only one class is present.
    """
    scores, labels = np.asarray(scores, dtype=float), np.asarray(labels, dtype=int)
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores)
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def brier(prob, labels):
    """Mean squared error between predicted probabilities and 0/1 outcomes (lower is better)."""
    return float(np.mean((np.asarray(prob, dtype=float) - np.asarray(labels, dtype=float)) ** 2))


def calibration_table(prob, labels, n_bins=10):
    """Mean predicted probability versus observed completion rate in equal-width bins of [0, 1]."""
    prob, labels = np.asarray(prob, dtype=float), np.asarray(labels, dtype=float)
    bins = np.minimum((prob * n_bins).astype(int), n_bins - 1)
    rows = [{"bin": b, "mean_predicted": prob[bins == b].mean(),
             "observed_rate": labels[bins == b].mean(), "count": int((bins == b).sum())}
            for b in range(n_bins) if (bins == b).any()]
    return pd.DataFrame(rows)


def bootstrap_ci(metric, prob, labels, n_boot=1000, seed=0, level=0.95):
    """Percentile bootstrap confidence interval for metric(prob, labels).

    Resamples passes with replacement n_boot times; resamples with an undefined metric
    (e.g. a single class) are skipped. Returns (low, high).
    """
    prob, labels = np.asarray(prob, dtype=float), np.asarray(labels)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(prob), len(prob))
        values.append(metric(prob[idx], labels[idx]))
    tail = 100 * (1 - level) / 2
    low, high = np.nanpercentile(values, [tail, 100 - tail])
    return float(low), float(high)
