"""Load Metrica Sports sample data (CSV) into metres on the centred pitch.

Source: Metrica Sports sample data (anonymised players and teams),
https://github.com/metrica-sports/sample-data

Output convention (see pitchcontrol.coords): metres, origin at the centre spot,
x along the 105 m length, y up along the 68 m width, and the HOME team attacking
towards +x in both halves.
"""
from pathlib import Path

import pandas as pd

from pitchcontrol.coords import to_metres

_LEAD_COLUMNS = ["period", "frame", "time_s"]
_EVENT_COLUMNS = {
    "Team": "team", "Type": "type", "Subtype": "subtype", "Period": "period",
    "Start Frame": "start_frame", "Start Time [s]": "start_time_s",
    "End Frame": "end_frame", "End Time [s]": "end_time_s",
    "From": "from_player", "To": "to_player",
    "Start X": "start_x", "Start Y": "start_y", "End X": "end_x", "End Y": "end_y",
}


def _tracking_column_names(name_row, team):
    """Build column names from Metrica's third header row.

    A name such as 'Player11' sits above the x column and the y column below
    it is blank. Returns e.g. ['period','frame','time_s','Home_Player11_x',
    'Home_Player11_y', ..., 'ball_x', 'ball_y'].
    """
    names, base = list(_LEAD_COLUMNS), None
    for value in name_row[len(_LEAD_COLUMNS):]:
        if isinstance(value, str) and value:
            base = "ball" if value == "Ball" else f"{team}_{value}"
            names.append(f"{base}_x")
        else:
            names.append(f"{base}_y")
    return names


def _convert_xy_pairs(df, x_cols):
    """Return a copy with each (x, y) pair converted from [0, 1] to metres."""
    out = df.copy()
    for x_col in x_cols:
        y_col = x_col[:-1] + "y"
        out[x_col], out[y_col] = to_metres(df[x_col], df[y_col])
    return out


def _xy_columns(df):
    """Names of all x and y columns in a tracking or event table."""
    return [c for c in df.columns if c.endswith("_x") or c.endswith("_y")]


def home_defends_right(df, period):
    """True if the home team defends the right goal (x > 0) in this period.

    The home goalkeeper is taken to be the home player with the most extreme mean x
    in the period (the goalkeeper stays near his own goal); its sign gives the side.

    Args:
        df: tracking table in metres with a 'period' column and 'Home_*_x' columns.
        period: 1 or 2.
    """
    home_x = [c for c in df.columns if c.startswith("Home_") and c.endswith("_x")]
    mean_x = df.loc[df["period"] == period, home_x].mean()
    return bool(mean_x[mean_x.abs().idxmax()] > 0)


def periods_to_rotate(df):
    """Periods in which the home team defends the right goal and so must be rotated."""
    return [int(p) for p in sorted(df["period"].unique()) if home_defends_right(df, p)]


def orient_home_attacks_right(df, rotate_periods):
    """Rotate the given periods 180 degrees about the centre spot, (x, y) -> (-x, -y).

    After rotating the periods in which home defends the right goal, the home team
    defends x < 0 and attacks x > 0 in every half. Input is not modified.

    Args:
        df: table with a 'period' column and x/y columns in metres.
        rotate_periods: periods to rotate (see periods_to_rotate).
    """
    out = df.copy()
    rows = out["period"].isin(rotate_periods)
    cols = _xy_columns(out)
    out.loc[rows, cols] = -out.loc[rows, cols]
    return out


def read_tracking_team(path, team):
    """Read one team's raw tracking CSV; positions stay in Metrica [0, 1] units.

    Args:
        path: Metrica tracking CSV for one team.
        team: 'Home' or 'Away', used as the column-name prefix.

    Raises:
        ValueError: if the file has more or fewer columns than its header names.
    """
    header = pd.read_csv(path, header=None, nrows=3)
    names = _tracking_column_names(header.iloc[2].tolist(), team)
    data = pd.read_csv(path, header=None, skiprows=3, names=names)
    if data.shape[1] != len(names) or not isinstance(data.index, pd.RangeIndex):
        raise ValueError(f"{path}: columns do not match the header ({len(names)} names)")
    return data


def _find_one(directory, pattern):
    """Return the single file in directory matching the glob pattern, or raise ValueError."""
    matches = sorted(Path(directory).glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {pattern} in {directory}, found {len(matches)}")
    return matches[0]


def _load_tracking_unoriented(game_dir):
    """Home + away tracking in metres, in Metrica's own (not yet oriented) directions."""
    home = read_tracking_team(_find_one(game_dir, "*RawTrackingData_Home_Team.csv"), "Home")
    away = read_tracking_team(_find_one(game_dir, "*RawTrackingData_Away_Team.csv"), "Away")
    if not home[["period", "frame"]].equals(away[["period", "frame"]]):
        raise ValueError("home and away tracking files do not cover the same frames")
    away = away.drop(columns=_LEAD_COLUMNS + ["ball_x", "ball_y"])
    merged = pd.concat([home, away], axis=1)
    x_cols = [c for c in merged.columns if c.endswith("_x")]
    return _convert_xy_pairs(merged, x_cols)


def load_tracking(game_dir):
    """Load a game's home + away tracking into one wide table in metres.

    One row per frame (25 Hz). Columns: period, frame, time_s [s],
    '<Team>_<PlayerNN>_x/_y' [m] (NaN when the player is not on the pitch),
    and ball_x/ball_y [m]. Home attacks towards +x in both halves; the halves
    to rotate are detected from the home goalkeeper's side (games differ).
    """
    unoriented = _load_tracking_unoriented(game_dir)
    return orient_home_attacks_right(unoriented, periods_to_rotate(unoriented))


def load_events(path, rotate_periods):
    """Load the events CSV with snake_case columns and positions in metres.

    Positions (start_x/y, end_x/y) are converted, then the given periods are rotated
    like the tracking data. Frames index into the 25 Hz tracking table.

    Args:
        path: Metrica events CSV.
        rotate_periods: periods to rotate; must match the tracking (see load_game).
    """
    ev = pd.read_csv(path).rename(columns=_EVENT_COLUMNS)
    for prefix in ("start", "end"):
        ev[f"{prefix}_x"], ev[f"{prefix}_y"] = to_metres(ev[f"{prefix}_x"], ev[f"{prefix}_y"])
    return orient_home_attacks_right(ev, rotate_periods)


def load_game(game_dir):
    """Load tracking and events for one game with the same orientation.

    Returns:
        (tracking, events): both in metres with the home team attacking +x in every half.
    """
    unoriented = _load_tracking_unoriented(game_dir)
    rotate = periods_to_rotate(unoriented)
    events = load_events(_find_one(game_dir, "*RawEventsData.csv"), rotate)
    return orient_home_attacks_right(unoriented, rotate), events
