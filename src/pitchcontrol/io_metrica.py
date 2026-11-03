"""Load Metrica Sports sample data (CSV) into metres on the centred pitch.

Source: Metrica Sports sample data (anonymised players and teams),
https://github.com/metrica-sports/sample-data

Output convention (see pitchcontrol.coords): metres, origin at the centre spot,
x along the 105 m length, y up along the 68 m width, and the HOME team attacking
towards +x in both halves.
"""
from pathlib import Path

import pandas as pd

from pitchcontrol.coords import rotate_half_turn, to_metres

TRACKING_FRAME_RATE_HZ = 25.0
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


def orient_home_attacks_right(df):
    """Rotate second-half (period 2) positions 180 degrees about the centre.

    Teams swap ends at half-time; rotating period 2 makes the home team defend
    x < 0 and attack x > 0 in both halves. Input is not modified.

    Args:
        df: table with a 'period' column and x/y columns in metres.
    """
    out = df.copy()
    second_half = out["period"] == 2
    cols = _xy_columns(out)
    out.loc[second_half, cols] = -out.loc[second_half, cols]
    return out


def read_tracking_team(path, team):
    """Read one team's raw tracking CSV; positions stay in Metrica [0, 1] units."""
    header = pd.read_csv(path, header=None, nrows=3)
    names = _tracking_column_names(header.iloc[2].tolist(), team)
    return pd.read_csv(path, header=None, skiprows=3, names=names)


def load_tracking(game_dir):
    """Load a game's home + away tracking into one wide table in metres.

    One row per frame (25 Hz). Columns: period, frame, time_s [s],
    '<Team>_<PlayerNN>_x/_y' [m] (NaN when the player is not on the pitch),
    and ball_x/ball_y [m]. Home attacks towards +x in both halves.
    """
    game_dir = Path(game_dir)
    home = read_tracking_team(game_dir / "Sample_Game_1_RawTrackingData_Home_Team.csv", "Home")
    away = read_tracking_team(game_dir / "Sample_Game_1_RawTrackingData_Away_Team.csv", "Away")
    away = away.drop(columns=_LEAD_COLUMNS + ["ball_x", "ball_y"])
    merged = pd.concat([home, away], axis=1)
    x_cols = [c for c in merged.columns if c.endswith("_x")]
    return orient_home_attacks_right(_convert_xy_pairs(merged, x_cols))


def load_events(path):
    """Load the events CSV with snake_case columns and positions in metres.

    Positions (start_x/y, end_x/y) are converted and oriented like the tracking
    data. Frames index into the 25 Hz tracking table.
    """
    ev = pd.read_csv(path).rename(columns=_EVENT_COLUMNS)
    for prefix in ("start", "end"):
        ev[f"{prefix}_x"], ev[f"{prefix}_y"] = to_metres(ev[f"{prefix}_x"], ev[f"{prefix}_y"])
    second_half = ev["period"] == 2
    cols = ["start_x", "start_y", "end_x", "end_y"]
    ev.loc[second_half, cols] = -ev.loc[second_half, cols]
    return ev
