"""Coordinate conversion from Metrica's normalised system to pitch metres.

Pitch frame used throughout this project:
    * 105 m long (x) by 68 m wide (y), origin at the centre spot.
    * x runs from -52.5 (left goal line) to +52.5 (right goal line).
    * y runs from -34 (bottom touchline) to +34 (top touchline), pointing UP,
      as in an ordinary maths plot.

Metrica's normalised system puts (0, 0) at the top-left of the pitch image and
(1, 1) at the bottom-right, so its y-axis points DOWN and must be flipped.
"""
import numpy as np

PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0


def to_metres(x_norm, y_norm):
    """Convert Metrica coordinates in [0, 1] to metres on the centred pitch.

    Args:
        x_norm: x in [0, 1], 0 = left goal line, 1 = right goal line (scalar or array).
        y_norm: y in [0, 1], 0 = top touchline, 1 = bottom touchline (scalar or array).

    Returns:
        (x, y) in metres, origin at the centre, y pointing up. NaN stays NaN.

    Assumption: the tracked pitch is exactly 105 x 68 m. Metrica's data does not
    state the true pitch size, so this is a modelling assumption, not a measurement.
    """
    x = (np.asarray(x_norm, dtype=float) - 0.5) * PITCH_LENGTH_M
    y = (0.5 - np.asarray(y_norm, dtype=float)) * PITCH_WIDTH_M
    return x, y

