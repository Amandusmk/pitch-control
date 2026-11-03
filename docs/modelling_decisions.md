# Modelling decisions

Every modelling choice in this project, with the options considered and the reason for the
choice. Choices were made by the author; the assistant explained the options and trade-offs.
Data: Metrica Sports sample data (anonymised), game 1 first.

## Conventions (not modelling choices)

- Pitch 105 x 68 m, origin at the centre spot, x along the length, y pointing up.
- Metrica coordinates (0-1, y pointing down) are converted to metres with y flipped.
- Halves are rotated 180 degrees so the home team attacks +x in every half. Which halves need
  rotating is detected from the data (the home goalkeeper's side), because games differ:
  in game 1 the home team defends the left goal in the first half, in game 2 the right.

## 1. Smoothing and velocities

| | |
|---|---|
| Options | raw finite differences; moving average; Savitzky-Golay; Butterworth low-pass |
| Choice | Savitzky-Golay, window 13 frames (0.52 s), polynomial order 2 |
| Why | A local least-squares polynomial whose derivative is the velocity: centred (no lag) and derivable by hand. Window 7 kept frame-to-frame jitter, window 25 flattened real changes in speed (see `scripts/plot_smoothing.py`). Order 2 (a parabola) matches constant acceleration. |
| Cleaning | Samples next to steps faster than 12 m/s are dropped (players only, never the ball). Gaps of at most 12 frames are filled by linear interpolation only if the bridge speed is at most 12 m/s. The first and last 6 frames of every continuous stretch are trimmed, because the window is one-sided there. |
| Limitations | A short displaced plateau (jump away and back) is not caught. Some glitches below 12 m/s (position hold then catch-up) are damped, not removed. |

## 2. Time to intercept

| | |
|---|---|
| Options | straight line at top speed; plus reaction time; project forward by the reaction time then run (Spearman 2018); acceleration-limited |
| Choice | Project forward, then run: p' = p + v t_react, TTI = t_react + \|r - p'\| / v_max |
| Why | Uses the velocities estimated in step 1, is a closed form that can be derived by hand, and is the published baseline. |
| Limitations | No acceleration limit: top speed is reached instantly after the reaction time. A missing velocity is treated as standing still. |

## 3. One top speed for every player

| | |
|---|---|
| Options | one shared v_max; per-player from the data; per-role; shared with a separate goalkeeper |
| Choice | One shared v_max, goalkeepers included |
| Why | Matches the published baseline, keeps the explanation to one sentence, and leaves per-player differences as a testable improvement. The data (99th-percentile speeds per player) shows only the two goalkeepers differ clearly (2.7 and 3.6 m/s against 5.3-7.0 for outfield players). Outfield differences mostly reflect how often a player sprinted, not ability. |
| Limitations | Goalkeepers are credited with outfield speed; the effect should be small because they rarely contest points far from their goal. |

## 4. From arrival times to control

| | |
|---|---|
| Options | logistic on each team's fastest arrival; Spearman time-integrated model; softmax over all players; Gaussian-influence model (not used: it does not use arrival times) |
| Choice | Spearman time-integrated model, with the logistic-on-fastest-arrival model kept as a baseline |
| Why | Every player contributes, control saturates at 1, and the ball arrival time is built in. It is the model the time-to-intercept formula comes from. The baseline shows whether the extra complexity helps. |
| Numerical scheme | An exponential step: within a step the uncontrolled fraction decays as exp(-S dt) with S = sum of f_i lambda_i, shared between players in proportion to f_i lambda_i. A plain Euler step overshoots with 22 players (dt S > 1) and gave nonsense control values; this step cannot overshoot. |
| Ball arrival time | 0 for static pictures; the real pass flight time (from the events data) for validation. |

## 5. Parameters

| Parameter | Value | Source |
|---|---|---|
| v_max | 5 m/s | Spearman (2018) as used in the published tutorial implementation (quoted from memory; verify before citing) |
| t_react | 0.7 s | same |
| sigma | 0.45 s | same |
| lambda | 4.3 per s, one value for all players | same (the published model uses different values for attacking and defending teams; this project uses one) |

Adopted a priori, **not tuned on this project's data**. The observed speeds (99th percentile
5.3-7.0 m/s for outfield players) show v_max = 5 is below what players reach 1-5% of the
time, but the model has no acceleration limit, so an effective speed below sprint speed is
reasonable. This is tested, not assumed, in the sensitivity analysis below.

## 6. Evaluation plan

- No parameter tuning, so no train/test split is needed inside game 1.
- Primary evaluation and sensitivity analysis (v_max in {5, 6, 7}, t_react in {0.3, 0.7}) on game 1.
- Independent replication with the same fixed parameters on game 2, then game 3 (a different
  file format, read with kloppy).
- If any parameter is ever tuned, tune on game 1 and report only on games 2 and 3.

## 7. Validation design

| | |
|---|---|
| Passes | Completed: open-play `PASS` events (goal kicks and clearances excluded). Failed: `BALL LOST` events whose subtype contains INTERCEPTION (goal-kick interceptions excluded). Secondary check: plain `BALL LOST` events also counted as failed. Failed passes are not labelled as passes in Metrica's events; this is how they are recovered. |
| Measurement | The model is evaluated at the frame where the pass is hit, at the pass end point, with ball arrival time = the real flight time (frames x 0.04 s). Score = passing team's control share, mine / (mine + theirs). |
| Metrics | AUC, Brier score and a calibration table with 95% bootstrap intervals, for the Spearman model and the logistic baseline. The baseline's logistic scale is sigma x sqrt(3) / pi, so it has no free parameter of its own. |
| Robustness | Sensitivity grid (v_max in {5, 6, 7} x t_react in {0.3, 0.7}); a no-look-ahead variant (state taken 6 frames before the kick, because the centred velocity filter otherwise uses about 0.24 s of frames after it); the plain-loss definition; replication on a second game. |
| Limitation | A pass's end point is where the ball ended up, so it partly reveals the outcome. For failed passes it is where the opponent took the ball, not the intended target (not recorded). The AUC is therefore optimistic: compare models and settings, not absolute values. |
