# Pitch control project

Goal: build a pitch control model from football tracking data, test it against
real passes, and present it clearly. The author is a maths master's student
applying for a student role at a football-tech company, so they must be able to
explain every modelling choice themselves.

## Rules
- Before each modelling choice (smoothing, speeds, time-to-intercept,
  parameters), explain the options and trade-offs, and let the user decide.
- Keep all functions small, with docstrings stating units and assumptions.
- Python only. Use numpy/pandas/scipy/matplotlib. Pitch is 105 x 68 m,
  origin at the centre. Metrica coordinates run 0-1 and must be converted.
- Never commit data. Credit Metrica Sports as the data source.
- Write a test for each module. Run tests before saying something works.
- Look at every plot you make and check it before reporting it done.

## Token economy
- Use ECC agents/skills (tdd-guide, python-reviewer) but keep tests few and
  important: cover the behaviour that would silently corrupt results
  (coordinate conversion, half-time flip, velocity units, control-surface
  invariants). Skip exhaustive or cosmetic tests.
- Run only the tests relevant to the module being changed.

## Data
Metrica sample data. Start with game 1 only (CSV, 25 fps); generalise to game 2
and game 3 (EPTS, via kloppy) later. Anonymised players and teams. Lives in
`data/` (git-ignored).

## Layout
`src/pitchcontrol/` package, `tests/`, `scripts/`, `notebooks/`, `docs/`, `outputs/`.
