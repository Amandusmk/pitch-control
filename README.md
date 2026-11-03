# Pitch control

A pitch control model built from football tracking data, tested against real
passes. Work in progress.

## Data

Tracking and event data from the
[Metrica Sports sample data](https://github.com/metrica-sports/sample-data)
(anonymised players and teams). Credit and thanks to Metrica Sports. The data
is not included in this repository; download it into `data/raw/` yourself and
check the licence in the Metrica repository.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest
```
