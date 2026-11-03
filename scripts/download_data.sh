#!/usr/bin/env bash
# Download the Metrica Sports sample data (games 1 and 2, CSV format) into data/raw/.
# Source: https://github.com/metrica-sports/sample-data (anonymised players and teams).
# Metrica asks that the data is used responsibly and the source acknowledged in anything
# public. The data is git-ignored and must not be committed or redistributed here.
set -euo pipefail

cd "$(dirname "$0")/.."
BASE="https://raw.githubusercontent.com/metrica-sports/sample-data/master/data"

for game in 1 2; do
  mkdir -p "data/raw/game${game}"
  for part in RawTrackingData_Home_Team RawTrackingData_Away_Team RawEventsData; do
    name="Sample_Game_${game}_${part}.csv"
    target="data/raw/game${game}/${name}"
    if [ -f "${target}" ]; then
      echo "already have ${target}"
    else
      echo "downloading ${target}"
      curl -fsSL -o "${target}" "${BASE}/Sample_Game_${game}/${name}"
    fi
  done
done
