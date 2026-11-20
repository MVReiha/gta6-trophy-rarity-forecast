# GTA VI Trophy Rarity Forecaster

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/MVReiha/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/MVReiha/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Entry for [make.fun](https://make.fun)'s "Forecast how many GTA VI players earn each trophy in week one" bounty
([gta6-trophy-rarity-forecast-01](https://make.fun/bounty/gta6-trophy-rarity-forecast-01), $250 in USDC):
a Python 3.13 stdlib-only program that forecasts, for each of the 55 GTA VI trophies, the five
quantiles (5/25/50/75/95%) of its Exophase earn rate at 2026-11-26T00:00:00Z, a week after launch.

## How it works

An earn rate is earners divided by players tracked, and the two move in opposite directions:
earners grow as players progress through the game, while players tracked grows as more people
sync their profiles, which dilutes every rate. The forecaster models earners and players
separately, both as saturating growth in log time, with per-tier slope priors (bronze chapter
trophies saturate fast, gold and platinum slow), a saturating-curve cap so high-rate trophies
cannot blow past the 100% ceiling, and quantile spreads on the logit scale that widen with
extrapolation distance and with the chance the page refreshes mid-window (a frozen page that
suddenly refreshes is what breaks narrow bands).

## Run

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=output python3 main.py
```

Reads one case per file `<id>.in` from the input directory and writes `<id>.out` to the output
directory: exactly one row per trophy, five finite non-decreasing quantiles in 0–100.
Deterministic (two runs are byte-identical), offline, standard library only, well under the
8-second limit per case.

## License

MIT
