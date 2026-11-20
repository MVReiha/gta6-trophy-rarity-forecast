"""GTA VI trophy rarity forecaster.

For every trophy of Grand Theft Auto VI on PlayStation 5, forecasts five
quantiles (5/25/50/75/95%) of its Exophase earn rate at a case's target,
from the house's reads of Exophase's trophy page before the case's
checkpoint. Reads one case file per run from /input (or
$MAKEFUN_INPUT_DIR) and writes one forecast file to /output (or
$MAKEFUN_OUTPUT_DIR).

Model: earn-rate decomposition in log time.
    rate_i(t) = earners_i(t) / players(t)
  - players(t) is the number of players Exophase tracks. It only grows.
    Fitted as a power law in log(1 + hours since launch) over the case's
    distinct reads, then extrapolated to the target.
  - earners_i(t) is the tracked players who hold trophy i. Fitted per
    trophy the same way, anchored at the last read, with the slope shrunk
    toward a tier prior when the history is thin: bronze chapter trophies
    saturate early, gold and platinum keep growing as the cohort plays
    deeper into the game.
  - The median forecast is earners_i(target) / players(target).

Spreads live on the logit scale and combine the fit residual, the
log-time distance of the extrapolation, a base allowance for the model
itself being wrong, and a refresh allowance. Exophase refreshes the
page on its own schedule, so a page that has NEVER changed in a case's
whole history is stale: the next refresh can move every rate at once
(the practice cases straddle exactly such a refresh: 638 players
tracked, frozen for hours, then 3,868). A merely old last read in a
history that refreshes regularly is just the normal cadence and gets
no such treatment. Fits use the last 24 hours of distinct reads: the
recent slope is what the next days follow, not the launch-night
explosion. Everything is anchored at the last distinct state's first
appearance, the best estimate of when the page last actually changed.

Everything here is a pure function of the case file: no randomness, no
network, no clock reads, so two runs on the same case write the same
bytes. Standard library only.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# Quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# Standard normal quantiles at those levels.
Z_SCORES = [-1.6448536269514722, -0.6744897501960817, 0.0,
            0.6744897501960817, 1.6448536269514722]

# Launch of GTA VI, the origin of the log-time axis.
LAUNCH = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# How much faster than the tracked population a trophy's earners grow,
# by tier, when the history is too thin to fit a slope of its own.
TIER_SLOPE_BONUS = {"platinum": 0.6, "gold": 0.45, "silver": 0.3,
                    "bronze": 0.1}
# Earners growth (log-time slope) of the existing cohort alone, used
# when the page never refreshed in the case's history: no population
# growth is visible, so only the old cohort's own progression counts.
# Early trophies grow fastest; late ones barely move.
PROGRESSION_SLOPE = {"platinum": 0.2, "gold": 0.3, "silver": 0.4,
                     "bronze": 0.5}
# Fallback earn rate (percent) for a trophy never seen in any read.
TIER_RATE_PRIOR = {"platinum": 0.3, "gold": 1.5, "silver": 6.0,
                   "bronze": 20.0}


def clamp_rate(pct):
    """Keep an earn rate inside the range the logit can express."""
    return min(99.99, max(0.01, pct))


def logit_of(pct):
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def rate_from_logit(lg):
    return 100.0 / (1.0 + math.exp(-lg))


def round4(x):
    """Round to 4 decimals, halves away from zero, as the house rounds."""
    scaled = x * 10000.0
    base = math.floor(scaled)
    if scaled - base >= 0.5:
        base += 1
    value = base / 10000.0
    return int(value) if value == int(value) else value


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hours_since_launch(moment):
    return (moment - LAUNCH).total_seconds() / 3600.0


def log_time(hours):
    """log(1 + hours), safe for the slightly negative hours before launch."""
    return math.log1p(max(hours, -0.9))


def linfit(xs, ys):
    """Least squares line through (xs, ys): (intercept, slope, resid std)."""
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx <= 1e-12:
        return mean_y, 0.0, 0.0
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    resid = math.sqrt(
        sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys)) / n)
    return intercept, slope, resid


def distinct_states(history):
    """Collapse consecutive reads of the same page into one state.

    Exophase only changes the page when it refreshes, so a run of
    identical reads carries one piece of information. Each state is kept
    at the time of its FIRST read: that is when the page had already
    reached it, and the gap from there to the checkpoint is how stale
    the page is.
    """
    states = []
    for read in history:
        players = read.get("players_tracked")
        rates = read.get("rates")
        if not players or not rates:
            continue
        key = (players, tuple(rates))
        if states and states[-1][2] == key:
            continue
        states.append((read["at"], players, key))
    return states


def saturating_rate_fit(states, idx, t_target):
    """Fit rate(t) = A * (1 - exp(-(t+1)/tau)) to the distinct states.

    A saturating curve in linear time: the rate approaches its asymptote
    A as the cohort plays through the game, and can never pass it. This
    is what keeps a high rate honest near the 100% ceiling, where a
    power-law extrapolation of earners over players would blow straight
    through it. Grid-searches tau; A follows by least squares through
    the origin. Returns the fitted rate at the target (percent), or None
    when the history is too thin or the shape does not fit at all (a
    falling rate, for instance, which dilution can cause and this curve
    cannot describe: then the earners-over-players median stands alone).
    """
    pts = []
    for at, _, (_, rates) in states:
        if idx < len(rates) and rates[idx] is not None:
            pts.append((hours_since_launch(parse_time(at)),
                        rates[idx] / 100.0))
    if len(set(p for _, p in pts)) < 3:
        return None
    mean_y = sum(y for _, y in pts) / len(pts)
    var = sum((y - mean_y) ** 2 for _, y in pts)
    if var <= 1e-12:
        return None
    best = None
    tau = 2.0
    while tau <= 1000.0:
        num = 0.0
        den = 0.0
        for t, y in pts:
            x = 1.0 - math.exp(-(t + 1.0) / tau)
            num += x * y
            den += x * x
        if den > 1e-12:
            a = num / den
            resid = sum((y - a * (1.0 - math.exp(-(t + 1.0) / tau))) ** 2
                        for t, y in pts)
            if best is None or resid < best[0]:
                best = (resid, a, tau)
        tau *= 1.25
    if best is None or best[0] > 0.35 * var:
        return None
    _, a, tau = best
    a = min(1.0, max(0.001, a))
    return 100.0 * a * (1.0 - math.exp(-(t_target + 1.0) / tau))


def forecast_case(case):
    trophies = case["trophies"]
    history = case.get("history") or []
    checkpoint = parse_time(case["at"])
    target = parse_time(case["target"])
    t_target = hours_since_launch(target)

    states = distinct_states(history)
    if not states:
        # No reads at all: fall back to tier priors with a wide spread.
        rows = []
        for trophy in trophies:
            med = TIER_RATE_PRIOR.get(trophy.get("tier"), 5.0)
            lg = logit_of(med)
            rows.append({"id": trophy["id"], "q": [
                round4(rate_from_logit(lg + z * 1.2)) for z in Z_SCORES]})
        return {"trophies": rows}

    # Anchor at the last distinct state's FIRST appearance: the best
    # estimate of when the page last actually changed. The log-time
    # distance from there to the target is what the forecast travels.
    last_state_at = parse_time(states[-1][0])
    t_anchor = hours_since_launch(last_state_at)
    reach = max(0.0, log_time(t_target) - log_time(t_anchor))

    # --- frozen or overdue? ---
    # Frozen: the page never changed in the whole history (one distinct
    # state). A refresh is pending and it can move every rate at once.
    # Overdue: the page refreshes regularly, but the last state is far
    # older than the usual gap between refreshes. A merely old last read
    # in a regularly refreshing history is normal cadence, not a signal.
    stale_hours = max(0.0, (checkpoint - last_state_at).total_seconds() / 3600.0)
    frozen = len(states) == 1
    gaps = [(parse_time(states[i + 1][0]) - parse_time(states[i][0])
             ).total_seconds() / 3600.0 for i in range(len(states) - 1)]
    typical_gap = sorted(gaps)[len(gaps) // 2] if gaps else 0.0
    overdue = (not frozen and stale_hours > 2.0
               and stale_hours > 3.0 * max(typical_gap, 0.5))

    # --- players(t): power law in log time, fitted on the last 24h ---
    last_players = states[-1][1]
    fit_states = [s for s in states
                  if hours_since_launch(parse_time(s[0])) >= t_anchor - 24.0]
    if len(fit_states) < 3:
        fit_states = states
    xs = [log_time(hours_since_launch(parse_time(at)))
          for at, _, _ in fit_states]
    ys = [math.log(players) for _, players, _ in fit_states]
    _, slope_p, resid_p = linfit(xs, ys)
    if len(set(ys)) < 2:
        slope_p = 1.0  # frozen page: no growth seen, assume launch-curve growth
    if resid_p > 0.4:
        slope_p = 0.5 * slope_p + 0.5 * 1.0  # noisy fit: shrink toward the prior
    slope_p = min(2.5, max(0.0, slope_p))
    players_mult = min(25.0, max(1.0, math.exp(slope_p * reach)))
    players_target = last_players * players_mult
    if overdue:
        players_target = min(25.0 * last_players, players_target * 1.5)

    rows = []
    for idx, trophy in enumerate(trophies):
        tier = trophy.get("tier", "bronze")
        # Last observed rate for this trophy, searching back through states.
        rate_last = None
        for _, _, (_, rates) in reversed(states):
            if idx < len(rates) and rates[idx] is not None:
                rate_last = rates[idx]
                break
        if rate_last is None:
            rate_last = TIER_RATE_PRIOR.get(tier, 5.0)
        earners_last = max(rate_last / 100.0 * last_players, 0.5)

        # --- earners_i(t): same log-time fit, anchored at the last read ---
        # Fit on the last 24h of distinct states, like the players fit.
        ex, ey = [], []
        for at, players, (_, rates) in fit_states:
            if idx < len(rates) and rates[idx] is not None:
                earners = max(rates[idx] / 100.0 * players, 0.5)
                ex.append(log_time(hours_since_launch(parse_time(at))))
                ey.append(math.log(earners))
        slope_e = None
        resid_e = 0.0
        if len(set(ey)) >= 3:
            _, slope_e, resid_e = linfit(ex, ey)
        if frozen:
            # No population growth is visible at all: only the old
            # cohort's own progression moves the earners.
            slope_prior = PROGRESSION_SLOPE.get(tier, 0.5)
        else:
            slope_prior = slope_p + TIER_SLOPE_BONUS.get(tier, 0.1)
        if slope_e is None:
            slope_e = slope_prior
        else:
            weight = len(ey) / (len(ey) + 3.0)  # shrink thin fits to the prior
            slope_e = weight * slope_e + (1.0 - weight) * slope_prior
        slope_e = min(3.0, max(0.0, slope_e))
        earners_target = earners_last * math.exp(slope_e * reach)

        median = clamp_rate(100.0 * earners_target / players_target)

        # Ceiling check: the earners-over-players extrapolation is a
        # power law, and a power law does not know the rate stops at
        # 100%. Where a saturating curve fits the trophy's own history,
        # it caps the median at what the trophy is actually approaching.
        # Taking the lower of the two is the safe side: near the ceiling
        # an overshoot costs far more on the logit scale than an
        # undershoot of the same size.
        sat = saturating_rate_fit(states, idx, t_target)
        if sat is not None:
            median = min(median, sat)

        # --- spread on the logit scale ---
        fit_noise = min(resid_e, 0.6) if len(set(ey)) >= 3 else 0.25
        growth_noise = 0.5 * math.sqrt(reach) + 0.08 * reach
        if frozen:
            refresh_noise = 0.7
        elif overdue:
            refresh_noise = 0.5
        else:
            refresh_noise = 0.15
        tier_noise = 0.5 if rate_last >= 99.5 else (
            0.1 if rate_last < 1.0 else (0.05 if rate_last < 5.0 else 0.0))
        sigma = math.sqrt(fit_noise ** 2 + growth_noise ** 2
                          + refresh_noise ** 2 + tier_noise ** 2 + 0.25 ** 2)
        sigma = min(1.6, max(0.4, sigma))

        if rate_last >= 99.5:
            # A trophy everyone holds stayed at 100% even through the
            # practice refresh's sixfold jump in tracked players: it is
            # effectively universal. Keep the median at the top and let
            # the spread carry the chance that it slips.
            median = 99.9
            sigma = 1.2

        centre = logit_of(median)
        quantiles = [round4(rate_from_logit(centre + z * sigma))
                     for z in Z_SCORES]
        # Rounding can only tie neighbours, never invert them; enforce it.
        for k in range(1, 5):
            if quantiles[k] < quantiles[k - 1]:
                quantiles[k] = quantiles[k - 1]
        rows.append({"id": trophy["id"], "q": quantiles})

    rows.sort(key=lambda row: row["id"])
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        text = json.dumps(forecast, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / (path.stem + ".out")).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
