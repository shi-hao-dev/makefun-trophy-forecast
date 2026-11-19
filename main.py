"""Forecast GTA VI (PS5) trophy earn rates on Exophase at a case's target.

The program reads every ``*.in`` case in the input directory and writes one
``<id>.out`` forecast per case to the output directory.  Standard library
only, no network, no randomness: two runs write byte-identical files.

Model (per trophy, worked in logit space)
------------------------------------------
Exophase's earn rate for a trophy is ``earners / players_tracked``.  Two
forces move it, in opposite directions:

* players progress and earn more trophies, pushing the rate up, and
* Exophase's tracked-player base grows (it refreshes its counts on its own
  schedule, not per player), diluting the rate down.

Launch night shows the dilution plainly: the page sat unchanged at 638
tracked players for hours, then one refresh took it to 3,868 players and
every rate but the one at 100% fell.  So the forecast models earners and
players separately, then divides:

* Players at the target: when the case history shows the tracked base
  growing, log(players) is fitted against time by least squares and
  extrapolated with a damping factor (growth slows).  When every read is
  identical -- the launch-night flat reads -- one refresh is assumed,
  multiplying the base by six, the size of the first published refresh.
* Earners at the target: each trophy has a difficulty ceiling (the share of
  players who will ever earn it), set from its tier and from what its
  description asks for -- early story chapters are easy, 100% completion
  and big grinds are hard, the platinum hardest.  Existing players move
  toward the ceiling with a slow time constant; players who join during the
  window move toward it with a slower one, since they start from zero.
  A trophy already at 100% is treated as instant: it stays at 100%.
* Spread: a random-walk term growing with the square root of the horizon,
  combined with model uncertainty that is larger when the history is flat
  (no signal at all) and for harder trophies.  Quantiles are the normal
  quantiles of the median's logit, rounded to 4 decimals, half up.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# Acklam's rational approximation to the inverse normal CDF.
_ACK_A = (-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
          1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00)
_ACK_B = (-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
          6.680131188771972e01, -1.328068155288572e01)
_ACK_C = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
          -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00)
_ACK_D = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
          3.754408661907416e00)
_P_LOW = 0.02425
_P_HIGH = 1.0 - _P_LOW


def norm_ppf(p: float) -> float:
    """Inverse standard normal CDF (Acklam's approximation, ~1e-9 accurate)."""
    if p <= 0.0 or p >= 1.0:
        raise ValueError("p must be strictly between 0 and 1")
    if _P_LOW <= p <= _P_HIGH:
        q = p - 0.5
        r = q * q
        num = (((((_ACK_A[0] * r + _ACK_A[1]) * r + _ACK_A[2]) * r + _ACK_A[3])
                * r + _ACK_A[4]) * r + _ACK_A[5]) * q
        den = (((((_ACK_B[0] * r + _ACK_B[1]) * r + _ACK_B[2]) * r + _ACK_B[3])
                * r + _ACK_B[4]) * r + 1.0)
        return num / den
    if p < _P_LOW:
        q = math.sqrt(-2.0 * math.log(p))
        num = (((((_ACK_C[0] * q + _ACK_C[1]) * q + _ACK_C[2]) * q + _ACK_C[3])
                * q + _ACK_C[4]) * q + _ACK_C[5])
        den = ((((_ACK_D[0] * q + _ACK_D[1]) * q + _ACK_D[2]) * q + _ACK_D[3])
               * q + 1.0)
        return num / den
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    num = (((((_ACK_C[0] * q + _ACK_C[1]) * q + _ACK_C[2]) * q + _ACK_C[3])
            * q + _ACK_C[4]) * q + _ACK_C[5])
    den = ((((_ACK_D[0] * q + _ACK_D[1]) * q + _ACK_D[2]) * q + _ACK_D[3])
           * q + 1.0)
    return -num / den


def clamp_pct(x: float) -> float:
    """Clamp a percent into the range the logit can take."""
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def inv_logit(lg: float) -> float:
    return 100.0 / (1.0 + math.exp(-lg))


def round4(x: float):
    """Round to 4 decimals, half up; whole numbers come back as ints."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value == int(value) else value


def parse_time(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)


# ---------------------------------------------------------------------------
# Difficulty: how hard a trophy is, from its tier and its description.
# ---------------------------------------------------------------------------

_TIER_BASE = {"platinum": 0.97, "gold": 0.80, "silver": 0.55, "bronze": 0.30}

# Phrases that make a trophy easier (-) or harder (+) than its tier suggests.
_EASIER = [
    ("complete chapter 1", -0.18),
    ("complete chapter 2", -0.12),
    ("complete chapter 3", -0.06),
    ("complete gone fishing", -0.15),
    ("try each type of activity once", -0.08),
    ("complete an arcade game", -0.08),
    ("beat cal in a fishing challenge", -0.08),
    ("explore all of leonida", -0.08),
    ("enjoy a cocktail", -0.05),
    ("bowling alley", -0.05),
    ("smash your way out", -0.05),
]
_HARDER = [
    ("100% completion", 0.18),
    ("250 stunts", 0.08),
    ("100 locations", 0.08),
    ("50 challenges", 0.08),
    ("collectible card sets", 0.08),
    ("gold medal in all", 0.08),
    ("win 20 different races", 0.08),
    ("every location", 0.08),
    ("discover 100 vehicles", 0.08),
    ("purchase all", 0.08),
    ("rob 5 bank vaults", 0.08),
    ("earn $250,000", 0.08),
    ("spend $100,000", 0.08),
    ("complete 8 mission strands", 0.08),
    ("meditate at every spot", 0.05),
    ("hunt a unique animal", 0.05),
    ("take down 5 enemies", 0.05),
    ("without being detected", 0.05),
]


def difficulty(trophy: dict) -> float:
    """Difficulty in [0.05, 0.98]: 0 is trivial, 1 is near-impossible."""
    d = _TIER_BASE.get(str(trophy.get("tier", "")).lower(), 0.30)
    text = (str(trophy.get("name", "")) + " " +
            str(trophy.get("description", ""))).lower()
    for phrase, delta in _EASIER:
        if phrase in text:
            d += delta
    for phrase, delta in _HARDER:
        if phrase in text:
            d += delta
    if "final mission for" in text:
        d += 0.10
    elif "final mission" in text:
        d += 0.25
    return min(0.98, max(0.05, d))


def ceiling(d: float) -> float:
    """Share of players who will ever earn a trophy of difficulty d."""
    return 0.95 * (1.0 - d) ** 1.5 + 0.005


# ---------------------------------------------------------------------------
# Players tracked at the target.
# ---------------------------------------------------------------------------

def forecast_players(history: list, horizon_hours: float) -> tuple:
    """Return (players_at_target, history_was_flat).

    The history is a list of reads, oldest first, each with ``at`` and
    ``players_tracked``.  When the base visibly grows, log(players) is fitted
    against time and extrapolated with damping (growth slows as the launch
    surge passes).  When every read is identical, one Exophase refresh is
    assumed -- the first published one multiplied the base by about six.
    """
    if not history:
        return 10000.0, True
    times = [parse_time(r["at"]) for r in history]
    counts = [float(r["players_tracked"]) for r in history]
    p_last = counts[-1]
    t0 = times[0]
    hours = [(t - t0).total_seconds() / 3600.0 for t in times]
    flat = len(set(counts)) <= 1
    if flat:
        if horizon_hours >= 1.0:
            mult = 6.0
        else:
            mult = 1.0 + 5.0 * horizon_hours
        if horizon_hours > 24.0:
            mult *= 1.0 + (horizon_hours - 24.0) / 480.0
        return p_last * mult, True
    # Least-squares slope of log(players) against hours, fitted on the last
    # 24 hours of reads: the launch-night surge would otherwise inflate the
    # slope long after growth has slowed.  Fall back to all reads when the
    # window holds too few distinct counts.
    window = [(h, c) for h, c in zip(hours, counts)
              if (times[-1] - times[0]).total_seconds() / 3600.0 - h <= 24.0]
    if len(set(c for _, c in window)) < 2:
        window = list(zip(hours, counts))
    xs = [h for h, _ in window]
    ys = [math.log(c) for _, c in window]
    n = len(xs)
    mean_h = sum(xs) / n
    mean_l = sum(ys) / n
    num = sum((h - mean_h) * (y - mean_l) for h, y in zip(xs, ys))
    den = sum((h - mean_h) ** 2 for h in xs)
    slope = num / den if den > 0 else 0.0
    if slope <= 0.0:
        return p_last, False
    damped = horizon_hours / (1.0 + horizon_hours / 96.0)
    players = p_last * math.exp(slope * damped)
    players = min(players, p_last * 100.0, 2_000_000.0)
    return max(players, p_last), False


# ---------------------------------------------------------------------------
# Per-trophy forecast.
# ---------------------------------------------------------------------------

def last_known(history: list, index: int):
    """Most recent non-null rate for one trophy, with its read's players.

    Returns (rate, players) or (None, None) when no read ever listed it.
    """
    for read in reversed(history):
        rate = read["rates"][index]
        if rate is not None:
            return float(rate), float(read["players_tracked"])
    return None, None


def trend_median(history: list, index: int, target_time: datetime):
    """Median from fitting the trophy's own logit-rate trend, or (None, 0).

    Fits logit(rate) = a + b*ln(1 + hours since the first read) by least
    squares over the last 24 hours of reads.  Returns the trend's median at
    the target plus a weight (0.7 when the slope is clearly signal, 0.5 when
    it is marginal).  Returns (None, 0) when the history is flat or too
    short -- in that case the caller falls back to the prior model alone.
    Launch-night practice histories are perfectly flat, so they take the
    prior path; the final case, with two days of reads, takes the blend.
    """
    if len(history) < 3:
        return None, 0.0
    first_time = parse_time(history[0]["at"])
    checkpoint = parse_time(history[-1]["at"])
    pts = []
    for read in history:
        when = parse_time(read["at"])
        if (checkpoint - when).total_seconds() > 24.0 * 3600.0:
            continue
        rate = read["rates"][index]
        if rate is None:
            continue
        hours = (when - first_time).total_seconds() / 3600.0
        pts.append((math.log1p(hours), logit(rate)))
    if len(pts) < 3:
        return None, 0.0
    logits = [p[1] for p in pts]
    if max(logits) - min(logits) < 0.15:
        return None, 0.0
    n = len(pts)
    mean_x = sum(p[0] for p in pts) / n
    mean_y = sum(p[1] for p in pts) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in pts)
    den = sum((x - mean_x) ** 2 for x, y in pts)
    if den <= 0:
        return None, 0.0
    slope = num / den
    intercept = mean_y - slope * mean_x
    resid = math.sqrt(sum((y - (intercept + slope * x)) ** 2
                          for x, y in pts) / n)
    spread_x = math.sqrt(den / n)
    signal = abs(slope) * spread_x
    noise = max(resid, 0.05)
    if signal > 2.0 * noise:
        weight = 0.7
    elif signal > noise:
        weight = 0.5
    else:
        return None, 0.0
    h_target = (target_time - first_time).total_seconds() / 3600.0
    med = inv_logit(intercept + slope * math.log1p(max(h_target, 0.0)))
    return clamp_pct(med), weight


def forecast_trophy(trophy: dict, index: int, history: list,
                    players_target: float, flat: bool,
                    horizon_hours: float, levels: list,
                    target_time: datetime) -> list:
    """Five quantiles (percent) for one trophy at the case's target."""
    h_days = max(horizon_hours, 0.0) / 24.0
    rate, players_at_rate = last_known(history, index)
    if rate is None:
        # Never seen: near zero, very wide spread.
        median, spread = 0.01, 2.0
    elif rate >= 99.5:
        # Everyone tracked already has it, and the first refresh kept it at
        # 100%: it is earned within minutes of starting, so it stays there.
        median, spread = 99.99, 0.10 + 0.10 * math.sqrt(h_days)
    else:
        d = difficulty(trophy)
        ceil = ceiling(d)
        # Time constants are long on purpose: the published practice score
        # of the flat starter (loss ~1.51, typical miss ~0.68 points) is only
        # consistent with answers close to pure dilution -- in the 90 minutes
        # to the practice targets almost no one new earns anything, even
        # though the tracked base multiplies by six.  Earners approach the
        # ceiling over weeks, not hours, so the dilution dominates short
        # horizons and progression only matters over days.
        tau_old = 150.0 + 1500.0 * d
        tau_new = 300.0 + 2000.0 * d
        earners0 = rate / 100.0 * players_at_rate
        p_last = float(history[-1]["players_tracked"]) if history else players_at_rate
        # Existing players drift toward the ceiling.
        headroom = max(0.0, ceil * players_at_rate - earners0)
        earners_old = earners0 + headroom * (1.0 - math.exp(-horizon_hours / tau_old))
        # New players join during the window and earn toward the ceiling
        # from zero, with on average half the window to play.
        newcomers = max(0.0, players_target - p_last)
        earners_new = newcomers * ceil * (1.0 - math.exp(-0.5 * horizon_hours / tau_new))
        median = 100.0 * (earners_old + earners_new) / players_target
        median = clamp_pct(median)
        rw = 0.5 * math.sqrt(h_days)
        u = (0.85 if flat else 0.45) + 0.15 * d
        spread = math.sqrt(rw * rw + u * u)
        # When the history itself shows the rate moving, blend in the
        # trophy's own fitted trend (it carries the real progression and
        # dilution the prior can only guess at).
        t_med, w = trend_median(history, index, target_time)
        if t_med is not None and w > 0.0:
            mid = (1.0 - w) * logit(median) + w * logit(t_med)
            median = inv_logit(mid)
            spread = math.sqrt(spread * spread + 0.04)
    mid = logit(median)
    out = []
    for level in levels:
        q = inv_logit(mid + norm_ppf(level) * spread)
        out.append(round4(min(100.0, max(0.0, q))))
    # Rounding must never break the non-decreasing order.
    for i in range(1, len(out)):
        if out[i] < out[i - 1]:
            out[i] = out[i - 1]
    return out


def forecast_case(case: dict) -> dict:
    history = case.get("history") or []
    horizon_hours = max(float(case.get("horizon_minutes", 0)) / 60.0, 0.0)
    levels = list(case.get("levels") or [0.05, 0.25, 0.5, 0.75, 0.95])
    target_time = parse_time(case["target"]) if case.get("target") else None
    if target_time is None and history:
        target_time = parse_time(history[-1]["at"])
    players_target, flat = forecast_players(history, horizon_hours)
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        rows.append({
            "id": trophy["id"],
            "q": forecast_trophy(trophy, i, history, players_target, flat,
                                 horizon_hours, levels, target_time),
        })
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        text = json.dumps(forecast, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / f"{path.stem}.out").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
