"""Forecast GTA VI trophy earn rates on Exophase at a case's target.

Model, all on the logit scale:

* Denominator -- players tracked.  Tracked players grow like a power law
  in (1 + hours since release): log P is linear in log(1 + t).  The slope
  is fitted by least squares over the last 24 h of reads and shrunk toward
  a launch-week prior of 1.0.  When the history is perfectly flat (no
  refresh has been seen yet) the power law has nothing to fit, so a step
  model takes over: the page is overdue a refresh, and each expected
  refresh multiplies the tracked base by the launch-night jump factor
  (638 -> 3,868 players, about 6.1x) with a ~4.5 h refresh interval.

* Numerator -- earners per trophy.  Earners also grow like a power law in
  (1 + t).  The slope is fitted per trophy over the last 24 h of reads and
  shrunk toward a prior read from the trophy's place in the game: its tier
  and its description.  Early-story bronze (opening chapters, first
  missions, basic mechanics) earn fast; side strands and mid-game content
  earn at a middling pace; the story finale, completion grinds and the
  platinum earn slowest.  Secret trophies are mostly story beats, so the
  same description rules apply to them.

* Median -- rate(target) = earners(target) / players(target).  When the
  history is flat the denominator jump dominates, so medians sit below the
  last rate (refresh dilution: new players arrive with ~0 trophies before
  the existing base earns much).  Over a long horizon the two forces fight
  trophy by trophy: common early trophies rise (progression wins), rare
  late trophies fall (dilution wins).

* Spread -- standard normal z-scores around the median on the logit
  scale.  The logit sd grows with sqrt(horizon in days), with the size of
  the expected dilution (log players growth), with flat history (no
  signal), with the residual noise of the trophy's own trend fit, and a
  little extra for platinum/gold.  A trophy whose last rate is 100% gets a
  fixed narrow band just under 100 instead of the generic spread.

Only the standard library is used.  No randomness, no wall clock, no
network; output rows are sorted by trophy id, so two runs are identical.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# Launch-night refresh facts, from the kit's brief: the page sat unchanged
# at 638 players tracked from 23:48Z to 04:00Z, then refreshed to 3,868.
# Used only as the prior when the history shows no refresh at all.
PRIOR_JUMP = 3868.0 / 638.0
PRIOR_INTERVAL_HOURS = 4.5
PRIOR_DENOM_SLOPE = 1.0

# Window (hours) over which trends are fitted.
FIT_WINDOW_HOURS = 24.0

# Spread coefficients (logit sd).
BASE_SIGMA_PER_SQRT_DAY = 0.5
DILUTION_SIGMA = 0.30
FLAT_HISTORY_SIGMA = 0.5
NOFIT_SIGMA = 0.25
PLATINUM_EXTRA_SIGMA = 0.15
GOLD_EXTRA_SIGMA = 0.10
SIGMA_MIN = 0.25
SIGMA_MAX = 2.5

# Inverse normal CDF (Acklam's rational approximation), so any level in a
# case file maps to its z-score instead of hard-coding five numbers.
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
    """Standard normal quantile at p (0 < p < 1)."""
    if p <= 0.0 or p >= 1.0:
        raise ValueError("norm_ppf needs 0 < p < 1")
    if p < _P_LOW:
        q = math.sqrt(-2.0 * math.log(p))
        num = ((((_ACK_C[0] * q + _ACK_C[1]) * q + _ACK_C[2]) * q
                + _ACK_C[3]) * q + _ACK_C[4]) * q + _ACK_C[5]
        den = (((_ACK_D[0] * q + _ACK_D[1]) * q + _ACK_D[2]) * q
               + _ACK_D[3]) * q + 1.0
        return num / den
    if p > _P_HIGH:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        num = ((((_ACK_C[0] * q + _ACK_C[1]) * q + _ACK_C[2]) * q
                + _ACK_C[3]) * q + _ACK_C[4]) * q + _ACK_C[5]
        den = (((_ACK_D[0] * q + _ACK_D[1]) * q + _ACK_D[2]) * q
               + _ACK_D[3]) * q + 1.0
        return -num / den
    q = p - 0.5
    r = q * q
    num = ((((_ACK_A[0] * r + _ACK_A[1]) * r + _ACK_A[2]) * r
            + _ACK_A[3]) * r + _ACK_A[4]) * r + _ACK_A[5]
    den = ((((_ACK_B[0] * r + _ACK_B[1]) * r + _ACK_B[2]) * r
            + _ACK_B[3]) * r + _ACK_B[4]) * r + 1.0
    return num * q / den


def clamp_pct(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit_pct(pct: float) -> float:
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def inv_logit_pct(lg: float) -> float:
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
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def ols(xs, ys):
    """Least squares y = a + b*x; returns (intercept, slope, resid_sd)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0.0:
        return my, 0.0, 0.0
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = my - slope * mx
    if n > 2:
        resid = math.sqrt(sum((y - (intercept + slope * x)) ** 2
                              for x, y in zip(xs, ys)) / (n - 2))
    else:
        resid = 0.0
    return intercept, slope, resid


def median(values):
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def earners_prior(trophy: dict, last_rate) -> float:
    """Prior slope of log earners vs log(1 + t), from tier + description."""
    text = (trophy.get("description", "") + " " + trophy.get("name", "")).lower()
    tier = trophy.get("tier", "bronze")
    if tier == "platinum":
        return 0.35
    if "100% completion" in text:
        return 0.50
    if "chapter 1" in text or "chapter 2" in text or "gone fishing" in text:
        return 1.90
    if "chapter 3" in text or "chapter 4" in text:
        return 1.60
    if "chapter" in text:
        return 1.20
    if "final mission for" in text:
        return 1.10
    if "final mission" in text:
        return 0.50
    if "try each type of activity" in text:
        return 1.70
    if "discover 100 vehicles" in text:
        return 1.30
    if ("250 stunts" in text or "50 challenges" in text
            or "3 collectible card sets" in text or "100 locations" in text
            or "every spot" in text or "explore all of leonida" in text):
        return 0.55
    if "gold medal" in text:
        return 0.70
    if "8 mission strands" in text:
        return 0.90
    if "5 bank vaults" in text:
        return 0.80
    if "top fun drone race" in text:
        return 0.80
    if "meditate" in text:
        return 0.60
    if "laundering" in text:
        return 1.00
    if "spend $100,000" in text:
        return 1.10
    if "garage properties" in text:
        return 0.90
    base = {"bronze": 1.20, "silver": 0.90, "gold": 0.60}.get(tier, 1.00)
    if last_rate is not None:
        if last_rate < 0.7:
            base *= 0.85
        elif last_rate > 1.5:
            base *= 1.10
    return min(2.0, max(0.30, base))


def project_players(case, reads, x_last, x_target, horizon_hours):
    """Project players tracked at the target. Returns (p_target, p_last,
    flat_history, log_growth)."""
    last = reads[-1]
    p_last = float(last["players_tracked"])
    flat = all(r["players_tracked"] == last["players_tracked"] for r in reads)

    # Power-law fit over the last 24 h of reads.
    window = [r for r in reads if r["t"] >= reads[-1]["t"] - FIT_WINDOW_HOURS]
    if len(window) < 2:
        window = reads
    xs = [r["x"] for r in window]
    ys = [math.log(float(r["players_tracked"])) for r in window]
    distinct = len({r["players_tracked"] for r in window})
    if distinct >= 2 and len(window) >= 3:
        _, slope_fit, _ = ols(xs, ys)
        k = min(4, distinct - 1)
        slope = (k * slope_fit + 2.0 * PRIOR_DENOM_SLOPE) / (k + 2.0)
    else:
        slope = PRIOR_DENOM_SLOPE
    slope = min(2.0, max(0.2, slope))
    p_smooth = p_last * math.exp(slope * (x_target - x_last))

    if flat:
        # No refresh seen yet: the page is overdue one.  Step model.
        stale = reads[-1]["t"] - reads[0]["t"]
        expected = (stale + horizon_hours) / PRIOR_INTERVAL_HOURS
        p_step = p_last * (PRIOR_JUMP ** expected)
        p_target = max(p_smooth, p_step)
        return p_target, p_last, True, math.log(p_target / p_last)

    # Non-flat: if the page is stale relative to its own refresh rhythm,
    # one more refresh is partly due before the target.
    change_times = []
    jump_ratios = []
    for prev, cur in zip(reads, reads[1:]):
        if cur["players_tracked"] != prev["players_tracked"]:
            change_times.append(cur["t"])
            jump_ratios.append(float(cur["players_tracked"])
                               / float(prev["players_tracked"]))
    p_target = p_smooth
    if len(change_times) >= 1:
        gaps = [b - a for a, b in zip(change_times, change_times[1:])]
        interval = median(gaps) if gaps else PRIOR_INTERVAL_HOURS
        interval = max(interval, 0.5)
        stale = reads[-1]["t"] - change_times[-1]
        if stale > 0.75 * interval and jump_ratios:
            jump = median(jump_ratios[-3:])
            extra = min(1.0, stale / interval - 0.75)
            p_target = p_smooth * (jump ** extra)
    return p_target, p_last, False, math.log(p_target / p_last)


def forecast_case(case: dict, name: str) -> dict:
    levels = case["levels"]
    z_scores = [norm_ppf(float(tau)) for tau in levels]
    trophies = case["trophies"]
    history = case["history"]
    if not history:
        raise ValueError(f"{name} has no history")

    release_text = case.get("game", {}).get("release_date", "2026-11-19")
    release = datetime.fromisoformat(release_text).replace(tzinfo=timezone.utc)
    target = parse_time(case["target"])
    horizon_hours = case["horizon_minutes"] / 60.0
    horizon_days = case["horizon_minutes"] / 1440.0

    reads = []
    for read in history:
        at = parse_time(read["at"])
        t_hours = (at - release).total_seconds() / 3600.0
        reads.append({
            "t": t_hours,
            "x": math.log(max(1.0 + t_hours, 0.05)),
            "players_tracked": read["players_tracked"],
            "rates": read["rates"],
        })
    x_last = reads[-1]["x"]
    x_target = math.log(max(1.0 + (target - release).total_seconds() / 3600.0, 0.05))

    p_target, p_last, flat, log_growth = project_players(
        case, reads, x_last, x_target, horizon_hours)

    window = [r for r in reads if r["t"] >= reads[-1]["t"] - FIT_WINDOW_HOURS]
    if len(window) < 2:
        window = reads

    rows = []
    for i, trophy in enumerate(trophies):
        # Last non-null rate for this trophy, with its own anchor read.
        anchor = None
        for read in reversed(reads):
            rate = read["rates"][i]
            if rate is not None:
                anchor = (read, float(rate))
                break
        if anchor is None:
            # No read ever listed this trophy: flat 1% median, wide band.
            lg = logit_pct(1.0)
            sd = SIGMA_MAX
            qs = [round4(min(100.0, max(0.0, inv_logit_pct(lg + z * sd))))
                  for z in z_scores]
            rows.append({"id": trophy["id"], "q": qs})
            continue
        anchor_read, last_rate = anchor
        if last_rate >= 99.99:
            # Every tracked player has it, and it survived the first
            # refresh at exactly 100% (638 -> 3,868 players): keep it just
            # under 100 with a tight band.
            rows.append({"id": trophy["id"],
                         "q": [99.5, 99.85, 99.97, 99.99, 100]})
            continue

        earners_last = max(last_rate / 100.0 * float(anchor_read["players_tracked"]), 0.5)
        x_anchor = anchor_read["x"]

        # Per-trophy earners trend over the last 24 h.
        xs, ys = [], []
        for read in window:
            rate = read["rates"][i]
            if rate is None or rate <= 0:
                continue
            xs.append(read["x"])
            ys.append(math.log(max(rate / 100.0
                                   * float(read["players_tracked"]), 0.5)))
        prior = earners_prior(trophy, last_rate)
        resid_term = NOFIT_SIGMA
        if len(xs) >= 3 and len(set(ys)) > 1:
            _, slope_fit, resid = ols(xs, ys)
            m = min(4, len(xs) - 1)
            slope = (m * slope_fit + 2.0 * prior) / (m + 2.0)
            resid_term = min(0.6, 0.4 * resid)
        else:
            slope = prior
        slope = min(3.0, max(0.0, slope))

        earners_target = earners_last * math.exp(slope * (x_target - x_anchor))
        earners_target = min(earners_target, p_target * 0.999)
        median_rate = 100.0 * earners_target / p_target
        lg = logit_pct(median_rate)

        sd = math.sqrt(
            (BASE_SIGMA_PER_SQRT_DAY * math.sqrt(horizon_days)) ** 2
            + (DILUTION_SIGMA * log_growth) ** 2
            + (FLAT_HISTORY_SIGMA if flat else 0.0) ** 2
            + resid_term ** 2
            + (PLATINUM_EXTRA_SIGMA if trophy.get("tier") == "platinum" else 0.0) ** 2
            + (GOLD_EXTRA_SIGMA if trophy.get("tier") == "gold" else 0.0) ** 2
        )
        sd = min(SIGMA_MAX, max(SIGMA_MIN, sd))

        qs = []
        prev = None
        for z in z_scores:
            q = round4(min(100.0, max(0.0, inv_logit_pct(lg + z * sd))))
            if prev is not None and q < prev:
                q = prev
            qs.append(q)
            prev = q
        rows.append({"id": trophy["id"], "q": qs})

    rows.sort(key=lambda row: row["id"])
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
