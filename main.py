"""GTA VI launch-day Twitch CCV forecaster (gta6-twitch-ccu-forecast-01)
-- agus-ite entry: per-hour ridge regression on size and breadth.

Method
------
The case file ships `comparables`: past big launches' hourly Twitch
category counts. For each forecast hour h this program fits a small
ridge regression of the hour's log ratio on TWO covariates of the
launch's first hour:

    log(viewers_h / viewers_1)  ~  a_h + b_h * x1 + c_h * x2

    x1 = log(viewers_1) - 12          (how big hour 1 already was)
    x2 = log(channels_1 + 1) - 8      (how broad hour 1 was)

The second covariate is the point of this entry: hour 1's channel
count measures how many different streamers were already live, i.e.
whether the audience was spread across the directory or piled onto
a few big streams. A broad hour 1 (many channels, high x2) means the
category had already opened up everywhere and the day grows less;
a narrow hour 1 (few channels for the viewers) means the launch was
still concentrated and the audience fans out as more streamers go
live. The two covariates are correlated, so the fit is ridge
(penalty 2.0 on b_h and c_h, intercept unpenalised), solved in
closed form on the 3x3 normal equations -- no iterative solver.

Spread: no normal assumption at all. The five quantiles are the
empirical 5/25/50/75/95% percentiles of that hour's ridge residuals
(linear interpolation), added to the fitted median; the resulting
offsets are widened just enough that the 5-95% gap is at least 0.40
log units and at most 1.60, so a lucky-tight hour cannot claim
false certainty. Empirical quantiles keep whatever skew the real
launch days have (evenings skew up, overnights skew down), which a
symmetric normal spread would erase.

Only comparables with a real launch hour 1 are used (hour-1 viewers
>= 20,000, relaxed to >= 5,000 then >= 100 if too few qualify;
channels missing for an hour-1 row are replaced by the training
set's median hour-1 channel count). The target game's own
pre-release category history ("Grand Theft Auto VI", day_start
2026-08-27, a premiere not a launch) is excluded.

Fallbacks: if a forecast hour has fewer than 8 training rows the
ridge fit is replaced by the plain median log-ratio of the training
set with the same empirical-spread rule; with no usable comparables
at all, the program falls back to the house persistence baseline
(flat at hour 1, spread 0.15 * sqrt(h - 1)). Deterministic, no
network, no clock reads, standard library only.
"""

import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
LOG_P1_CENTER = 12.0
LOG_CH_CENTER = 8.0
RIDGE = 2.0
MIN_ROWS_REGRESSION = 8
SPREAD_MIN = 0.40   # min 5-95% gap, log units
SPREAD_MAX = 1.60   # max 5-95% gap, log units
BASELINE_SIGMA = 0.15
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]


def round2(x):
    """Round to 2 decimals, half up (as JavaScript Math.round does)."""
    scaled = x * 100.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100.0
    return int(value) if float(value).is_integer() else value


def _quantile(sorted_vals, tau):
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    pos = tau * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def _release_day_rows(comp):
    """(hour-1 viewers, hour-1 channels, {hour: viewers}) for a release day."""
    hourly = comp.get("hourly") or []
    day = (comp.get("day_start") or "")[:10]
    if len(day) != 10:
        return None
    by_hour = {}
    channels = {}
    for row in hourly:
        start = row.get("from") or ""
        if not start.startswith(day):
            continue
        try:
            hour = int(start[11:13]) + 1  # 00:00 UTC -> hour 1
        except (ValueError, IndexError):
            continue
        viewers = row.get("viewers")
        if isinstance(viewers, (int, float)) and viewers > 0:
            by_hour[hour] = float(viewers)
            ch = row.get("channels")
            if isinstance(ch, (int, float)) and ch >= 0:
                channels[hour] = float(ch)
    p1 = by_hour.get(1)
    if p1 is None or p1 < 100:
        return None
    answers = {h: v for h, v in by_hour.items() if 2 <= h <= 24}
    if len(answers) < 20:
        return None
    return p1, channels.get(1), answers


def _training_set(case):
    rows = []
    for comp in case.get("comparables") or []:
        if "Grand Theft Auto VI" in (comp.get("name") or ""):
            continue
        parsed = _release_day_rows(comp)
        if parsed is not None:
            rows.append(parsed)
    # Fill missing hour-1 channels with the training median.
    present = sorted(r[1] for r in rows if r[1] is not None)
    med_ch = present[len(present) // 2] if present else 100.0
    filled = [(p1, (ch if ch is not None else med_ch), ans)
              for p1, ch, ans in rows]
    for threshold in (20000.0, 5000.0, 100.0):
        kept = [r for r in filled if r[0] >= threshold]
        if len(kept) >= MIN_ROWS_REGRESSION:
            return kept
    return filled


def _solve3(matrix, rhs):
    """3x3 Gaussian elimination with partial pivoting; None if singular."""
    n = 3
    aug = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[piv][col]) < 1e-12:
            return None
        aug[col], aug[piv] = aug[piv], aug[col]
        for r in range(col + 1, n):
            f = aug[r][col] / aug[col][col]
            for c in range(col, n + 1):
                aug[r][c] -= f * aug[col][c]
    out = [0.0] * n
    for r in range(n - 1, -1, -1):
        s = aug[r][n] - sum(aug[r][c] * out[c] for c in range(r + 1, n))
        out[r] = s / aug[r][r]
    return out


def _fit_hour(train, p1, ch1, hour):
    """Five log-quantile offsets (from log p1) for one forecast hour."""
    xs = []
    ys = []
    for comp_p1, comp_ch, answers in train:
        if hour in answers:
            xs.append((math.log(comp_p1) - LOG_P1_CENTER,
                       math.log(comp_ch + 1.0) - LOG_CH_CENTER))
            ys.append(math.log(answers[hour] / comp_p1))
    if not xs:
        return None
    x = (math.log(p1) - LOG_P1_CENTER, math.log(ch1 + 1.0) - LOG_CH_CENTER)
    if len(xs) >= MIN_ROWS_REGRESSION:
        k = 3
        ata = [[0.0] * k for _ in range(k)]
        atb = [0.0] * k
        for (x1, x2), y in zip(xs, ys):
            feats = [1.0, x1, x2]
            for i in range(k):
                atb[i] += feats[i] * y
                for j in range(k):
                    ata[i][j] += feats[i] * feats[j]
        ata[1][1] += RIDGE
        ata[2][2] += RIDGE
        beta = _solve3(ata, atb)
        if beta is None:
            return None
        median = beta[0] + beta[1] * x[0] + beta[2] * x[1]
        residuals = sorted(
            y - (beta[0] + beta[1] * a + beta[2] * b)
            for (a, b), y in zip(xs, ys))
    else:
        ordered = sorted(ys)
        n = len(ordered)
        median = (ordered[n // 2] if n % 2
                  else (ordered[n // 2 - 1] + ordered[n // 2]) / 2.0)
        residuals = sorted(y - median for y in ys)
    # Empirical quantiles of the residuals, centred on the fitted median.
    offsets = [_quantile(residuals, tau) + median for tau in LEVELS]
    # Widen/narrow only the tails so the 5-95% gap stays in bounds,
    # keeping the median fixed.
    gap = offsets[4] - offsets[0]
    if gap < SPREAD_MIN or gap > SPREAD_MAX:
        target = max(SPREAD_MIN, min(SPREAD_MAX, gap))
        scale = target / gap if gap > 0 else 1.0
        offsets = [median + (o - median) * scale for o in offsets]
    for i in range(1, 5):
        if offsets[i] < offsets[i - 1]:
            offsets[i] = offsets[i - 1]
    return offsets


def _baseline_offsets(hour):
    return [z * BASELINE_SIGMA * math.sqrt(hour - 1) for z in Z]


def forecast_case(case, name):
    if case.get("levels") != LEVELS:
        raise ValueError("%s asks for levels %s, not %s"
                         % (name, case.get("levels"), LEVELS))
    first = next((k for k in case.get("known", []) if k.get("hour") == 1),
                 None)
    if first is None:
        raise ValueError("%s has no hour-1 print" % name)
    p1 = max(1.0, float(first["viewers"]))
    ch1 = first.get("channels")
    ch1 = float(ch1) if isinstance(ch1, (int, float)) and ch1 >= 0 else 100.0
    train = _training_set(case)
    rows = case.get("forecast") or [{"hour": h} for h in range(2, 25)]
    hours = []
    for row in rows:
        hour = row["hour"]
        offsets = _fit_hour(train, p1, ch1, hour) if train else None
        if offsets is None:
            offsets = _baseline_offsets(hour)
        quantiles = []
        for off in offsets:
            value = p1 * math.exp(max(-4.0, min(5.5, off)))
            value = max(1.0, min(1_000_000_000.0, value))
            quantiles.append(round2(value))
        for i in range(1, 5):
            if quantiles[i] < quantiles[i - 1]:
                quantiles[i] = quantiles[i - 1]
        hours.append({"hour": hour, "q": quantiles})
    hours.sort(key=lambda r: r["hour"])
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / (path.stem + ".out")).write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()
