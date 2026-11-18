# GTA VI launch-day Twitch CCV forecast — size + breadth ridge

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/malik2112/gta6-twitch-ccu-forecast-01-agus-ite/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/malik2112/gta6-twitch-ccu-forecast-01-agus-ite/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Entry for make.fun competition `gta6-twitch-ccu-forecast-01`: forecast how
many people watch Grand Theft Auto VI live on Twitch in each hour of its
release day (2026-11-19 UTC), as five quantiles (5/25/50/75/95%) per hour,
given only everything recorded up to the end of hour 1.

## Program

`main.py` (Python 3.13, standard library only). Run it the way the grader
does:

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

It reads every `<id>.in` in the input directory and writes
`<id>.out` (`{"hours": [{"hour": h, "q": [q05, q25, q50, q75, q95]}, ...]}`,
hours 2–24) to the output directory. Deterministic, no network, well
under the 8 s per-case limit (~0.05 s a case locally).

## Method

The case file ships `comparables`: past big launches' hourly Twitch
category counts. For each forecast hour `h` this entry fits a small
ridge regression of the hour's log ratio on TWO covariates of the
launch's first hour:

```
log(viewers_h / viewers_1)  ~  a_h + b_h·x1 + c_h·x2

x1 = log(viewers_1) − 12        (how big hour 1 already was)
x2 = log(channels_1 + 1) − 8    (how broad hour 1 was)
```

The second covariate is the point of this entry: hour 1's channel
count measures how many different streamers were already live — whether
the audience was spread across the directory or piled onto a few big
streams. A broad hour 1 (many channels) means the category had already
opened up everywhere and the day grows less; a narrow hour 1 (few
channels for the viewers) means the launch was still concentrated and
the audience fans out as more streamers go live. The two covariates
are correlated, so the fit is ridge (penalty 0.5 on the slopes,
intercept unpenalised), solved in closed form on the 3×3 normal
equations.

Spread: no normal assumption at all. The five quantiles are the
empirical 5/25/50/75/95% percentiles of that hour's ridge residuals
(linear interpolation), added to the fitted median; the 5–95% gap is
kept between 0.40 and 1.60 log units so a lucky-tight hour cannot
claim false certainty. Empirical quantiles keep whatever skew the
real launch days have (evenings skew up, overnights skew down), which
a symmetric normal spread would erase.

Only comparables with a real launch hour 1 are used (hour-1 viewers
≥ 20,000, relaxed to ≥ 5,000 then ≥ 100 if too few qualify; a missing
hour-1 channel count is replaced by the training set's median). The
target game's own pre-release category history (the "Grand Theft Auto
VI" comparable, whose day is the 2026-08-27 premiere, not a launch)
is excluded.

## Validation

Leave-one-out over the kit's comparables (each launch held out in
turn, forecast from the rest), scored with the contest's own pinball
loss on log scale:

| set | this model | house persistence baseline |
| --- | --- | --- |
| all 37 usable comparables (× 4 practice cases) | **0.650** | 0.930 |

Local checks on the kit's four practice cases: outputs valid (23 rows,
five non-decreasing quantiles each), byte-identical across two runs,
~0.2 s total for all four cases.

## Fallbacks

If a forecast hour has fewer than 8 training rows the ridge fit is
replaced by the plain median log-ratio with the same empirical-spread
rule; with no usable comparables at all the program falls back to the
house persistence baseline (flat at hour 1, spread 0.15·√(h−1)), so
the output is always valid.

## License

MIT — see `LICENSE`.

---

<sub>Entered on [make.fun](https://make.fun/?ref=pwfn5nwa&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
