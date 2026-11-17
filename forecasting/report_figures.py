"""Build every figure and number used in report/REPORT.md.

Run after run.py and final.py:  uv run python forecasting/report_figures.py
Writes report/figures/*.png and report/numbers.json.
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(__file__))
from models import forecast, load_series, metrics  # noqa: E402
from run import H, folds  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

FIG = "report/figures"
# dataviz reference palette (light mode): categorical slots 1-3 validate all-pairs
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898880", "#e6e5e1", "#fcfcfb"
SPIKE_DAY = pd.Timestamp("2026-11-04")

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titlesize": 12,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False,
})
M_FMT = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v/1e6:.0f}M")
M1_FMT = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v/1e6:.1f}M")
NICE = {"holt_winters": "Holt-Winters", "holt": "Holt", "ses": "SES", "sarima": "SARIMA", "auto_arima": "auto_arima",
        "naive": "Naive", "seasonal_naive": "Seasonal naive", "moving_average": "Moving average", "regression": "Regression"}


def nice(f):
    return NICE.get(f, f)



def _fold_maes(args):
    family, params = args
    s = load_series("data/daily_views.csv")
    out = []
    for o in folds():
        mean, _, _, _ = forecast(family, s.iloc[:o], H, params)
        out.append(float(np.abs(s.iloc[o : o + H].to_numpy() - mean).mean()))
    return family, out


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"{FIG}/{name}.png", dpi=160)
    plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    s = load_series("data/daily_views.csv")
    raw = pd.read_csv("data/pageviews.csv")
    summ = pd.read_csv("results/summary.csv")
    train, test = s.iloc[:-H], s.iloc[-H:]
    N = {}

    # ---------------- data quality (analyst skill: verify before analysing)
    daily = pd.read_csv("data/daily_views.csv", parse_dates=["day"])
    full = pd.date_range(daily.day.min(), daily.day.max())
    missing = [d.date().isoformat() for d in full.difference(pd.DatetimeIndex(daily.day))]
    N["quality"] = {
        "raw_rows": int(len(raw)), "raw_nulls": int(raw.isna().sum().sum()),
        "dup_article_day": int(raw.duplicated(["date", "article"]).sum()),
        "rows_per_day_unique": sorted(raw.groupby("date").size().unique().tolist()),
        "days_observed": int(len(daily)), "days_expected": int(len(full)), "missing_days": missing,
        "nonpositive_days": int((s <= 0).sum()),
    }
    med = s.rolling(29, center=True, min_periods=7).median()
    dev = (s / med - 1)
    big = dev[dev > 0.8].sort_values(ascending=False)
    N["spikes"] = [{"date": d.date().isoformat(), "views": float(s[d]), "x_median": float(1 + v)} for d, v in big.head(10).items()]
    N["n_spikes_gt80pct"] = int((dev > 0.8).sum())
    N["series"] = {"mean": float(s.mean()), "median": float(s.median()), "min": float(s.min()), "max": float(s.max()),
                   "cv": float(s.std() / s.mean())}
    dow = s.groupby(s.index.day_of_week).median()
    N["dow_median"] = {["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][i]: float(v) for i, v in dow.items()}
    yr = s.groupby(s.index.year).median()
    N["year_median"] = {int(k): float(v) for k, v in yr.items()}

    # ---------------- Fig 1: full history
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(s.index, s, color=BLUE, lw=0.6, alpha=0.55, label="Daily views")
    ax.plot(s.index, s.rolling(30, min_periods=7).median(), color=ORANGE, lw=1.8, label="30-day rolling median")
    ax.set_ylim(0, None); ax.yaxis.set_major_formatter(M_FMT)
    ax.set_ylabel("Views per day"); ax.set_title("Daily views, 2016 to Nov 2026: stable baseline, sharp news spikes")
    for d, lab in [("2020-11-08", "US election 2020"), ("2024-11-06", "US election 2024"), ("2016-11-09", "US election 2016")]:
        ax.annotate(lab, (pd.Timestamp(d), s[d]), xytext=(6, -2), textcoords="offset points", fontsize=8, color=INK2)
    ax.legend(loc="upper left", ncol=2)
    save(fig, "01_history")

    # ---------------- Fig 2: weekday profile + year medians
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    axs[0].bar(names, dow.values, color=BLUE, width=0.6)
    axs[0].set_ylim(0, None); axs[0].yaxis.set_major_formatter(M_FMT)
    axs[0].set_title("Median views by weekday"); axs[0].grid(axis="x", visible=False)
    for i, v in enumerate(dow.values):
        axs[0].text(i, v + 1.5e5, f"{v/1e6:.1f}M", ha="center", fontsize=8, color=INK2)
    axs[1].bar([str(y) for y in yr.index], yr.values, color=BLUE, width=0.6)
    axs[1].set_ylim(0, None); axs[1].yaxis.set_major_formatter(M_FMT)
    axs[1].set_title("Median views by year (no clear trend)"); axs[1].grid(axis="x", visible=False)
    axs[1].tick_params(axis="x", labelsize=8)
    save(fig, "02_seasonality")

    # ---------------- Fig 3: CV vs test MAE per model
    sm = summ.sort_values("cv_MAE")
    labels = [nice(f) for f in sm.family]
    x = np.arange(len(sm)); w = 0.38
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - w / 2, sm.cv_MAE, w, color=BLUE, label="Cross-validation MAE (6 folds, training period)")
    ax.bar(x + w / 2, sm.test_MAE, w, color=ORANGE, label="Test MAE (last 28 days)")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0, None); ax.yaxis.set_major_formatter(M1_FMT); ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(5e5))
    ax.set_ylabel("Mean absolute error (views/day)"); ax.grid(axis="x", visible=False)
    ax.set_title("Tuned models beat naive in cross-validation, not on the test window")
    nv = summ.set_index("family").loc["naive"]
    ax.axhline(nv.cv_MAE, color=BLUE, lw=0.8, ls=":"); ax.axhline(nv.test_MAE, color=ORANGE, lw=0.8, ls=":")
    ax.legend(loc="upper left", fontsize=8.5)
    save(fig, "03_model_comparison")

    # ---------------- Fig 4: test forecasts
    tf = {f: pd.read_csv(f"results/{f}/test_forecast.csv", parse_dates=["date"]) for f in ["naive", "holt_winters", "sarima", "regression"]}
    fig, ax = plt.subplots(figsize=(10, 4.2))
    hist = s.iloc[-70:-H]
    ax.plot(hist.index, hist, color=INK, lw=1.6)
    ax.plot(test.index, test, color=INK, lw=1.6, label="Actual")
    hw = tf["holt_winters"]
    ax.fill_between(hw.date, hw.lo95, hw.hi95, color=BLUE, alpha=0.12, lw=0, label="Holt-Winters 95% interval")
    ax.plot(hw.date, hw.forecast, color=BLUE, lw=1.8, label="Holt-Winters (CV winner)")
    ax.plot(tf["naive"].date, tf["naive"].forecast, color=ORANGE, lw=1.8, label="Naive")
    ax.plot(tf["sarima"].date, tf["sarima"].forecast, color=AQUA, lw=1.8, label="SARIMA")
    ax.axvline(test.index[0], color=MUTED, lw=0.8, ls="--")
    ax.annotate("Election-night spike\n19.5M (not forecastable)", (SPIKE_DAY, s[SPIKE_DAY]), xytext=(-150, -8), textcoords="offset points",
                fontsize=8.5, color=INK2, arrowprops=dict(arrowstyle="-", color=MUTED))
    ax.set_ylim(0, None); ax.yaxis.set_major_formatter(M_FMT); ax.set_ylabel("Views per day")
    ax.set_title("Held-out test window (28 days): forecasts vs actual"); ax.legend(loc="upper left", ncol=2, fontsize=8.5)
    save(fig, "04_test_forecasts")

    # ---------------- Fig 5: residual ACF + Ljung-Box for the SARIMA winner
    from statsmodels.graphics.tsaplots import plot_acf
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.tsa.arima.model import ARIMA

    best = json.load(open("results/sarima/best.json"))["params"]
    y = train.iloc[-best["window"]:]
    yy = np.log(y) if best["log"] else y
    res = ARIMA(yy, order=tuple(best["order"]), seasonal_order=tuple(best["seasonal"]) + (7,)).fit()
    skip = best["seasonal"][1] * 7 + best["order"][1]
    resid = res.resid.iloc[max(skip, 1):]
    lb = acorr_ljungbox(resid, lags=[7, 14, 28], return_df=True)
    N["sarima_ljung_box"] = {int(k): {"stat": float(r.lb_stat), "p": float(r.lb_pvalue)} for k, r in lb.iterrows()}
    fig, ax = plt.subplots(figsize=(10, 3.4))
    plot_acf(resid, lags=36, zero=False, ax=ax, color=BLUE, vlines_kwargs={"colors": BLUE}, alpha=0.05)
    ax.set_title(f"SARIMA {tuple(best['order'])}{tuple(best['seasonal'])}7 residual ACF (bars outside the band = unexplained pattern)")
    ax.set_xlabel("Lag (days)")
    save(fig, "05_sarima_residual_acf")

    # ---------------- fold-level comparison vs naive (parallel)
    tasks = [(r.family, json.loads(r.params)) for r in summ.itertuples()]
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
        fold_maes = dict(ex.map(_fold_maes, tasks))
    N["fold_mae"] = fold_maes
    N["wins_vs_naive"] = {f: int(sum(a < b for a, b in zip(v, fold_maes["naive"]))) for f, v in fold_maes.items() if f != "naive"}
    fig, ax = plt.subplots(figsize=(10, 3.8))
    k = np.arange(1, len(fold_maes["naive"]) + 1)
    for f, c in [("naive", ORANGE), ("holt_winters", BLUE), ("sarima", AQUA)]:
        ax.plot(k, fold_maes[f], color=c, lw=1.8, marker="o", ms=5, label=nice(f))
    ax.set_ylim(0, None); ax.yaxis.set_major_formatter(M1_FMT); ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(5e5))
    ax.set_xticks(k); ax.set_xticklabels([f"fold {i}" for i in k]); ax.set_ylabel("MAE")
    ax.set_ylim(0, 3.4e6); ax.set_xlabel("Rolling-origin fold (1 = most recent 28 days of training, 6 = oldest)")
    ax.set_title("Per-fold MAE: the ranking changes from fold to fold"); ax.legend(ncol=3, loc="upper left")
    save(fig, "06_fold_mae")

    # ---------------- test metrics with / without the election spike (skill: report both)
    rows = {}
    for f in summ.family:
        t = pd.read_csv(f"results/{f}/test_forecast.csv", parse_dates=["date"])
        keep = t.date != SPIKE_DAY
        rows[f] = {"with_spike": metrics(t.actual, t.forecast), "without_spike": metrics(t.actual[keep], t.forecast[keep])}
    N["test_with_without"] = rows
    order = sorted(rows, key=lambda f: rows[f]["without_spike"]["MAE"])
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(order)); w = 0.38
    ax.bar(x - w / 2, [rows[f]["with_spike"]["MAE"] for f in order], w, color=ORANGE, label="All 28 test days")
    ax.bar(x + w / 2, [rows[f]["without_spike"]["MAE"] for f in order], w, color=BLUE, label="Excluding Nov 4 election spike")
    ax.set_xticks(x); ax.set_xticklabels([nice(f) for f in order], rotation=20, ha="right")
    ax.set_ylim(0, None); ax.yaxis.set_major_formatter(M1_FMT); ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(5e5))
    ax.grid(axis="x", visible=False); ax.set_ylabel("Test MAE"); ax.legend(loc="upper left")
    ax.set_title("One spike day accounts for a large share of every model's test error")
    save(fig, "07_spike_effect")

    # ---------------- Fig 8: future forecast from the CV winner
    best_f = summ.sort_values("cv_MAE").iloc[0].family
    ff = pd.read_csv(f"results/{best_f}/future_forecast.csv", parse_dates=["date"])
    fig, ax = plt.subplots(figsize=(10, 4))
    h = s.iloc[-90:]
    ax.plot(h.index, h, color=INK, lw=1.6, label="Actual (last 90 days)")
    ax.fill_between(ff.date, ff.lo95, ff.hi95, color=BLUE, alpha=0.14, lw=0, label="95% interval")
    ax.plot(ff.date, ff.forecast, color=BLUE, lw=2, label=f"{nice(best_f)} forecast")
    ax.set_ylim(0, None); ax.yaxis.set_major_formatter(M_FMT); ax.set_ylabel("Views per day")
    ax.set_title("Next 28 days (2026-11-15 to 2026-12-12)"); ax.legend(loc="upper right", ncol=3, fontsize=8.5)
    save(fig, "08_future_forecast")
    N["future"] = {"model": best_f, "mean_forecast": float(ff.forecast.mean()), "lo_mean": float(ff.lo95.mean()), "hi_mean": float(ff.hi95.mean()),
                   "first": ff.iloc[0][["date", "forecast", "lo95", "hi95"]].astype(str).to_dict()}

    json.dump(N, open("report/numbers.json", "w"), indent=1, default=str)
    print("figures:", sorted(os.listdir(FIG)))


if __name__ == "__main__":
    main()
