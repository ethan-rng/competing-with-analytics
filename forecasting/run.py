"""Parallel hyperparameter search with rolling-origin cross-validation.

Usage:  uv run python forecasting/run.py 1      # round 1: broad grid
        uv run python forecasting/run.py 2      # round 2: refine around round-1 winners

The last H days are a held-out TEST set and are never touched here: all tuning
uses K rolling origins inside the training period.
"""
import itertools
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache

sys.path.insert(0, os.path.dirname(__file__))
from models import forecast, load_series, metrics  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

H = 28          # forecast horizon = 4 full weekly cycles
K = 6           # rolling-origin folds inside the training period
WORKERS = os.cpu_count() or 4
OUT = "results"


@lru_cache(maxsize=1)
def _series():
    return load_series("data/daily_views.csv")


def folds():
    n = len(_series())
    train_end = n - H
    return [train_end - H * (k + 1) for k in range(K)]  # origins (index of first eval day)


def cv_task(task):
    """Evaluate one (family, params) config over K folds. Runs in a worker process."""
    family, params = task
    s = _series()
    t0 = time.time()
    rows, err = [], None
    for o in folds():
        try:
            mean, _, _, _ = forecast(family, s.iloc[:o], H, params)
            if not np.all(np.isfinite(mean)):
                raise ValueError("non-finite forecast")
            rows.append(metrics(s.iloc[o : o + H].to_numpy(), mean))
        except Exception as ex:  # keep the search going
            err = f"{type(ex).__name__}: {ex}"[:120]
            break
    out = {"family": family, "params": json.dumps(params, sort_keys=True), "secs": time.time() - t0, "error": err}
    if rows and err is None:
        for m in ("ME", "MAE", "RMSE", "MAPE"):
            out[m] = float(np.mean([r[m] for r in rows]))
        out["MAE_sd"] = float(np.std([r["MAE"] for r in rows]))
    return out


# --------------------------------------------------------------------------- grids
def grid_round1():
    g = []
    g.append(("naive", {}))
    g.append(("seasonal_naive", {}))
    for N in (7, 14, 28, 56):
        g.append(("moving_average", {"N": N}))
    wins = (730, 1825, None)  # 2y, 5y, all history
    for w in wins:
        g.append(("ses", {"window": w}))
        for a in (0.1, 0.3, 0.5, 0.8):
            g.append(("ses", {"window": w, "alpha": a}))
        for damped in (False, True):
            g.append(("holt", {"window": w, "damped": damped}))
        for a, b in itertools.product((0.2, 0.5, 0.8), (0.05, 0.2)):
            g.append(("holt", {"window": w, "alpha": a, "beta": b}))
        for trend, damped, seas in itertools.product((None, "add"), (False, True), ("add", "mul")):
            if trend is None and damped:
                continue
            g.append(("holt_winters", {"window": w, "trend": trend, "damped": damped, "seasonal": seas}))
        for seas, a, b, gm in itertools.product(("add", "mul"), (0.1, 0.3, 0.5), (0.01, 0.1), (0.1, 0.3)):
            g.append(("holt_winters", {"window": w, "trend": "add", "seasonal": seas, "alpha": a, "beta": b, "gamma": gm}))
    for w, sdf, fk, lg in itertools.product((365, 730, 1825, None), (0, 4, 8), (0, 2, 4, 8), (False, True)):
        g.append(("regression", {"window": w, "spline_df": sdf, "fourier_k": fk, "log": lg}))
    for (p, d, q), (P, D, Q) in itertools.product(
        itertools.product((0, 1, 2), (0, 1), (0, 1, 2)), itertools.product((0, 1), (0, 1), (0, 1))
    ):
        if D == 0 and (P or Q):
            continue  # seasonal ARMA terms without seasonal differencing: handled in round 2 if needed
        g.append(("sarima", {"window": 1095, "order": [p, d, q], "seasonal": [P, D, Q], "log": False}))
    for lg in (False, True):
        g.append(("auto_arima", {"window": 1095, "log": lg}))
    return g


def grid_round2(prev):
    """Refine: neighbourhoods of the top-3 CV configs of the model families that matter."""
    g, seen = [], {(r.family, r.params) for r in prev.itertuples()}

    def add(f, p):
        key = (f, json.dumps(p, sort_keys=True))
        if key not in seen:
            seen.add(key)
            g.append((f, p))

    ok = prev[prev.error.isna()]
    for fam, sub in ok.groupby("family"):
        top = sub.nsmallest(3, "MAE")
        for r in top.itertuples():
            p = json.loads(r.params)
            if fam == "sarima":
                (pp, d, q), (P, D, Q) = p["order"], p["seasonal"]
                for dp, dq, dP, dQ in itertools.product((-1, 0, 1), repeat=4):
                    np_, nq, nP, nQ = pp + dp, q + dq, P + dP, Q + dQ
                    if min(np_, nq, nP, nQ) < 0 or np_ > 4 or nq > 4 or nP > 2 or nQ > 2:
                        continue
                    for lg, w in itertools.product((p["log"], not p["log"]), (p["window"], 1825)):
                        add(fam, {**p, "order": [np_, d, nq], "seasonal": [nP, D, nQ], "log": lg, "window": w})
            elif fam == "regression":
                for sdf, fk, w in itertools.product((p["spline_df"], p["spline_df"] + 4, 12, 16), (p["fourier_k"], p["fourier_k"] + 2, 12, 16), (p["window"], None)):
                    add(fam, {**p, "spline_df": sdf, "fourier_k": fk, "window": w})
            elif fam == "holt_winters":
                for a, b, gm in itertools.product((0.05, 0.1, 0.2, 0.3), (0.001, 0.01, 0.05), (0.05, 0.1, 0.2, 0.4)):
                    add(fam, {**p, "trend": "add", "alpha": a, "beta": b, "gamma": gm})
                for w in (365, 1095):
                    add(fam, {**{k: v for k, v in p.items() if k not in ("alpha", "beta", "gamma")}, "window": w})
            elif fam == "moving_average":
                for N in (3, 5, 10, 21, 35, 42):
                    add(fam, {"N": N})
            elif fam in ("ses", "holt"):
                for a in (0.05, 0.2, 0.4, 0.6):
                    for b in ((0.01, 0.1) if fam == "holt" else (None,)):
                        add(fam, {**p, "alpha": a, **({"beta": b} if b else {})})
    return g


def run(tasks, tag):
    os.makedirs(OUT, exist_ok=True)
    print(f"[{tag}] {len(tasks)} configs x {K} folds on {WORKERS} workers", flush=True)
    rows, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(cv_task, t) for t in tasks]
        for i, f in enumerate(as_completed(futs), 1):
            rows.append(f.result())
            if i % 25 == 0 or i == len(futs):
                print(f"  {i}/{len(futs)} done, {time.time()-t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df["round"] = tag
    df.to_csv(f"{OUT}/cv_round{tag}.csv", index=False)
    print(f"[{tag}] finished in {time.time()-t0:.0f}s; errors: {int(df.error.notna().sum())}", flush=True)
    return df


def summarize(df):
    best = df[df.error.isna()].sort_values("MAE").groupby("family").head(1).sort_values("MAE")
    print(best[["family", "params", "MAE", "RMSE", "MAPE", "ME"]].to_string(index=False, float_format=lambda x: f"{x:,.1f}"))


if __name__ == "__main__":
    rnd = sys.argv[1]
    if rnd == "1":
        df = run(grid_round1(), "1")
    else:
        prev = pd.read_csv(f"{OUT}/cv_round1.csv")
        df = run(grid_round2(prev), "2")
    summarize(df)
