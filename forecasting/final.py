"""Final evaluation: pick each family's winner by CV (never by test), refit on the
full training set, score on the held-out test set, and save everything per model.

Output layout (results/):
    summary.csv                       one row per model: CV score, test score, params
    <family>/cv_results.csv           every config tried for that family (all rounds)
    <family>/best.json                winning hyperparameters + CV and test metrics
    <family>/test_forecast.csv        date, actual, forecast, lo95, hi95 on the test set
    <family>/future_forecast.csv      28 days beyond the last observation (refit on all data)
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(__file__))
from models import forecast, load_series, metrics  # noqa: E402
from run import H, OUT  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def _fit_eval(args):
    family, params = args
    s = load_series("data/daily_views.csv")
    train, test = s.iloc[:-H], s.iloc[-H:]
    mean, lo, hi, info = forecast(family, train, H, params)
    m = metrics(test.to_numpy(), mean)
    m["coverage95"] = float(((test.to_numpy() >= lo) & (test.to_numpy() <= hi)).mean())
    tf = pd.DataFrame({"date": test.index, "actual": test.to_numpy(), "forecast": mean, "lo95": lo, "hi95": hi})
    # refit on ALL data -> genuine out-of-sample forecast of the next H days
    fm, flo, fhi, _ = forecast(family, s, H, params)
    fidx = pd.date_range(s.index[-1] + pd.Timedelta(days=1), periods=H, freq="D")
    ff = pd.DataFrame({"date": fidx, "forecast": fm, "lo95": flo, "hi95": fhi})
    return family, m, tf, ff, (str(info) if info else None)


def main():
    cv = pd.concat([pd.read_csv(f"{OUT}/{f}") for f in sorted(os.listdir(OUT)) if f.startswith("cv_round")], ignore_index=True)
    ok = cv[cv.error.isna()]
    best = ok.sort_values("MAE").groupby("family").head(1)
    tasks = [(r.family, json.loads(r.params)) for r in best.itertuples()]
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
        results = list(ex.map(_fit_eval, tasks))
    rows = []
    for (family, m, tf, ff, info), r in zip(results, best.itertuples()):
        d = f"{OUT}/{family}"
        os.makedirs(d, exist_ok=True)
        cv[cv.family == family].sort_values("MAE").to_csv(f"{d}/cv_results.csv", index=False)
        tf.to_csv(f"{d}/test_forecast.csv", index=False)
        ff.to_csv(f"{d}/future_forecast.csv", index=False)
        rec = {
            "family": family,
            "params": json.loads(r.params),
            "auto_arima_selected": info,
            "n_configs_tried": int((cv.family == family).sum()),
            "cv": {k: float(getattr(r, k)) for k in ("ME", "MAE", "RMSE", "MAPE")},
            "test": m,
        }
        json.dump(rec, open(f"{d}/best.json", "w"), indent=2)
        rows.append({"family": family, "n_configs": rec["n_configs_tried"], "params": r.params,
                     **{f"cv_{k}": v for k, v in rec["cv"].items()}, **{f"test_{k}": v for k, v in m.items()}})
    summ = pd.DataFrame(rows).sort_values("cv_MAE")
    summ.to_csv(f"{OUT}/summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(summ.drop(columns="params").to_string(index=False, float_format=lambda x: f"{x:,.1f}"))


if __name__ == "__main__":
    main()
