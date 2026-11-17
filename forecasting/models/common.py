"""Shared constants, data loading and metrics used by every model."""
import os

# One BLAS thread per worker process: parallelism comes from the process pool.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

M = 7  # weekly seasonality for daily data
Z95 = 1.96


def load_series(path="data/daily_views.csv"):
    """Daily total views with the 11 missing days filled by linear interpolation."""
    s = pd.read_csv(path, parse_dates=["day"], index_col="day")["view_count"].astype(float)
    full = pd.date_range(s.index.min(), s.index.max(), freq="D")
    s = s.reindex(full).interpolate(method="linear")
    s.index.freq = "D"
    return s


def metrics(actual, pred):
    e = np.asarray(actual, float) - np.asarray(pred, float)
    return {
        "ME": float(e.mean()),
        "MAE": float(np.abs(e).mean()),
        "RMSE": float(np.sqrt((e**2).mean())),
        "MAPE": float((np.abs(e) / np.asarray(actual, float)).mean() * 100),
    }


def band(mean, resid):
    """Approximate 95% band from the sd of in-sample residuals (for models
    without an analytic interval)."""
    sd = float(np.nanstd(np.asarray(resid, float)))
    return mean - Z95 * sd, mean + Z95 * sd


def tr(y, log):
    return np.log(y) if log else y


def inv(a, log):
    return np.exp(a) if log else a
