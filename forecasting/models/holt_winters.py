"""Triple exponential smoothing (Holt-Winters): level + trend + weekly seasonality."""
import numpy as np

from .common import M, band


def holt_winters(y, h, trend="add", damped=False, seasonal="add", alpha=None, beta=None, gamma=None, **_):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    manual = alpha is not None
    mod = ExponentialSmoothing(
        y,
        trend=trend,
        damped_trend=bool(damped and trend),
        seasonal=seasonal,
        seasonal_periods=M,
        initialization_method="heuristic" if manual else "estimated",
    )
    if manual:
        kw = dict(smoothing_level=alpha, smoothing_seasonal=gamma, optimized=False)
        if trend:
            kw["smoothing_trend"] = beta
            if damped:
                kw["damping_trend"] = 0.98
        res = mod.fit(**kw)
    else:
        res = mod.fit()
    mean = np.asarray(res.forecast(h))
    lo, hi = band(mean, res.resid)
    return mean, lo, hi
