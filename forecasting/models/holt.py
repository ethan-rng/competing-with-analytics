"""Double exponential smoothing (Holt): level + trend."""
import numpy as np

from .common import band


def holt(y, h, alpha=None, beta=None, damped=False, **_):
    from statsmodels.tsa.holtwinters import Holt

    mod = Holt(y, damped_trend=damped, initialization_method="estimated")
    if alpha is None:
        res = mod.fit()
    else:
        kw = dict(smoothing_level=alpha, smoothing_trend=beta, optimized=False)
        if damped:
            kw["damping_trend"] = 0.98
        res = mod.fit(**kw)
    mean = np.asarray(res.forecast(h))
    lo, hi = band(mean, res.resid)
    return mean, lo, hi
