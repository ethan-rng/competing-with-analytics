"""Simple exponential smoothing (level only)."""
import numpy as np

from .common import band


def ses(y, h, alpha=None, **_):
    from statsmodels.tsa.holtwinters import SimpleExpSmoothing

    mod = SimpleExpSmoothing(y, initialization_method="estimated")
    res = mod.fit() if alpha is None else mod.fit(smoothing_level=alpha, optimized=False)
    mean = np.asarray(res.forecast(h))
    lo, hi = band(mean, res.resid)
    return mean, lo, hi
