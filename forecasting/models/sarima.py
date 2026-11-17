"""ARIMA(p,d,q)(P,D,Q)_7 with explicit orders."""
import numpy as np

from .common import M, inv, tr


def sarima(y, h, order=(1, 0, 1), seasonal=(0, 1, 1), log=False, **_):
    from statsmodels.tsa.arima.model import ARIMA

    res = ARIMA(tr(y, log), order=tuple(order), seasonal_order=tuple(seasonal) + (M,)).fit()
    fc = res.get_forecast(steps=h)
    ci = np.asarray(fc.conf_int(alpha=0.05))
    return inv(np.asarray(fc.predicted_mean), log), inv(ci[:, 0], log), inv(ci[:, 1], log)
