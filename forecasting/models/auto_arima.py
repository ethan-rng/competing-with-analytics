"""pmdarima stepwise order search, refit with statsmodels ('let the software search')."""
import numpy as np

from .common import M, inv, tr


def auto_arima(y, h, log=False, **_):
    import pmdarima as pm
    from statsmodels.tsa.arima.model import ARIMA

    yy = tr(y, log)
    am = pm.auto_arima(yy, seasonal=True, m=M, stepwise=True, suppress_warnings=True, error_action="ignore",
                       max_p=3, max_q=3, max_P=2, max_Q=2, max_d=1, max_D=1)
    res = ARIMA(yy, order=am.order, seasonal_order=am.seasonal_order).fit()
    fc = res.get_forecast(steps=h)
    ci = np.asarray(fc.conf_int(alpha=0.05))
    return (inv(np.asarray(fc.predicted_mean), log), inv(ci[:, 0], log), inv(ci[:, 1], log)), (am.order, am.seasonal_order)
