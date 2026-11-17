"""One module per technique; `forecast()` dispatches by family name."""
import numpy as np

from .auto_arima import auto_arima
from .common import M, load_series, metrics  # noqa: F401
from .holt import holt
from .holt_winters import holt_winters
from .moving_average import moving_average
from .naive import naive
from .regression import regression
from .sarima import sarima
from .seasonal_naive import seasonal_naive
from .ses import ses

FAMILIES = {
    "naive": naive,
    "seasonal_naive": seasonal_naive,
    "moving_average": moving_average,
    "ses": ses,
    "holt": holt,
    "holt_winters": holt_winters,
    "regression": regression,
    "sarima": sarima,
    "auto_arima": auto_arima,
}


def forecast(family, y, h, params):
    """Apply the training-window hyperparameter, run the model, return (mean, lo, hi, info)."""
    params = dict(params)
    window = params.pop("window", None)
    if window:
        y = y.iloc[-window:]
    out = FAMILIES[family](y, h, **params)
    info = None
    if family == "auto_arima":
        out, info = out
    mean, lo, hi = out
    return np.asarray(mean), np.asarray(lo), np.asarray(hi), info
