"""OLS regression: trend (optionally a cubic spline) + day-of-week dummies
(+ optional annual Fourier terms), optionally on the log scale."""
import numpy as np
import pandas as pd

from .common import inv, tr


def _design(index, t0, fourier_k):
    df = pd.DataFrame({"trend": (index - t0).days.to_numpy() / 365.25, "dow": index.day_of_week.to_numpy()}, index=index)
    doy = index.day_of_year.to_numpy() / 365.25
    for k in range(1, fourier_k + 1):
        df[f"s{k}"] = np.sin(2 * np.pi * k * doy)
        df[f"c{k}"] = np.cos(2 * np.pi * k * doy)
    return df


def regression(y, h, spline_df=0, fourier_k=0, log=False, **_):
    from statsmodels.formula.api import ols

    t0 = y.index[0]
    X = _design(y.index, t0, fourier_k)
    X["y"] = tr(y, log).to_numpy()
    # patsy splines cannot extrapolate: extend the boundary knots over the forecast horizon
    ub = float(X["trend"].iloc[-1]) + (h + 1) / 365.25
    terms = [f"bs(trend, df={spline_df}, degree=3, lower_bound=0.0, upper_bound={ub})" if spline_df else "trend", "C(dow)"]
    terms += [f"s{k} + c{k}" for k in range(1, fourier_k + 1)]
    res = ols("y ~ " + " + ".join(terms), data=X).fit()
    fidx = pd.date_range(y.index[-1] + pd.Timedelta(days=1), periods=h, freq="D")
    sf = res.get_prediction(_design(fidx, t0, fourier_k)).summary_frame(alpha=0.05)
    return (
        inv(sf["mean"].to_numpy(), log),
        inv(sf["obs_ci_lower"].to_numpy(), log),
        inv(sf["obs_ci_upper"].to_numpy(), log),
    )
