# Competing with Analytics: daily Wikipedia page-view forecasting

Daily total views (sum of the top-100 articles per day, 2016-01-01 to 2026-11-14) forecast with the course techniques.

- `data/pageviews.csv` raw data; `data/daily_views.csv` daily totals (11 missing days interpolated at load time)
- `forecasting/models/` one module per technique: naive, seasonal naive, moving average, SES, Holt (double), Holt-Winters (triple), OLS regression (trend, day-of-week, Fourier, cubic spline), SARIMA, auto_arima
- `forecasting/run.py` parallel rolling-origin CV grid search (`uv run python forecasting/run.py 1` then `2`)
- `forecasting/final.py` picks each family's winner by CV, scores on the held-out last 28 days, writes `results/<model>/`
- `report/REPORT.md` final report with figures in `report/figures/`; `forecasting/report_figures.py` regenerates them
- `results/summary.csv` model comparison; `results/<model>/{cv_results.csv,best.json,test_forecast.csv,future_forecast.csv}`

Setup: `uv sync`. Winners are chosen on CV only; the 28-day test set is never used for tuning.
