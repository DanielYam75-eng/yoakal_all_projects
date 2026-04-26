# Revenues Project

This repository contains a time series forecasting project built to estimate revenue for 2026 using historical payment and order data. The work is documented in the notebook `notebook hahnasot.ipynb` and includes end-to-end data cleaning, exploratory analysis, decomposition, forecasting, and output generation.

## Project Overview

The goal of this project is to produce a robust revenue forecast for 2026 from past transactional data. The code ingests invoice/payment data and order metadata, applies business-specific preprocessing, performs deep exploratory data analysis, and generates forecast outputs that compare model predictions with past actual revenues.

This workflow is aligned with the forecasting work delivered at Yoakal, where multiple statistical and machine learning approaches were evaluated and the final deliverable included CSV forecast outputs for business planning.

## Data Sources

- `invoice.csv`: payment records, including fields like `budget_year`, `order_id`, `paymant_date`, `mimun`, and additional accounting dimensions.
- `order.csv`: order metadata, including sales office, order starting date, fund code, document type, and more.
- `yearly_forecast_vs_actual_1.csv` and `yearly_forecast_vs_actual_5.csv`: generated forecast outputs for the relevant coin filters, containing forecasted revenue and actual revenue history.

## Notebook and Key Files

- `notebook hahnasot.ipynb`: main analysis notebook containing all data preparation, visualization, time series decomposition, forecasting logic, and final output export.
- `README.md`: this project summary.

## Exploratory Data Analysis (EDA)

The notebook performs extensive EDA, including:

- Data cleaning and normalization of column names.
- Parsing payment dates and order starting dates into proper datetime types.
- Filtering by business-relevant fields such as `coin`, `huka`, `expenditure_type`, `doc_type`, and budget year.
- Aggregating payments by budget year, order type, and payment category.
- Visualizing budget-year breakdowns and comparing values on and off the official order register.
- Inspecting the distribution of transaction values using signed log transformation and density plots.
- Detecting spikes and anomalies in the time series to isolate irregular payments prior to forecasting.
- Using monthly seasonality tests to identify regular seasonality patterns.

## Forecast Pipeline

The forecasting pipeline in the notebook follows a structured time series modeling approach:

1. `fill_inactive_months` and active-series detection:
   - Ensure the series has uniform monthly frequency.
   - Detect a consistent active range for reliable forecasting.

2. Variance diagnostics and decomposition selection:
   - Compute trend and rolling standard deviation.
   - Estimate whether additive or multiplicative decomposition is more suitable.
   - In this implementation, additive decomposition is explicitly used for the final forecasting path.

3. STL decomposition:
   - Decompose the cleaned monthly series into trend, seasonal, and residual components using `statsmodels.tsa.seasonal.STL`.
   - Both additive and multiplicative variants are supported by the notebook.

4. Trend forecasting:
   - Primary method: `pmdarima.auto_arima` to identify an optimized ARIMA model.
   - Fallback: `statsmodels` ARIMA, Holt exponential smoothing, or polynomial trend regression.
   - Trend damping logic is applied if forecast values become negative.
   - The code includes a `get_non_negative_forecast` routine to enforce non-negative monthly forecast behavior when needed.

5. Seasonality forecasting:
   - Use the latest 12 months of the seasonal component and repeat it to create a seasonal forecast horizon.
   - Align the seasonal forecast to the forecast period using month-end indexing.

6. Residual and confidence range handling:
   - Compute residual standard deviation from the decomposition.
   - Generate upper/lower confidence bounds for forecast results.
   - Create both `forecast_monthly` and `forecast_yearly` outputs.

7. Aggregation and evaluation:
   - Produce yearly forecast sums and compare against actual yearly values.
   - Keep actual data only for fully observed years.

## Methods and Libraries Used

The project demonstrates a combination of advanced time series techniques and business-aware forecasting:

- `statsmodels` for STL decomposition, ARIMA, SARIMAX, Holt-Winters, and statistics diagnostics.
- `pmdarima` for `auto_arima` model selection.
- `scikit-learn` for regression analysis and trend diagnostics.
- `xgboost` is imported in the notebook for advanced experimentation with machine learning models.
- `scipy` and `scikit_posthocs` for distribution testing and seasonality significance.
- `seaborn` and `matplotlib` for plotting and visual diagnostics.

## Forecast Output

The generated CSV outputs include:

- `huka`, `expediture_type`, `fc3`, `doc_type` identifying the time series slice.
- `total_forecasted_value`: the final forecasted revenue for the prediction year.
- `total_forecasted_value_no_resid`: the forecast value excluding residual contribution.
- `actual_2023`, `actual_2024`, `actual_2025`: historical actual totals for comparison.

Example schema from `yearly_forecast_vs_actual_1.csv` and `yearly_forecast_vs_actual_5.csv`:

- `huka`
- `expediture_type`
- `fc3`
- `doc_type`
- `total_forecasted_value`
- `total_forecasted_value_no_resid`
- `actual_2023`
- `actual_2024`
- `actual_2025`

## Repository and Push Status

This repository is configured with the correct GitHub remote:

- `https://github.com/DanielYam75-eng/revenues_project.git`

I am creating this README directly in the `revenues_project` repository and will commit it now.
