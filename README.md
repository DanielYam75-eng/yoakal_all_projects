# Yoakal All Projects

This repository collects five separate forecasting and data-handling projects developed during my time at Yoakal. Each project was built to solve a distinct business problem in budget forecasting, expense simulation, and data pipeline management for ministry-level financial planning.

## Projects Included

1. `yoakal_bucket`
2. `yoakal_mof_class_model`
3. `yoakal_re_model`
4. `yoakal_revenu_model`
5. `yoakal_RNN_model`

---

## 1. yoakal_bucket

### What it is
A command-line utility package for managing templates and financial data stored in a centralized cloud bucket.

### What it is used for
- Uploading files to the bucket.
- Downloading files from the bucket.
- Listing bucket contents.
- Showing available templates.
- Marking template keys as broken (`break-key`).
- Removing keys that were marked as broken.

### Key design points
- Uses a configuration-driven CLI with explicit entry points for each action.
- Provides dedicated data management operations for bucket-based workflows.
- Supports template validation through break/unbreak semantics.

### Algorithms / behavior
- No advanced ML algorithms; instead, the project emphasizes robust data tooling for cloud bucket operations and template lifecycle management.

---

## 2. yoakal_mof_class_model

### What it is
A forecasting engine designed for Ministry of Finance (MOF) classification budgets. It generates quarterly expenditure forecasts for Treasury groups and categorizes forecasts by topic and currency type.

### What it is used for
- Producing official quarterly forecasts for budget planning.
- Evaluating several alternative models and selecting the best-performing model for each category.
- Forecasting both local currency (NIS) and foreign currency expenditures.

### Algorithms used
- Internal statistical models:
  - `NaiveModel` — repeats the last observed value.
  - `SeasonalNaiveModel` — repeats the value from the same month last year.
  - `MeanModel` — uses the historical average.
  - `MonthlyModel` — averages each calendar month separately.
  - `SeasonalLinearModel` — fits a trend and seasonality, combining linear regression with seasonal ratios.
  - `AvgFactorModel` — projects forward using average growth factors.

- External statistical models from `statsmodels`:
  - `SARIMAX` — captures trend, seasonality, and autocorrelation.
  - `ExponentialSmoothing` / Holt-Winters — models trend and seasonality with exponential decay.
  - `SimpleExpSmoothing` — for stable series without trend.
  - `Holt` — linear trend smoothing.

### Selection approach
- For every Treasury Group, the system fits multiple candidate models.
- The winning model is selected based on the highest R² score.
- This empirical selection makes the forecast adaptive to the data characteristics of each topic.

---

## 3. yoakal_re_model

### What it is
A machine-learning system for forecasting expense invoices associated with procurement contracts (RE invoices). It models both the generation of future purchase orders and the invoice amount for each contract.

### What it is used for
- Predicting invoice cash flows for contracts tied to the Ministry of Defense.
- Simulating future purchase orders using probabilistic generation.
- Forecasting monthly invoicing behavior per purchase order.

### Algorithms used
- **Augmenter**
  - Uses a Naive Bayes generative modeling approach to simulate future purchase orders (POs).
  - Samples PO net values from a lognormal distribution fitted to actual data.
  - Inverts the Naive Bayes model to generate PO attributes from target net values.

- **Predictor**
  - Uses `XGBoost` regression to forecast the monthly invoice amount of each PO.
  - Models the normalized invoice share per PO age.

### Data processing
- Converts multi-index tables to PO-level rows with age and engineered features.
- Uses categorical features such as PO type, procurement org, expenditure type, quarter, and more.
- Smooths intermittent monthly invoice signals with a rolling window to improve predictability.
- Applies sampling to manage very large datasets and avoid memory crashes.

---

## 4. yoakal_revenu_model

### What it is
A time series forecasting project built to estimate revenue for a future year (2026) using historical payment and order data.

### What it is used for
- Producing revenue forecasts for business planning.
- Comparing predicted revenue to actual historical results.
- Supporting analysis and planning through detailed notebook-driven experiments.

### Algorithms used
- `statsmodels` for STL decomposition and time-series modeling.
- `pmdarima.auto_arima` for automated ARIMA selection.
- ARIMA, SARIMAX, Holt exponential smoothing, and polynomial trend regression as candidate trend models.
- Seasonal forecasting by repeating the most recent seasonal cycle.
- Confidence interval estimation using residual standard deviation.

### Workflow
- Cleans and normalizes invoice/payment/order data.
- Aggregates series by budget year, document type, and accounting dimensions.
- Detects seasonality and chooses additive decomposition.
- Forecasts trend and seasonality separately.
- Produces both monthly and yearly forecast outputs.
- Exports results as CSV files for comparison with actual revenues.

---

## 5. yoakal_RNN_model

### What it is
A deep learning project using recurrent neural networks to forecast budget-related time series.

### What it is used for
- Modeling sequential invoice and budget data with RNN/LSTM architectures.
- Forecasting monthly budget values while learning temporal patterns.
- Comparing deep learning performance against simpler forecasting alternatives.

### Algorithms used
- RNN-based sequence modeling, specifically LSTM-style recurrent units.
- Training on sequential monthly budget series.
- Hyperparameter control for epochs, batch size, learning rate, regularization, and noise injection.

### Workflow
- Prepares the data in `predocs.ipynb` for deep learning.
- Trains and evaluates the model in `rnn.ipynb`.
- Supports filtering and grouping by document type, account prefixes, and other categorical partitions.
- Emphasizes model simplicity, code quality, and practical forecasting usage.

---

## Summary
This workspace contains five distinct projects, each focused on a different part of the forecasting and financial data lifecycle:

- `yoakal_bucket`: data and template management tools.
- `yoakal_mof_class_model`: modular statistical forecasting for budget categories.
- `yoakal_re_model`: ML-driven contract expense forecasting with PO simulation.
- `yoakal_revenu_model`: revenue forecasting using time-series decomposition and ARIMA.
- `yoakal_RNN_model`: deep learning forecasting with RNNs for sequential budget data.

Together, they capture a broad set of techniques from CLI data engineering to classical time series modeling and modern machine learning.
