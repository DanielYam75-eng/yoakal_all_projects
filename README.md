📘 Documentation — general_forecast.py
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
⚙️ Overview

The general_forecast module is part of the mof_class_forecaster package and serves as the central forecasting engine for predicting the execution of invoices by Treasury Groups that do not include .RE invoices.
It is executed once every quarter to produce official forecasts for Treasury expenditure groups across multiple topics.

The forecasting process combines several statistical and heuristic models, automatically selecting the most suitable model for each group based on past performance.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
🔵 Methodology

The forecasting process follows a structured economic workflow with several key stages:
Data Preparation – All historical data is cleaned, processed, and organized into thematic categories.
Each topic contains multiple Treasury Groups, each requiring an independent forecast.

Model Application – The system applies a predefined set of models per topic.
These models are defined in code and may vary between topics.

Model Evaluation – For each Treasury Group, all models are fitted and evaluated using historical data.
Accuracy is measured using the R-squared (R²) metric, which quantifies how well each model explains past behavior.

Model Selection – The model with the highest R² score becomes the “winning model”, used to generate future forecasts.
Forecast Generation – The winning model produces a forward-looking forecast, which is then combined with the current year’s actuals to estimate the total expected annual value up to the current month.
This approach ensures adaptability, empirical validation, and stability over time.

🔹 Most models are generic, designed for wide applicability.
🔹 For specific topics with advanced methodologies, the module integrates external specialized models via APIs from other internal forecasting packages.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

🕒 Frequency of Use

Forecasts are generated quarterly, following data consolidation.
Results are used for budget planning, expenditure forecasting, and monitoring across Treasury categories.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

📈 Forecasting Models

The forecasting framework combines internally implemented models with established methods from the statsmodels library.
Each model has a consistent structure:

model.fit()       # Train on historical data
model.forecast()  # Produce forward forecast


🔵 NaiveModel

Idea: Future values remain constant — equal to the last observed value.
Use Case: Stable data with no clear trend or seasonality.


🔵 SeasonalNaiveModel

Idea: Each forecasted month repeats the value from the same month last year.
Use Case: Data with strong annual seasonality.


🔵 MeanModel

Idea: Forecast equals the historical mean of all observed data.
Use Case: Data fluctuating around a long-term average.



🔵 MonthlyModel

Idea: Each calendar month is forecasted using its historical average.
Use Case: Data with recurring monthly patterns.



🔵 SeasonalLinearModel

Idea: Combines linear trend + seasonal component.
Steps:
1. Smooth data with a 12-month rolling mean.
2. Fit linear regression to the trend.
3. Estimate seasonality using ratios and a MonthlyModel.
4. Combine both for the final forecast.

Use Case: Data with steady growth and seasonal effects.

🔵 AvgFactorModel

Idea: Uses the average historical growth rate to project forward.
Use Case: Stable series with consistent growth patterns.

----------------------------

📈 External Statistical Models
In addition to internal models, the module leverages robust statistical approaches from the statsmodels library.

🌀 SARIMAX

Captures trend, seasonality, and autocorrelation.
Can include exogenous variables for richer modeling.
Best suited for complex time-series with strong temporal dependencies.

🔷 ExponentialSmoothing

Applies exponentially decreasing weights to past observations.
Can model trend and seasonality (Holt-Winters).
Ideal for data evolving gradually over time.

🔹 SimpleExpSmoothing

Simplified exponential smoothing with no trend or seasonality.
Performs well for stable, noise-dominated series.

🔸 Holt (Linear Trend Model)

Extends exponential smoothing by adding a trend component.
Best for data with consistent upward or downward drift.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

🧠 Model Usage by Topic and Currency Type

Each topic (type_) uses a tailored combination of models depending on its statistical profile and currency type (coin_type):
1 → Israeli Shekels (NIS)
5 → Foreign currency (FX)
For every Treasury Group, all relevant models are tested; the one with the highest R² becomes the winning model.

💼 Salaries and Compensation

Topics: career_salary, drafted_salary, pensions, scholarships
Models: NaiveModel, SeasonalNaiveModel, MeanModel, AvgFactorModel
➡️ Used for stable, gradual trends with moderate seasonality.

⚡ Utilities and Energy

Topics: electricity, water, communications, fuel
Models: SeasonalLinearModel, MonthlyModel, ExponentialSmoothing, Holt
➡️ Captures recurring seasonal cycles and long-term consumption trends.

🧰 Procurement and Operations

Topics: maintenance, services, equipment, construction, projects
Models: AvgFactorModel, ExponentialSmoothing, Holt, SARIMAX
➡️ Handles irregular, project-based spending patterns.

🚚 Transportation and Logistics

Topics: transportation, vehicles, shipping, tariffs
Models: SeasonalNaiveModel, SeasonalLinearModel, ExponentialSmoothing, Holt, SARIMAX
➡️ Balances seasonal, trend, and short-term fluctuations.

🏦 Insurance and Transfers

Topics: insurance, compensations, grants, transfers
Models: NaiveModel, MeanModel, Holt, ExponentialSmoothing, SARIMAX
➡️ Stable patterns with occasional large, policy-driven events.

🌍 Foreign Currency Expenditures (coin_type = 5)

Models: AvgFactorModel, SeasonalLinearModel, ExponentialSmoothing, Holt
➡️ Designed for FX-based forecasts — stable under exchange-rate variability.

🗂️ Other / Default Topics

Models: NaiveModel, SeasonalLinearModel, ExponentialSmoothing, Holt
➡️ Generic, balanced model suite for all other topics.

✅ This structure ensures that each Treasury Group forecast is data-driven, empirically validated, and aligned with the statistical behavior of its underlying expenditure pattern.
The system remains modular, transparent, and adaptive — key qualities for reliable public-sector forecasting.















