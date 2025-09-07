import os
import pandas as pd
import xgboost as xgb
import pickle
import warnings
import mlflow

warnings.filterwarnings("ignore")
from . import globals as glb


def forecast(
    model: xgb.XGBRFRegressor, data: pd.DataFrame, horizon: int
) -> pd.DataFrame:

    predictions = []
    for h in range(1, horizon + 1):
        prediction = pd.Series(
            model.predict(data[data["age"] >= 0]), index=data[data["age"] >= 0].index
        )
        prediction.name = h

        # update step
        data["age"] += 1
        data["cumulative_portion"] += prediction
        predictions.append(prediction)

    total_predictions = pd.concat(predictions, axis=1).fillna(0)
    return total_predictions


def infer(
    orders: pd.DataFrame,
    invoices: pd.DataFrame,
    past_sums: pd.DataFrame,
    curr_year: int,
    curr_month: int,
    output_path: str,
):

    with open(glb.MODEL, "rb") as f:
        model = pickle.load(f)
    data = augment_orders(orders, curr_year, curr_month)
    data = data.merge(invoices, how="left", left_index=True, right_index=True)

    data["age"] = (
        (curr_year - data["order_year"]).mul(12).add(curr_month - data["order_month"])
    )

    def get_cumulative_portion(row: pd.Series):

        max_month = max(col for col in row.index if not isinstance(col, str))
        data_lim = min(row["age"], max_month)
        so_far = row.loc[list(range(data_lim))].sum()
        so_far_prc = so_far / row["po_net_value"]

        return so_far_prc

    data["cumulative_portion"] = data.apply(get_cumulative_portion, axis=1)
    categorial_features = [
        "po_type",
        "fingroup",
        "huka",
        "porcurment_organization",
        "expanditure_type",
        "quarter",
    ]
    floating_features = ["po_net_value", "cumulative_portion"]
    integer_features = ["age", "N"]

    data[categorial_features] = data[categorial_features].astype("category")
    data[integer_features] = data[integer_features].astype("int32")
    data[floating_features] = data[floating_features].astype("float32")
    data = data[categorial_features + integer_features + floating_features]

    forecasted_orders = forecast(model, data, 12 - curr_month)
    forecasted_orders.to_csv("raw_output.csv")
    mlflow.log_artifact(
        "raw_output.csv", artifact_path="forecast_output"
    )

    sum_forecasted_orders : pd.Series = (
        forecasted_orders.sum(axis=1)
        .mul(data["po_net_value"], axis=0)
        .add(past_sums, fill_value=0)
    )
    sum_forecasted_orders = sum_forecasted_orders.to_frame().merge(data[["fingroup"]], left_index=True, right_index=True)
    sum_forecasted_orders = sum_forecasted_orders.groupby("fingroup").sum().reset_index()
    sum_forecasted_orders.to_csv(output_path, index=False)
    mlflow.log_artifact(
        os.path.join(os.getcwd(), output_path), artifact_path="forecast_outputs"
    )


def augment_orders(
    orders: pd.DataFrame, curr_year: int, curr_month: int
) -> pd.DataFrame:
    temp = orders[
        (orders["order_year"] == curr_year - 1) & (orders["order_month"] > curr_month)
    ]
    temp["order_date"] = pd.to_datetime(temp["order_date"])
    temp.loc[:, "order_date"] = temp["order_date"] + pd.DateOffset(years=1)
    temp.index = temp.index.set_levels(
        temp.index.levels[0].astype(str) + "N", level="doc_id"
    )
    temp.index = temp.index.set_levels(
        temp.index.levels[1].astype(int) + 1, level="fund_year"
    )
    temp.loc[:, "order_year"] = temp["order_year"] + 1
    augmented_orders = pd.concat([orders, temp])
    return augmented_orders
