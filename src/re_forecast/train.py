import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import warnings
import mlflow
import os
import time

warnings.filterwarnings("ignore")
from . import globals as glb
from .utils import get_cumulative_portion, get_target


def smooth_labels(invoices, smoothing_window):
    return invoices.T.groupby(pd.Series(invoices.columns).floordiv(smoothing_window)).transform("mean").T


def get_train_data(
    orders: pd.DataFrame,
    invoices: pd.DataFrame,
    edits: pd.DataFrame,
    curr_year: int,
    curr_month: int,
    sample_frac: int,
    smoothing_window: int,
    categorial_features,
    floating_features,
    integer_features,
    seed,
) -> tuple[pd.DataFrame, dict[str, float]]:
    time1 = time.time()
    sample_frac = np.sqrt(
        sample_frac
    )  # The sampling is two-staged, so we need to take a square root of the sampling fractions for equivalent sample sizes.

    # First, sample from the orders, only orders that exist in the current time as per the train parameters.
    sampled_orders = orders[
        (orders["order_year"] < curr_year - 1)
        | (orders["order_year"] == curr_year - 1)
        & (orders["order_month"] <= curr_month)
    ].sample(frac=sample_frac, random_state=seed)
    time2 = time.time()
    training_datasets = []
    n = 12
    for years_old in range(1, 11):
        n = 12 * years_old
        sampled_orders = sampled_orders[
            (sampled_orders["order_year"] < curr_year - years_old)
            | (sampled_orders["order_year"] == curr_year - years_old)
            & (sampled_orders["order_month"] <= curr_month)
        ]
        # This creates different instances of the same PO with different ages.
        one_year = pd.concat(
            [sampled_orders] * n,
            keys=range(n),
            names=["age"],
        ).reset_index(level="age")
        training_datasets.append(one_year)

    time3 = time.time()
    training_data = pd.concat(training_datasets)
    training_data["age"] = training_data["age"].astype(int)
    training_data: pd.DataFrame = training_data.sample(
        frac=sample_frac, random_state=seed
    )
    time4 = time.time()

    # Here we caclulate the cumluative portion that is equivalent to the balance at the stated age
    training_data["abs order date"] = (
        training_data["order_year"] * 12 + training_data["order_month"]
    )
    edits["abs edit date"] = edits["order_year"] * 12 + edits["order_month"]
    training_data = training_data.merge(
        edits[["abs edit date", "volume"]],
        how="left",
        left_index=True,
        right_index=True,
    )
    training_data = training_data[
        training_data["abs edit date"].between(
            training_data["abs order date"],
            training_data["abs order date"] + training_data["age"],
        )
    ]
    training_data["po_net_value"] = training_data.groupby(glb.KEY + ["age"])[
        "volume"
    ].transform("sum")
    training_data = training_data.drop_duplicates()
    # We only train on POs with currently positive amount
    training_data = training_data[training_data["po_net_value"] > 0]
    time5 = time.time()

    invoices = smooth_labels(invoices, smoothing_window)
    time6 = time.time()
    data = training_data.merge(invoices, how="left", left_index=True, right_index=True)
    time7 = time.time()

    # Here we compute the current cumulative portion (which is equivalent to 1 - balance)

    # data is the training_data + invoice columns, and is used as intermediate objects in these computations
    training_data["cumulative_portion"] = get_cumulative_portion(data)
    training_data["target"] = np.where(
        training_data["po_net_value"] == 0, 0, get_target(data)
    )

    time8 = time.time()
    # Do not train on POs with very small balance
    training_data = training_data[training_data["cumulative_portion"] < 0.98]

    # Here we define the features
    training_data[categorial_features] = training_data[categorial_features].astype(
        "category"
    )
    training_data[integer_features] = training_data[integer_features].astype("int32")
    training_data[floating_features] = training_data[floating_features].astype(
        "float32"
    )
    training_data["target"] = training_data["target"].astype("float32")
    training_data = training_data[
        categorial_features + integer_features + floating_features + ["target"]
    ]
    time9 = time.time()
    times = {
        "get_train_data::sample-training-data": time2 - time1,
        "get_train_data::create-raw-training-data": time3 - time2,
        "get_train_data::second-sample": time4 - time3,
        "get_train_data::feature-engineering": time5 - time4,
        "get_train_data::smoothing-invoice-labels": time6 - time5,
        "get_train_data::joining-labels-with-data": time7 - time6,
        "get_train_data::create-cumulative-portion-and-target-via-apply": time8 - time7,
        "get_train_data::set-training-data-dtypes": time9 - time8,
    }
    return training_data, times


def train_model(
    data: pd.DataFrame,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    debug,
    seed,
) -> tuple[xgb.XGBRFRegressor, dict[str, float]]:

    time1 = time.time()
    train_data, test_data = train_test_split(data, test_size=0.1, random_state=seed)
    X_train = train_data.drop(columns=["target"])
    y_train = train_data["target"]
    X_test = test_data.drop(columns=["target"])
    y_test = test_data["target"]

    model = xgb.XGBRFRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=seed,
        enable_categorical=True,
    )
    if debug:
        X_train.to_csv(os.path.join("debug-output", "X_train.csv"))
        mlflow.log_artifact(
            os.path.join("debug-output", "X_train.csv"), artifact_path="debug-output"
        )
        y_train.to_csv(os.path.join("debug-output", "y_train.csv"))
        mlflow.log_artifact(
            os.path.join("debug-output", "y_train.csv"), artifact_path="debug-output"
        )
    time2 = time.time()
    model.fit(X_train, y_train)
    time3 = time.time()
    y_pred = model.predict(X_test)
    if debug:
        y_pred = pd.Series(y_pred, index=X_test.index, name="model_prediction")
        predictions_on_X_test = pd.concat([X_test, y_test, y_pred], axis=1)
        predictions_on_X_test.to_csv(os.path.join("debug-output", "predictions-on-test-set.csv"))
        mlflow.log_artifact(
            os.path.join("debug-output", "predictions-on-test-set.csv"), artifact_path="debug-output"
        )
    rmse = root_mean_squared_error(y_test, y_pred)
    mlflow.log_metric("rmse", rmse)
    mae = mean_absolute_error(y_test, y_pred)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r-squared", model.score(X_test, y_test))
    if debug:
        print(f"rmse: {rmse}")
        print(f"mae: {mae}")
        print(f"r-squared: {model.score(X_test, y_test)}")
    time4 = time.time()
    times = {
        "train_model::setup-data": time2 - time1,
        "train_model::training": time3 - time2,
        "train_model::evaluation": time4 - time3,
    }
    return model, times


def train(
    orders: pd.DataFrame,
    invoices: pd.DataFrame,
    order_edits: pd.DataFrame,
    curr_year: int,
    curr_month: int,
    sample_frac: float,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    smoothing_window: int,
    categorial_features,
    floating_features,
    integer_features,
    debug,
    seed,
):

    training_data, times_get_train_data = get_train_data(
        orders,
        invoices,
        order_edits,
        curr_year,
        curr_month,
        sample_frac,
        smoothing_window,
        categorial_features,
        floating_features,
        integer_features,
        seed,
    )
    time1 = time.time()
    # Remove rows where the label is anomalous
    training_data = training_data[
        (training_data["target"] >= 0) & (training_data["target"] <= 1.05)
    ]
    print("The length of the training data is " + str(len(training_data)))
    if debug:
        training_data.to_csv(os.path.join("debug-output", "training_data.csv"))
    time2 = time.time()
    model, times_train_model = train_model(
        training_data, n_estimators, max_depth, learning_rate, debug, seed
    )
    times = times_get_train_data
    times.update({"train::remove-anomalies": time2 - time1})
    times.update(times_train_model)
    return model, times
