from .train import train
from .preprocess import preprocess, combine_dates, prepare_index
from .infer import infer
from .augmentation import augmentation_by_sum_per_month
import argparse
import pandas as pd
import read_file as rf
import dagshub
import mlflow
from importlib_metadata import version
import os
from . import utils
from typing import Self
import pickle
import signal
import sys
import time


def pprint(dict_: dict[str, float]):
    maximum_key_length = max([len(key) for key in dict_])
    # The length we allocate for each key should suffice for each key with 5 extra spaces, and be at least 15
    pretty_key_space_length = max(15, maximum_key_length + 5)
    for key, val in dict_.items():
        number_of_points = pretty_key_space_length - len(key)
        print(f"{key}" + "." * number_of_points + f"{val:.2f}s")
    print("-" * (pretty_key_space_length + 5))
    total_time = sum(dict_.values())
    number_of_points = pretty_key_space_length - len("total")
    print(f"total" + "." * number_of_points + f"{total_time:.2f}s")


class Configuration:
    def __init__(self):
        self.experiment = None
        self.key_orders = None
        self.key_invoices = None
        self.key_orders_dates = None
        self.key_order_edits = None
        self.curr_year = None
        self.curr_month = None
        self.forecast_to = None
        self.sample_frac = None
        self.n_estimators = None
        self.max_depth = None
        self.learning_rate = None
        self.smoothing_window = None
        self.mode = None
        self.augmentation_dict = None

    def set_config(self, config_path: os.PathLike) -> Self:
        base_config = self.get_base_config(config_path)
        config = self.complete_config(base_config)
        self.experiment = config["experiment"]
        self.key_orders = config["orders"]
        self.key_invoices = config["invoices"]
        self.key_orders_dates = config["orders_dates"]
        self.key_order_edits = config["order_edits"]
        self.curr_year = config["curr_year"]
        self.curr_month = config["curr_month"]
        self.forecast_to = config["forecast_to"]
        self.sample_frac = config["sample_frac"]
        self.n_estimators = config["n_estimators"]
        self.max_depth = config["max_depth"]
        self.learning_rate = config["learning_rate"]
        self.smoothing_window = config["smoothing_window"]
        self.mode = config["mode"] if "mode" in config else ""
        self.seed = config["seed"] if "seed" in config else 42
        if "augmentation_dict" not in config:
            self.augmentation_dict = {}
            print("Info: Augmentation isn't executed")
        else:
            self.augmentation_dict = config["augmentation_dict"]
        if "categorical_features" not in config:
            self.categorical_features = []
        else:
            self.categorical_features = config["categorical_features"]
        if "integer_features" not in config:
            self.integer_features = []
        else:
            self.floating_features = config["floating_features"]
        if "floating_features" not in config:
            self.floating_features = []
        else:
            self.integer_features = config["integer_features"]
        if len(self.categorical_features) == 0 and len(self.integer_features) == 0 and len(self.floating_features) == 0:
            self.categorical_features = [
                "po_type",
                "fingroup",
                "huka",
                "porcurment_organization",
                "expanditure_type",
                "quarter",
            ]
            self.floating_features = ["po_net_value", "cumulative_portion"]
            self.integer_features = ["age", "N"]
            print("Info: Features not stated in configuration. Using default setup")
        return self

    def get_base_config(self, config_path: os.PathLike) -> dict[str, str]:
        using_user_path = config_path is not None
        default_config_path = "default.conf"
        if using_user_path and not os.path.exists(config_path):
            print(f"\033[0;33mWarning\033[0m: {config_path} doesn't exist.")
            using_user_path = False
        if not using_user_path and os.path.exists(default_config_path):
            print(
                f"Info: Reading configuration from the default path {default_config_path}."
            )
            config_path = default_config_path

        config = {}
        if config_path is not None and os.path.exists(config_path):
            with open(config_path) as f:
                try:
                    config = utils.load_configuration(f)
                except utils.DecodingException as e:
                    print(f"\033[0;33mWarning\033[0m: {e.message}")
                    print(
                        f"\033[0;33mWarning\033[0m: {config_path} is not a valid configuration file. Ignoring it."
                    )
        return config

    def _get_experiment_from_user(self) -> str:
        return input("Please enter the experiment name: ")

    def _get_orders_from_user(self) -> str:
        return input("Please enter the key name for the uploaded file orders: ")

    def _get_invoices_from_user(self) -> str:
        return input("Please enter the key name for the uploaded file invoices: ")

    def _get_dates_from_user(self) -> str:
        return input("Please enter the key name for the uploaded file orders_dates: ")

    def _get_order_edits_from_user(self) -> str:
        return input("Please enter the key name for the uploaded file order_edits: ")

    def _get_curr_year_from_user(self) -> int:
        while True:
            curr_year = input("Please enter the current year (YYYY): ")
            try:
                cy = int(curr_year)
                if 2000 <= cy <= 2100:
                    return cy
                else:
                    print(
                        f"\033[0;33mWarning\033[0m: {curr_year} is not a valid year. Please enter a valid year between 2000 and 2100."
                    )
            except ValueError:
                print(
                    f"\033[0;33mWarning\033[0m: {curr_year} is not a valid integer. Please enter a valid year between 2000 and 2100."
                )

    def _get_curr_month_from_user(self) -> int:
        while True:
            curr_month = input("Please enter the current month (1-12): ")
            try:
                cm = int(curr_month)
                if 1 <= cm <= 12:
                    return cm
                else:
                    print(
                        f"\033[0;33mWarning\033[0m: {curr_month} is not a valid month. Please enter a valid month between 1 and 12."
                    )
            except ValueError:
                print(
                    f"\033[0;33mWarning\033[0m: {curr_month} is not a valid integer. Please enter a valid month between 1 and 12."
                )

    def _get_forecast_to_from_user(self) -> int:
        while True:
            forecast_to = input(
                "Please enter the forecast horizon (0 for N, 1 for N+1, etc): "
            )
            try:
                ft = int(forecast_to)
                if 1 <= ft <= 12:
                    return ft
                else:
                    print(
                        f"\033[0;33mWarning\033[0m: {forecast_to} is not a horizon. Please enter a valid non-negative horizon."
                    )
            except ValueError:
                print(
                    f"\033[0;33mWarning\033[0m: {forecast_to} is not a horizon. Please enter a valid non-negative horizon."
                )

    def _get_sample_frac_from_user(self) -> float:
        while True:
            sf = input(
                "Please enter the fraction of data to be used for training (between 0 and 1): "
            )
            try:
                sf = float(sf)
                if 0.0 < sf <= 1.0:
                    return sf
                else:
                    print(
                        f"\033[0;33mWarning\033[0m: {sf} is not in the valid range (0.0, 1.0]. Please enter a valid fraction."
                    )
            except ValueError:
                print(
                    f"\033[0;33mWarning\033[0m: {sf} is not a valid float. Please enter a valid fraction."
                )

    def _get_n_estimators_from_user(self) -> int:
        while True:
            ne = input("Please enter the number of estimators: ")
            try:
                ne = int(ne)
                if ne > 0:
                    return ne
                    print(
                        f"\033[0;33mWarning\033[0m: {ne} is not a valid positive integer. Please enter a valid positive integer."
                    )
            except ValueError:
                print(
                    f"\033[0;33mWarning\033[0m: {ne} is not a valid integer. Please enter a valid positive integer."
                )

    def _get_max_depth_from_user(self) -> int:
        while True:
            md = input("Please enter the maximum depth: ")
            try:
                md = int(md)
                if md > 0:
                    return md
                else:
                    print(
                        f"\033[0;33mWarning\033[0m: {md} is not a valid positive integer. Please enter a valid positive integer."
                    )
            except ValueError:
                print(
                    f"\033[0;33mWarning\033[0m: {md} is not a valid integer. Please enter a valid positive integer."
                )

    def _get_learning_rate_from_user(self) -> float:
        while True:
            lr = input("Please enter the learning rate: ")
            try:
                lr = float(lr)
                if 0.0 < lr <= 1.0:
                    return lr
                    print(
                        f"\033[0;33mWarning\033[0m: {lr} is not in the valid range (0.0, 1.0]. Please enter a valid learning rate."
                    )
            except ValueError:
                print(
                    f"\033[0;33mWarning\033[0m: {lr} is not a valid float. Please enter a valid learning rate."
                )

    def _get_smoothing_window_from_user(self) -> int:
        while True:
            sw = input("Please enter the smoothing_window: ")
            try:
                sw = int(sw)
                if sw > 0:
                    return sw
                    print(
                        f"\033[0;33mWarning\033[0m: {sw} is not a postive integer. Please enter a valid learning rate."
                    )
            except ValueError:
                print(
                    f"\033[0;33mWarning\033[0m: {sw} is not a postive integer. Please enter a valid learning rate."
                )

    def complete_config(self, config: dict[str, str]) -> dict[str, str]:
        if "experiment" not in config:
            config["experiment"] = self._get_experiment_from_user()
        if "orders" not in config:
            config["orders"] = self._get_orders_from_user()
        if "invoices" not in config:
            config["invoices"] = self._get_invoices_from_user()
        if "orders_dates" not in config:
            config["orders_dates"] = self._get_dates_from_user()
        if "order_edits" not in config:
            config["order_edits"] = self._get_order_edits_from_user()
        if "curr_year" not in config:
            config["curr_year"] = self._get_curr_year_from_user()
        if "curr_month" not in config:
            config["curr_month"] = self._get_curr_month_from_user()
        if "forecast_to" not in config:
            config["forecast_to"] = self._get_forecast_to_from_user()
        if "sample_frac" not in config:
            config["sample_frac"] = self._get_sample_frac_from_user()
        if "n_estimators" not in config:
            config["n_estimators"] = self._get_n_estimators_from_user()
        if "max_depth" not in config:
            config["max_depth"] = self._get_max_depth_from_user()
        if "learning_rate" not in config:
            config["learning_rate"] = self._get_learning_rate_from_user()
        if "smoothing_window" not in config:
            config["smoothing_window"] = self._get_smoothing_window_from_user()
        return config


def log_configuration(configuration: Configuration) -> None:
    package_version = version("re-forecast")
    mlflow.set_tags({"mlflow.source.name": "re-forecast"})
    mlflow.log_param("curr_year", configuration.curr_year)
    mlflow.log_param("curr_month", configuration.curr_month)
    mlflow.log_param("forecast_to", configuration.forecast_to)
    mlflow.log_param("sample_frac", configuration.sample_frac)
    mlflow.log_param("n_estimators", configuration.n_estimators)
    mlflow.log_param("max_depth", configuration.max_depth)
    mlflow.log_param("learning_rate", configuration.learning_rate)
    mlflow.log_param("smoothing_window", configuration.smoothing_window)
    mlflow.log_param("mode", configuration.mode)
    mlflow.log_param("categorical_features", configuration.categorical_features)
    mlflow.log_param("floating_features", configuration.floating_features)
    mlflow.log_param("integer_features", configuration.integer_features)
    mlflow.log_param("seed", configuration.seed)
    commit = package_version.split("+")
    if len(commit) >= 2:
        mlflow.set_tags({"mlflow.source.git.commit": commit[1][1:].split(".")[0]})
    else:
        mlflow.set_tags({"mlflow.source.git.commit": commit})


def set_cli_args():
    parser = argparse.ArgumentParser(description="Main script")
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        required=True,
        help="Path to the data file",
    )
    parser.add_argument("-c", "--config", type=str, help="Path to config file")
    parser.add_argument(
        "-m", "--model", type=str, help="Path to model file", default="model.pkl"
    )
    parser.add_argument("--fine", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--time", action="store_true")
    parser.add_argument("--monthly", action="store_true")

    return vars(parser.parse_args())


def preprocess_and_simulate_data(
    orders: pd.DataFrame,
    orders_dates: pd.DataFrame,
    order_edits: pd.DataFrame,
    invoices: pd.DataFrame,
    curr_year: int,
    curr_month: int,
    augmentation_dict: dict[str, dict[str, float]],
    skip_augmentation: bool,
    debug,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float]]:
    time1 = time.time()
    orders = prepare_index(orders)
    orders = orders.groupby(level=orders.index.names).agg(
        {
            "po_type": "first",
            "huka": "first",
            "porcurment_organization": "first",
            "expanditure_type": "first",
            "fingroup": "first",
            "po_net_value": "sum",
        }
    )
    orders_dates = prepare_index(orders_dates)
    order_edits = prepare_index(order_edits)
    invoices = prepare_index(invoices)
    orders = combine_dates(
        orders, orders_dates
    )  # adding to order the column order_date, order_year, order_month
    time2 = time.time()
    if not skip_augmentation:
        simulated_orders, simulated_dates, times_augmentation = (
            augmentation_by_sum_per_month(  # simulated orders are generated here (same format as orders), simulated dates contain for each index ( doc_id, item, fund_year) the corresponding order_date
                orders.loc[orders["order_year"] >= curr_year - 2], augmentation_dict
            )
        )
        simulated_orders = combine_dates(
            simulated_orders, simulated_dates
        )  # adding to order the column order_date, order_year, order_month
    else:
        times_augmentation = {}
    time3 = time.time()
    if not skip_augmentation:
        orders_for_inference = pd.concat(
            [orders, simulated_orders], ignore_index=False
        )  # combine simulated orders and actual orders
        dates_for_inference = pd.concat(
            [orders_dates, simulated_dates], ignore_index=False
        )
    else:
        orders_for_inference = orders
        dates_for_inference = orders_dates
    time4 = time.time()
    orders, invoices, past_sums, order_edits, times_preprocess = (
        preprocess(  # orders contains all the additional columns needed for inference and training
            orders_for_inference,
            invoices,
            order_edits,
            curr_year,
            curr_month,
            debug,
        )
    )
    times = {
        "preprocess_and_simulate_data::pre-preprocessing": time2 - time1,
    }
    times.update(times_augmentation)
    times.update(
        {
            "preprocess_and_simulate-data::combining-simulated-and-real-data": time4
            - time3,
        }
    )
    times.update(times_preprocess)
    return orders, invoices, past_sums, order_edits, times


def handle_sigint(signum, frame):
    print("\n\033[91mExecution interrupted by user\033[0m")
    sys.exit(0)


def main():
    cli_args = set_cli_args()
    # Change the key `time` to `time_` to not have name collision with the module `time`
    cli_args["time_"] = cli_args["time"]
    del cli_args["time"]
    train_and_forecast(**cli_args)

def train_and_forecast(output_path, config, model, fine, debug, time_, monthly):
    if time_:
        t0 = time.time()
    # Install signal hangler for SIGINT (user-interruption - CTRL+C)
    signal.signal(signal.SIGINT, handle_sigint)


    configuration = Configuration().set_config(config)
    if time_:
        t1 = time.time()
        times = {"read-configuration": t1 - t0}

    # Reading the datasets
    # Including caching mechanisms to dataset
    if configuration.key_orders.endswith(".csv"):
        orders = pd.read_csv(
            configuration.key_orders,
            dtype={
                "doc_id": str,
                "fund_year": str,
                "item": str,
                "po_type": str,
                "huka": str,
                "porcurment_organization": str,
                "expanditure_type": str,
                "fingroup": str,
                "po_net_value": float,
            },
        )
        source_key_orders = "disk"
    elif os.path.isdir("__cache__") and os.path.isfile(
        os.path.join("__cache__", f"{configuration.key_orders}.csv")
    ):
        orders = pd.read_csv(
            os.path.join("__cache__", f"{configuration.key_orders}.csv"),
            dtype={
                "doc_id": str,
                "fund_year": str,
                "item": str,
                "po_type": str,
                "huka": str,
                "porcurment_organization": str,
                "expanditure_type": str,
                "fingroup": str,
                "po_net_value": float,
            },
        )
        source_key_orders = "bucket"
    else:
        orders = rf.read(
            configuration.key_orders,
            dtype={
                "doc_id": str,
                "fund_year": str,
                "item": str,
                "po_type": str,
                "huka": str,
                "porcurment_organization": str,
                "expanditure_type": str,
                "fingroup": str,
                "po_net_value": float,
            },
        )
        source_key_orders = "bucket"
        if not os.path.isdir('__cache__'):
            os.mkdir('__cache__')
        orders.to_csv(os.path.join('__cache__', f'{configuration.key_orders}.csv'), index=False)
    if configuration.key_orders_dates.endswith(".csv"):
        orders_dates = pd.read_csv(
            configuration.key_orders_dates,
            dtype={"doc_id": str, "item": str, "fund_year": str, "order_date": str},
        )
        source_orders_dates = "disk"
    elif os.path.isdir("__cache__") and os.path.isfile(
        os.path.join("__cache__", f"{configuration.key_orders_dates}.csv")
    ):
        orders_dates = pd.read_csv(
            os.path.join("__cache__", f"{configuration.key_orders_dates}.csv"),
            dtype={"doc_id": str, "item": str, "fund_year": str, "order_date": str},
        )
        source_orders_dates = "bucket"
    else:
        orders_dates = rf.read(
            configuration.key_orders_dates,
            dtype={"doc_id": str, "item": str, "fund_year": str, "order_date": str},
        )
        source_orders_dates = "bucket"
        if not os.path.isdir('__cache__'):
            os.mkdir('__cache__')
        orders_dates.to_csv(os.path.join('__cache__', f'{configuration.key_orders_dates}.csv'), index=False)
    if configuration.key_order_edits.endswith(".csv"):
        order_edits = pd.read_csv(
            configuration.key_order_edits,
            dtype={"order_date": str, "doc_id": str, "item": str, "volume": float},
        )
        source_orders_edits = "disk"
    elif os.path.isdir("__cache__") and os.path.isfile(
        os.path.join("__cache__", f"{configuration.key_order_edits}.csv")
    ):
        order_edits = pd.read_csv(
            os.path.join("__cache__", f"{configuration.key_order_edits}.csv"),
            dtype={"order_date": str, "doc_id": str, "item": str, "volume": float},
        )
        source_orders_edits = "bucket"
    else:
        order_edits = rf.read(
            configuration.key_order_edits,
            dtype={"order_date": str, "doc_id": str, "item": str, "volume": float},
        )
        source_orders_edits = "bucket"
        if not os.path.isdir('__cache__'):
            os.mkdir('__cache__')
        order_edits.to_csv(os.path.join('__cache__', f'{configuration.key_order_edits}.csv'), index=False)
    if configuration.key_invoices.endswith(".csv"):
        invoices = pd.read_csv(
            configuration.key_invoices,
            dtype={
                "doc_id": str,
                "fund_year": str,
                "item": str,
                "invoice_year": int,
                "invoice_month": int,
                "mof_class": str,
                "RE": float,
                "ZY": float,
                "ZF": float,
            },
        )
        source_key_invoices = "disk"
    elif os.path.isdir("__cache__") and os.path.isfile(
        os.path.join("__cache__", f"{configuration.key_invoices}.csv")
    ):
        invoices = pd.read_csv(
            os.path.join("__cache__", f"{configuration.key_invoices}.csv"),
            dtype={
                "doc_id": str,
                "fund_year": str,
                "item": str,
                "invoice_year": int,
                "invoice_month": int,
                "mof_class": str,
                "RE": float,
                "ZY": float,
                "ZF": float,
            },
        )
        source_key_invoices = "bucket"
    else:
        invoices = rf.read(
            configuration.key_invoices,
            dtype={
                "doc_id": str,
                "fund_year": str,
                "item": str,
                "invoice_year": int,
                "invoice_month": int,
                "mof_class": str,
                "RE": float,
                "ZY": float,
                "ZF": float,
            },
        )
        source_key_invoices = "bucket"
        if not os.path.isdir("__cache__"):
            os.mkdir("__cache__")
        invoices.to_csv(os.path.join("__cache__", f"{configuration.key_invoices}.csv"), index=False)

    if time_:
        t2 = time.time()
        times["read_datasets"] = t2 - t1
    dagshub.init(
        repo_owner="yoacal.data.science",
        repo_name="exp-repo",
        mlflow=True,
    )
    if debug:
        mlflow.set_experiment("re_forecast_debug")
        if not os.path.exists("debug-output"):
            os.makedirs("debug-output")
    else:
        mlflow.set_experiment(configuration.experiment)
    # Logging the run parameters to mlflow
    with mlflow.start_run():
        log_configuration(configuration)

        mlflow.log_input(
            mlflow.data.from_pandas(
                orders.head(1),
                source=f"s3://{source_key_orders}/" + configuration.key_orders,
                name="data for orders",
            )
        )
        mlflow.log_input(
            mlflow.data.from_pandas(
                orders_dates.head(1),
                source=f"s3://{source_orders_dates}/" + configuration.key_orders_dates,
                name="data for orders dates",
            )
        )
        mlflow.log_input(
            mlflow.data.from_pandas(
                order_edits.head(1),
                source=f"s3://{source_orders_edits}/" + configuration.key_order_edits,
                name="data for order edits",
            )
        )
        mlflow.log_input(
            mlflow.data.from_pandas(
                invoices.head(1),
                source=f"s3://{source_key_invoices}/" + configuration.key_invoices,
                name="data for invoices",
            )
        )

        # Preprocessing
        if time_:
            t3 = time.time()
            times["log-in-mlflow"] = t3 - t2
        orders, invoices, past_sums, order_edits, preprocess_and_simulate_data_times = (
            preprocess_and_simulate_data(
                orders,
                orders_dates,
                order_edits,
                invoices,
                configuration.curr_year,
                configuration.curr_month,
                configuration.augmentation_dict,
                # skip augmentation in train mode
                configuration.mode == "train",
                debug,
            )
        )

        # mode `infer` means simply that the program won't train a model but rather load an existing one
        if time_:
            t4 = time.time()
            times.update(preprocess_and_simulate_data_times)
        if configuration.mode == "infer":
            trained_model = pickle.load(open(model, "rb"))
            times_train = {}
        else:
            trained_model, times_train = train(
                orders,
                invoices,
                order_edits,
                configuration.curr_year,
                configuration.curr_month,
                configuration.sample_frac,
                configuration.n_estimators,
                configuration.max_depth,
                configuration.learning_rate,
                configuration.smoothing_window,
                configuration.categorical_features,
                configuration.floating_features,
                configuration.integer_features,
                debug,
                configuration.seed,
            )
        # mode `train` means simply that the program won't infer but rather dump the trained model into the disk
        if configuration.mode == "train":
            pickle.dump(trained_model, open(output_path, "wb"))
            mlflow.log_artifact(
                output_path, artifact_path="model"
            )
        else:
            sum_forecasted_orders, total_forecast = infer(
                orders[
                    (orders["order_year"] >= configuration.curr_year - 7)
                    & (
                        orders.index.get_level_values("fund_year").astype(int)
                        <= configuration.curr_year + configuration.forecast_to
                    )
                ],
                invoices,
                past_sums,
                configuration.curr_year,
                configuration.curr_month,
                configuration.forecast_to,
                trained_model,
                configuration.categorical_features,
                configuration.floating_features,
                configuration.integer_features,
                debug,
                monthly,
            )
            with_index = True
            if not fine:
                sum_forecasted_orders = (
                    sum_forecasted_orders.groupby("fingroup").sum().reset_index()
                )
                with_index = False
            sum_forecasted_orders.to_csv(output_path, index=with_index)
            mlflow.log_artifact(
                os.path.join(os.getcwd(), output_path),
                artifact_path="forecast_outputs",
            )
            mlflow.log_metric("total forecast", total_forecast)
        if time_:
            t5 = time.time()
            times.update(times_train)
            pprint(times)


if __name__ == "__main__":
    main()
