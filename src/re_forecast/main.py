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
import json
from typing import Self
import pickle


class Configuration:
    def __init__(self):
        self.key_orders = None
        self.key_invoices = None
        self.key_orders_dates = None
        self.key_order_edits = None
        self.curr_year = None
        self.curr_month = None
        self.sample_frac = None
        self.n_estimators = None
        self.max_depth = None
        self.learning_rate = None
        self.mode = None
        self.augmentation_dict = None

    def set_config(self, config_path: os.PathLike) -> Self:
        base_config = self.get_base_config(config_path)
        config = self.complete_config(base_config)
        self.key_orders = config["orders"]
        self.key_invoices = config["invoices"]
        self.key_orders_dates = config["orders_dates"]
        self.key_order_edits = config["order_edits"]
        self.curr_year = config["curr_year"]
        self.curr_month = config["curr_month"]
        self.sample_frac = config["sample_frac"]
        self.n_estimators = config["n_estimators"]
        self.max_depth = config["max_depth"]
        self.learning_rate = config["learning_rate"]
        self.mode = config["mode"] if "mode" in config else ""
        if "augmentation_dict" not in config:
            self.augmentation_dict = {}
            print("Info: Augmentation isn't executed")
        else:
            self.augmentation_dict = config["augmentation_dict"]
        return self

    def get_base_config(self, config_path: os.PathLike) -> dict[str, str]:
        using_user_path = config_path is not None
        default_config_path = "config.json"
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
                    config = json.load(f)
                except json.JSONDecodeError:
                    print(
                        f"\033[0;33mWarning\033[0m: {config_path} is not a valid JSON file. Ignoring it."
                    )
        return config

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

    def complete_config(self, config: dict[str, str]) -> dict[str, str]:
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
        if "sample_frac" not in config:
            config["sample_frac"] = self._get_sample_frac_from_user()
        if "n_estimators" not in config:
            config["n_estimators"] = self._get_n_estimators_from_user()
        if "max_depth" not in config:
            config["max_depth"] = self._get_max_depth_from_user()
        if "learning_rate" not in config:
            config["learning_rate"] = self._get_learning_rate_from_user()
        return config


def log_configuration(configuration: Configuration) -> None:
    package_version = version("mof-class-forecaster")
    mlflow.set_tags({"mlflow.source.name": "mof-class-forecaster"})
    mlflow.log_param("curr_year", configuration.curr_year)
    mlflow.log_param("curr_month", configuration.curr_month)
    mlflow.log_param("sample_frac", configuration.sample_frac)
    mlflow.log_param("n_estimators", configuration.n_estimators)
    mlflow.log_param("max_depth", configuration.max_depth)
    mlflow.log_param("learning_rate", configuration.learning_rate)
    mlflow.log_param("mode", configuration.mode)
    mlflow.set_tags(
        {"mlflow.source.git.commit": package_version.split("+")[1][1:].split(".")[0]}
    )


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

    return parser.parse_args()


def preprocess_and_simulate_data(
    orders: pd.DataFrame,
    orders_dates: pd.DataFrame,
    order_edits: pd.DataFrame,
    invoices: pd.DataFrame,
    curr_year: int,
    curr_month: int,
    augmentation_dict: dict[str, dict[str, float]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
    simulated_orders, simulated_dates = (
        augmentation_by_sum_per_month(  # simulated orders are generated here (same format as orders), simulated dates contain for each index ( doc_id, item, fund_year) the corresponding order_date
            orders, augmentation_dict
        )
    )
    simulated_orders = combine_dates(
        simulated_orders, simulated_dates
    )  # adding to order the column order_date, order_year, order_month
    orders_for_inference = pd.concat(
        [orders, simulated_orders], ignore_index=False
    )  # combine simulated orders and actual orders
    dates_for_inference = pd.concat([orders_dates, simulated_dates], ignore_index=False)
    orders, invoices, past_sums, order_edits = (
        preprocess(  # orders contains all the additional columns needed for inference and training
            orders_for_inference,
            invoices,
            order_edits,
            curr_year,
            curr_month,
        )
    )
    return orders, invoices, past_sums, order_edits


def main():
    cli_args = set_cli_args()

    configuration = Configuration().set_config(cli_args.config)

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
    if configuration.key_orders_dates.endswith(".csv"):
        orders_dates = pd.read_csv(
            configuration.key_orders_dates,
            dtype={"doc_id": str, "item": str, "fund_year": str, "order_date": str},
        )
        source_orders_dates = "disk"
    else:
        orders_dates = rf.read(
            configuration.key_orders_dates,
            dtype={"doc_id": str, "item": str, "fund_year": str, "order_date": str},
        )
        source_orders_dates = "bucket"
    if configuration.key_order_edits.endswith(".csv"):
        order_edits = pd.read_csv(
            configuration.key_order_edits,
            dtype={"order_date": str, "doc_id": str, "item": str, "volume": float},
        )
        source_orders_edits = "disk"
    else:
        order_edits = rf.read(
            configuration.key_order_edits,
            dtype={"order_date": str, "doc_id": str, "item": str, "volume": float},
        )
        source_orders_edits = "bucket"
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

    dagshub.init(
        repo_owner="yoacal.data.science",
        repo_name="exp-repo",
        mlflow=True,
    )
    mlflow.set_experiment("re_forecast")
    with mlflow.start_run():
        log_configuration(configuration)

        mlflow.log_input(
            mlflow.data.from_pandas(
                orders,
                source=f"s3://{source_key_orders}/" + configuration.key_orders,
                name="data for orders",
            )
        )
        mlflow.log_input(
            mlflow.data.from_pandas(
                orders_dates,
                source=f"s3://{source_orders_dates}/" + configuration.key_orders_dates,
                name="data for orders dates",
            )
        )
        mlflow.log_input(
            mlflow.data.from_pandas(
                order_edits,
                source=f"s3://{source_orders_edits}/" + configuration.key_order_edits,
                name="data for order edits",
            )
        )
        mlflow.log_input(
            mlflow.data.from_pandas(
                invoices,
                source=f"s3://{source_key_invoices}/" + configuration.key_invoices,
                name="data for invoices",
            )
        )
        orders, invoices, past_sums, order_edits = preprocess_and_simulate_data(
            orders,
            orders_dates,
            order_edits,
            invoices,
            configuration.curr_year,
            configuration.curr_month,
            configuration.augmentation_dict,
        )
        if configuration.mode == "infer":
            trained_model = pickle.load(open(cli_args.model, "rb"))
        else:
            trained_model = train(
                orders,
                invoices,
                order_edits,
                configuration.curr_year,
                configuration.curr_month,
                configuration.sample_frac,
                configuration.n_estimators,
                configuration.max_depth,
                configuration.learning_rate,
            )
        if configuration.mode == "train":
            pickle.dump(trained_model, open(cli_args.output_path, "wb"))
        else:
            infer(
                orders[orders["order_year"] >= configuration.curr_year - 7],
                invoices,
                past_sums,
                configuration.curr_year,
                configuration.curr_month,
                cli_args.output_path,
                trained_model,
                cli_args.fine,
            )


if __name__ == "__main__":
    main()
