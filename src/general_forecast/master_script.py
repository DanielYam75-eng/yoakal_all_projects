import os
from concurrent.futures import ThreadPoolExecutor
import argparse
import pandas as pd
import warnings

warnings.filterwarnings("ignore")
import threading
from .preprocess_data import main as preprocess_main
from .run_notebook import main as run_notebook_main
from .hashbarot_model import main as hashbarot_main
from .clean import main as clean_main
from .evaluate import main as evaluate_main
from families_forecast import forecast_families
from disable_forecast import forecast_disabled


def main():
    # general data
    parser = argparse.ArgumentParser(description="The Main Program")
    parser.add_argument(
        "--hashbarot_data", type=str, required=True, help="Path to the hashbarot data"
    )
    parser.add_argument(
        "--main_data", type=str, required=True, help="Path to the main data"
    )
    parser.add_argument("--past_year", type=int, required=True, help="Year to forecast")
    parser.add_argument("--curr_year", type=int, required=True, help="Current Year")
    parser.add_argument("--curr_month", type=int, required=True, help="Current Month")
    parser.add_argument(
        "--months_back", type=int, required=False, default=-1, help="Month to train on"
    )
    parser.add_argument(
        "--experiment_mode",
        type=bool,
        required=False,
        default=False,
        help="Experiment mode",
    )
    parser.add_argument(
        "--coin_type", type=int, required=True, default=1, help="Coin Type"
    )

    # data for families
    parser.add_argument("--war_year", type=int, required=True)
    parser.add_argument("--hesh_data_widows", type=str, required=True)
    parser.add_argument("--hesh_data_orphans", type=str, required=True)
    parser.add_argument("--hesh_data_parents", type=str, required=True)
    parser.add_argument("--families_data", type=str, required=True)
    # data for disabled
    parser.add_argument("--hesh_data_disabled", type=str, required=True)
    parser.add_argument("--disabled_data", type=str, required=True)
    parser.add_argument("--CPI_health_changes", type=str, required=True)
    # data for families and  disabled
    parser.add_argument("--CPI_changes", type=str, required=True)

    past_year = parser.parse_args().past_year
    months_back = parser.parse_args().months_back
    curr_year = parser.parse_args().curr_year
    curr_month = parser.parse_args().curr_month
    exp_mode = parser.parse_args().experiment_mode
    coin_type = parser.parse_args().coin_type

    CPI_changes = parser.parse_args().CPI_changes
    hesh_data_orphans = parser.parse_args().hesh_data_orphans
    hesh_data_widows = parser.parse_args().hesh_data_widows
    hesh_data_parents = parser.parse_args().hesh_data_parents
    families_data = parser.parse_args().families_data
    war_year = parser.parse_args().war_year
    fund_codes = {
        "widows": 1400,
        "orphans": 1405,
        "parents": 1406,
    }
    hesh_data_disabled = parser.parse_args().hesh_data_disabled
    disabled_data = parser.parse_args().disabled_data
    CPI_health_changes = parser.parse_args().CPI_health_changes

    IND = "kvotzat otzar"
    COL = "kvuzat sahar"
    VAL = "anual"
    TABLES = (
        [f"_{curr_year}"]
        if exp_mode
        else [
            f"_{past_year}",
            f"_{curr_year}",
            f"actual_data_{past_year}_bad_otzar_only",
            f"actual_data_{curr_year}_bad_otzar_only",
        ]
    )
    TO_EVAL = []

    print("Comencing program...")
    print("Close all relevent tables !!")
    preprocess_thread = threading.Thread(
        target=preprocess_main,
        args=(parser.parse_args().main_data, curr_year, coin_type),
    )
    preprocess_thread.start()
    preprocess_thread.join()
    print("Finished preprocessing data")
    if not exp_mode:
        print("Working on hashbarot...")
        hashbarot_thread = threading.Thread(
            target=hashbarot_main,
            args=(
                parser.parse_args().hashbarot_data,
                past_year,
                curr_year,
                curr_month,
                months_back,
                coin_type,
            ),
        )
        hashbarot_thread.start()
        hashbarot_thread.join()
        print("Finished hashbarot")
    print("Working on forcasting the rest...")

    def run_table(table):
        name = table[len("result-") : -len("data-preprocessed-by-posting-date.csv") - 1]
        run_notebook_thread = threading.Thread(
            target=run_notebook_main,
            args=(
                table,
                name,
                past_year,
                curr_year,
                curr_month,
                months_back,
                coin_type,
            ),
        )
        run_notebook_thread.start()
        run_notebook_thread.join()

    tables = [
        table
        for table in os.listdir()
        if table.endswith("data-preprocessed-by-posting-date.csv")
    ]

    with ThreadPoolExecutor(max_workers=len(tables)) as executor:
        executor.map(run_table, tables)

    for table_type in TABLES:

        files = [
            f
            for f in os.listdir()
            if (f.startswith("forcast") and f.endswith(table_type + ".csv"))
            or f.startswith(table_type)
        ]

        forcasts = pd.concat([pd.read_csv(f) for f in files])

        months = forcasts.columns.difference([IND, COL])
        if not exp_mode and table_type == TABLES[1]:
            forcasts.to_csv(
                r"Data\ALL_" + table_type + "_monthly" + ".csv", index=False
            )
        forcasts[VAL] = forcasts[months[-12:]].sum(axis=1)

        forcasts = forcasts.pivot_table(
            index=IND, columns=COL, values=VAL, aggfunc="sum"
        )
        orphans_predictions = forecast_families(
            h=1,
            curr_year=int(curr_year),
            hesh_data=hesh_data_orphans,
            families_data=families_data,
            CPI_changes=CPI_changes,
            fund_code=fund_codes["orphans"],
            war_year=war_year,
        )
        widows_predictions = forecast_families(
            h=1,
            curr_year=int(curr_year),
            hesh_data=hesh_data_widows,
            families_data=families_data,
            CPI_changes=CPI_changes,
            fund_code=fund_codes["widows"],
            war_year=war_year,
        )
        parents_predictions = forecast_families(
            h=1,
            curr_year=int(curr_year),
            hesh_data=hesh_data_parents,
            families_data=families_data,
            CPI_changes=CPI_changes,
            fund_code=fund_codes["parents"],
            war_year=war_year,
        )

        orphans_predictions.columns = ["forcast_orphans_2025"]
        widows_predictions.columns = ["forcast_widows_2025"]
        parents_predictions.columns = ["forcast_parents_2025"]

        families_predictions = pd.concat(
            [orphans_predictions, widows_predictions, parents_predictions], axis=1
        )
        forcasts = pd.concat(
            [
                forcasts,
                families_predictions,
            ],
            axis=1,
        )

        disabled_predictions = forecast_disabled(
            h=1,
            curr_year=int(curr_year),
            hesh_data=hesh_data_disabled,
            disabled_data=disabled_data,
            CPI_changes=CPI_changes,
            CPI_health_changes=CPI_health_changes,
            war_year=war_year,
        )
        disabled_predictions.columns = ["forcast_disabled_2025"]
        forcasts = pd.concat([forcasts, disabled_predictions], axis=1)

        forcasts["sum"] = forcasts.select_dtypes(include="number").fillna(0).sum(axis=1)

        if table_type == TO_EVAL and not exp_mode:
            forcasts = forcasts["sum"]

        if exp_mode:
            forcasts.to_csv(
                r"Data\ALL_"
                + table_type
                + f"_{curr_year}_"
                + f"_{curr_month}_"
                + f"_{months_back}"
                + ".csv"
            )
        else:
            forcasts.to_csv(r"Data\ALL_" + table_type + ".csv")

    if not exp_mode:
        pd.concat(
            [
                pd.read_csv(f, index_col=IND)
                for f in os.listdir()
                if f.startswith("full_actual")
            ],
            axis=1,
        ).fillna(0).sort_index().to_csv(rf"Data\ALL_actual_data_{past_year}.csv")

    print("Cleaning...")
    clean_thread = threading.Thread(target=clean_main)
    clean_thread.start()
    clean_thread.join()


if __name__ == "__main__":
    main()
    print("Done")
