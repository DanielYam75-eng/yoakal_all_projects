import pandas as pd
import numpy as np
import os
from . import globals as glb
import time


INVOL = ["RE", "ZF", "ZY"]
TIMEIND = "relative_month"


def prepare_index(df: pd.DataFrame) -> pd.DataFrame:
    df["fund_year"] = df["fund_year"].astype(str).str[:4]
    df[glb.KEY] = df[glb.KEY].astype("str")
    df = df.set_index(glb.KEY)
    return df


def combine_dates(orders: pd.DataFrame, dates: pd.DataFrame) -> pd.DataFrame:
    dates = dates.loc[~dates.index.duplicated(keep="first")]
    dates = dates["order_date"]
    orders = orders.merge(dates, left_index=True, right_index=True, how="inner")
    orders["order_date"] = pd.to_datetime(orders["order_date"], format="%d.%m.%Y")
    orders["order_year"] = orders["order_date"].dt.year
    orders["order_month"] = orders["order_date"].dt.month
    return orders


def preprocess(
    orders: pd.DataFrame,
    invoices: pd.DataFrame,
    order_edits: pd.DataFrame,
    curr_year: int,
    curr_month: int,
    debug,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float]]:
    time1 = time.time()
    orders["po_net_value"] = (
        orders["po_net_value"].astype(str).str.replace(",", "").astype(float)
    )
    order_edits["order_year"] = order_edits["order_date"].str[:4].astype(int)
    order_edits["order_month"] = order_edits["order_date"].str[4:6].astype(int)

    mask_for_existing_invoices_this_year = (invoices["invoice_year"] == curr_year) & (
        invoices["invoice_month"] <= curr_month
    )

    time2 = time.time()
    past_sums = (
        invoices[mask_for_existing_invoices_this_year]
        .groupby(glb.KEY)[INVOL]
        .sum()
        .sum(axis=1)
    )

    time3 = time.time()
    orders = orders[orders["po_net_value"] > 0]
    invoices = invoices.join(
        orders[["order_year", "order_month"]],
        how="outer",
        on=glb.KEY,
        lsuffix="_invoice",
        rsuffix="_order",
    )

    invoices.dropna(subset=["order_year", "order_month"], inplace=True)
    invoices["invoice_year"] = invoices["invoice_year"].fillna(invoices["order_year"])
    invoices["invoice_month"] = invoices["invoice_month"].fillna(
        invoices["order_month"]
    )
    invoices.fillna(0, inplace=True)

    invoices[TIMEIND] = (
        invoices["invoice_year"]
        .sub(invoices["order_year"])
        .mul(12)
        .add(invoices["invoice_month"].sub(invoices["order_month"]))
        .astype(int)
    )
    invoices[TIMEIND] = invoices[TIMEIND].clip(lower=0)
    orders["N"] = (
        orders.index.get_level_values("fund_year").astype("int") - orders["order_year"]
    )
    invoices = invoices.fillna(0)
    invoices["invoice_year"] = invoices["invoice_year"] - invoices[TIMEIND]
    invoices = invoices.set_index(TIMEIND, append=True)

    invoices = invoices[INVOL].sum(axis=1).rename("volume")
    invoices = invoices.groupby(level=invoices.index.names).sum()

    invoices = invoices.unstack(TIMEIND).fillna(0)
    invoices = invoices.reindex(range(invoices.columns.max() + 1), axis=1, fill_value=0)

    orders = orders.join(
        pd.cut(
            np.log(
                orders.loc[orders["po_net_value"].astype(float) > 0, "po_net_value"]
            ),
            bins=10,
        ).rename("po_net_value_category"),
        how="left",
        on=glb.KEY,
    )
    orders["quarter"] = pd.to_datetime(orders["order_date"]).dt.quarter

    if debug:
        orders.to_csv(os.path.join("debug-output", "orders-post-preprocessing.csv"))
    time4 = time.time()
    times = {
        "preprocess:data-cleaning": time2 - time1,
        "preprocess:calculateing-past-sums": time3 - time2,
        "preprocess:data-tidying": time4 - time3,
    }
    return orders, invoices, past_sums, order_edits, times
