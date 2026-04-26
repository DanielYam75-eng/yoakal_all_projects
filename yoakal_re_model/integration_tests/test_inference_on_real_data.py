import dagshub
import re_forecast.globals as glb
import pandas as pd
import read_file as rf
import pytest
import mlflow
import numpy as np


@pytest.fixture
def orders():
    return rf.read("august-orders-Re-model")


@pytest.fixture
def dates():
    return rf.read("august--order-dates")


@pytest.fixture
def order_edits():
    return rf.read("aug-po-edits", dtype={"order_date": str})


@pytest.fixture
def invoices():
    return rf.read("aug-invoices-RE-model_updated_version")


@pytest.fixture
def curr_year():
    return 2025


@pytest.fixture
def curr_month():
    return 7


def test_infer(
    orders, order_edits, invoices, dates, curr_year, curr_month, monkeypatch
):

    monkeypatch.setattr(dagshub, "get_repo_bucket_client", lambda *args, **kwargs: None)
    monkeypatch.setattr(pd.DataFrame, "to_csv", lambda *args, **kwargs: None)
    monkeypatch.setattr(mlflow, "log_artifact", lambda *args, **kwargs: None)
    monkeypatch.setattr(glb, "MODEL", "test_model.pkl")
    monkeypatch.chdir("integration_tests")
    from re_forecast.infer import infer
    from re_forecast.preprocess import preprocess, prepare_index, combine_dates

    orders = prepare_index(orders)
    invoices = prepare_index(invoices)
    order_edits = prepare_index(order_edits)
    dates = prepare_index(dates)

    orders = combine_dates(orders, dates)

    orders, invoices, past_sums, order_edits = preprocess(
        orders, invoices, order_edits, curr_year, curr_month
    )

    data = orders.merge(invoices, how="left", left_index=True, right_index=True)

    data["age"] = (
        (curr_year - data["order_year"]).mul(12).add(curr_month - data["order_month"])
    )

    assert np.all(np.isfinite(data["age"]))

    infer(
        orders=orders,
        invoices=invoices,
        past_sums=past_sums,
        curr_year=curr_year,
        curr_month=curr_month,
        output_path="",
    )
