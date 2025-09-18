import dagshub
import re_forecast.globals as glb
import pandas as pd
import read_file as rf
import pytest
from re_forecast.preprocess import (
    preprocess,
    combine_dates,
    prepare_index,
)


@pytest.fixture
def orders():
    return rf.read('august-orders-Re-model')


@pytest.fixture
def dates():
    return rf.read('august--order-dates')

@pytest.fixture
def order_edits():
    return rf.read('aug-po-edits', dtype={"order_date": str})

@pytest.fixture
def invoices():
    return rf.read('aug-invoices-RE-model_updated_version')

@pytest.fixture
def curr_year():
    return 2025


@pytest.fixture
def curr_month():
    return 7



@pytest.mark.filterwarnings("ignore")
def test_with_real_data(orders, invoices, order_edits, dates, curr_year, curr_month):
    assert orders.index.is_unique
    orders = prepare_index(orders)
    assert orders.index.is_unique
    invoices = prepare_index(invoices)
    order_edits = prepare_index(order_edits)
    dates = prepare_index(dates)

    orders = combine_dates(orders, dates)
    assert orders.index.is_unique

    orders, invoices, order_edits, past_sums = preprocess(
        orders, invoices, order_edits, curr_year, curr_month
    )

    # dtypes
    assert type(orders) is pd.DataFrame
    assert type(invoices) is pd.DataFrame
    assert type(order_edits) is pd.Series
    assert type(past_sums) is pd.DataFrame

    # index
    assert orders.index.names == glb.KEY
    assert invoices.index.names == glb.KEY
    assert order_edits.index.names == glb.KEY
    assert past_sums.index.names == glb.KEY

    # columns
    assert set(["N", "quarter"]).issubset(orders.columns)

    # range
    assert not orders.empty
    assert not invoices.empty
    assert not order_edits.empty
    assert not past_sums.empty

    assert not orders.isnull().any().any()
    assert not invoices.isnull().any().any()
    assert not order_edits.isnull().any().any()
    assert not past_sums.isnull().any().any()

    assert orders.index.is_unique
    assert invoices.index.is_unique

