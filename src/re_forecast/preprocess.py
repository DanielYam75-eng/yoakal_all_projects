import pandas as pd
import numpy as np
from . import globals as glb


INVOL = ['RE', 'ZF', 'ZY']
TIMEIND = 'relative_month'


def combine_dates(orders: pd.DataFrame, dates: pd.DataFrame) -> pd.DataFrame:

    orders['fund_year'] = orders['fund_year'].astype(str).str[:4]
    orders['po_net_value'] = orders['po_net_value'].astype(str).str.replace(',', '').astype(float)
    

    dates[glb.KEY] = dates[glb.KEY].astype('str')
    orders[glb.KEY] = orders[glb.KEY].astype('str')
    dates = dates[glb.KEY + ["order_date"]]

    orders = orders.merge(dates, on=glb.KEY, how='inner')

    orders['order_date'] = pd.to_datetime(orders['order_date'], dayfirst=True)
    orders['order_year'] = orders['order_date'].dt.year
    orders['order_month'] = orders['order_date'].dt.month
    
    return orders

def process(orders: pd.DataFrame, invoices: pd.DataFrame, order_edits: pd.DataFrame, curr_year: int, curr_month: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    order_edits["order_year"] = order_edits["order_date"].str[:4].astype(int)
    order_edits["order_month"] = order_edits["order_date"].str[4:6].astype(int)
    order_edits[glb.KEY] = order_edits[glb.KEY].astype('str')
    order_edits = order_edits.set_index(glb.KEY)

    mask_for_existing_invoices_this_year = (invoices['invoice_year'] == curr_year) & (invoices['invoice_month'] <= curr_month)

    past_sums = invoices[mask_for_existing_invoices_this_year].groupby(glb.KEY)[INVOL].sum().sum(axis=1)

    orders = orders.drop_duplicates(subset=glb.KEY, keep='first')
    orders = orders.set_index(glb.KEY)
    orders = orders[orders['po_net_value'] > 0]
    invoices[glb.KEY] = invoices[glb.KEY].astype('str')
    invoices = invoices.set_index(glb.KEY)
    invoices = invoices.join(orders[["order_year", "order_month"]], how='outer', on=glb.KEY, lsuffix='_invoice', rsuffix='_order')

    invoices.dropna(subset=["order_year", "order_month"], inplace=True)
    invoices['invoice_year'] = invoices['invoice_year'].fillna(invoices['order_year'])
    invoices['invoice_month'] = invoices['invoice_month'].fillna(invoices['order_month'])
    invoices.fillna(0, inplace=True)

    invoices[TIMEIND] = invoices['invoice_year'].sub(invoices['order_year']).mul(12).add(invoices['invoice_month'].sub(invoices['order_month'])).astype(int)
    invoices[TIMEIND] = invoices[TIMEIND].clip(lower=0)
    orders['N'] = orders.index.get_level_values('fund_year').astype('int') - orders['order_year']
    invoices = invoices.fillna(0)
    invoices['invoice_year'] = invoices['invoice_year'] - invoices[TIMEIND]
    invoices = invoices.set_index(TIMEIND, append=True)


    invoices = invoices[INVOL].sum(axis=1).rename("volume")
    invoices = invoices.groupby(level = invoices.index.names).sum()

    invoices = invoices.unstack(TIMEIND).fillna(0)
    invoices = invoices.reindex(range(invoices.columns.max() + 1), axis=1, fill_value=0)

    orders = orders.join(pd.cut(np.log(orders.loc[orders['po_net_value'].astype(float) > 0, 'po_net_value']), bins=10).rename('po_net_value_category'), how='left', on=glb.KEY)
    orders['quarter'] = pd.to_datetime(orders['order_date']).dt.quarter

    return orders, invoices, past_sums, order_edits


def preprocess(
    orders: pd.DataFrame,
    invoices: pd.DataFrame,
    orders_dates: pd.DataFrame,
    order_edits: pd.DataFrame,
    curr_year: int,
    curr_month: int,
):

    orders = combine_dates(orders, orders_dates)
    return process(orders, invoices, order_edits, curr_year, curr_month)