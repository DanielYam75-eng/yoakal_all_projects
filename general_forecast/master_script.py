# %%
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd




def main():

    IND  = 'kvotzat otzar'
    COL  = 'kvuzat sahar'
    VAL  = 'anual'
    YEAR = '2024'
    TABLES = ['_2024', '_2025', 'actual_data_2024_without_bad_otzar_only', 'actual_data_2024_with_bad_otzar_only']



    print(subprocess.run(["python", "preprocess_data.py"], capture_output=True, text=True).stderr)

    def run_table(table):
        result = subprocess.run(["python", "run_notebook.py", "--path", table], capture_output=True, text=True)
    
        if result.stderr: print(f"Error in {table}:\n{result.stderr}")
        else:             print(f"Finished {table}")

    tables = [table for table in os.listdir() if table.endswith('data-preprocessed-by-posting-date.csv')]

    with ThreadPoolExecutor(max_workers = len(tables)) as executor:  executor.map(run_table, tables)

    for table_type in TABLES:

        files = [f for f in os.listdir() if (not f.startswith('ALL') and f.endswith(table_type + '.csv')) or f.startswith(table_type)]

        forcasts = pd.concat([pd.read_csv(f) for f in files])

        months = forcasts.columns.difference([IND, COL])
        forcasts[VAL] = forcasts[months].sum(axis = 1)

        forcasts = forcasts.pivot_table(index = IND, columns = COL, values = VAL, aggfunc = 'sum')

        forcasts['sum'] = forcasts.select_dtypes(include = 'number').fillna(0).sum(axis = 1)

        forcasts.to_csv('ALL_' + table_type + '.csv')

        TABLES = ['forcast', 'actual', 'result']



    for f in os.listdir():
        for table_type in TABLES:
            if f.startswith(table_type) and f.endswith('.csv'):
                os.remove(f)


    forcasts = pd.read_csv(f'ALL__{YEAR}.csv', index_col = IND)
    actual   = pd.read_csv(f'ALL_actual_data_{YEAR}_with_bad_otzar_only.csv', index_col = IND)

    forcasts = forcasts[sorted(forcasts.columns)].sort_index()
    actual   =   actual[sorted(actual.columns)  ].sort_index()

    def get_metric(metric) -> pd.DataFrame:
        return pd.DataFrame(np.vectorize(metric)(forcasts.values, actual.values), index = forcasts.index)

    get_metric(lambda x, y : abs(x - y)).to_csv(f'ALL_compare_{YEAR}.csv')





if __name__ == "__main__":
    main()