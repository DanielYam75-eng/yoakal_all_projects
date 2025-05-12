# %%
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
import argparse
import pandas as pd


def main():

    parser = argparse.ArgumentParser(description="The Main Program")
    parser.add_argument("--past_year", type=str, required=True, help="Year to forecast")
    parser.add_argument("--curr_year", type=str, required=True, help="Current Year")
    parser.add_argument("--curr_month", type=str, required=True, help="Current Month")
    parser.add_argument("--months_back", type=str, required=False, default = '-1', help="Month to train on")

    past_year = parser.parse_args().past_year
    months_back = parser.parse_args().months_back
    curr_year = parser.parse_args().curr_year
    curr_month = parser.parse_args().curr_month

    IND  = 'kvotzat otzar'
    COL  = 'kvuzat sahar'
    VAL  = 'anual'
    TABLES = [f'_{past_year}', f'_{curr_year}', f'actual_data_{past_year}_bad_otzar_only', f'actual_data_2025_bad_otzar_only']
    TO_ASHB = {TABLES[1], TABLES[3]}
    TO_EVAL = TABLES[0]
    BYPROD = ['forcast', 'actual', 'result', 'full_actual']

    print(subprocess.run(["python", "preprocess_data.py"], capture_output=True, text=True).stderr)
    print(subprocess.run(["python", "ashbarot_model.py"], capture_output=True, text=True).stderr)
    hasbarot = pd.read_csv('ashbarot_2025.csv', index_col = IND)

    def run_table(table):
        result = subprocess.run(["python", "run_notebook.py", "--path", table, "--past_year", past_year, "--curr_year", curr_year, "--curr_month", curr_month, "--months_back", months_back], capture_output=True, text=True)
    
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

        if table_type in TO_ASHB:
            forcasts = forcasts.merge(hasbarot, how = 'left', left_on = IND, right_on = IND)

        forcasts['sum'] = forcasts.select_dtypes(include = 'number').fillna(0).sum(axis = 1)

        if table_type == TO_EVAL: forcasts = forcasts['sum']
            
        forcasts.to_csv('ALL_' + table_type + '.csv')

    pd.concat([pd.read_csv(f, index_col = IND) for f in os.listdir() if f.startswith('full_actual')], axis = 1).fillna(0).sum(axis = 1).sort_index().to_csv(f'ALL_actual_data_{past_year}.csv')

    for f in os.listdir():
        for table_type in BYPROD:
            if f.startswith(table_type) and f.endswith('.csv'):
               os.remove(f)
   
    forcasts = pd.read_csv(f'ALL__{past_year}.csv', index_col = IND)
    actual   = pd.read_csv(f'ALL_actual_data_{past_year}_bad_otzar_only.csv', index_col = IND)

    forcasts = forcasts[sorted(forcasts.columns)].sort_index()
    actual   =   actual[sorted(actual.columns)  ].sort_index()


    print(subprocess.run(["python", "evaluate.py", '-f', f"ALL__{past_year}.csv", '-t', f"ALL_actual_data_{past_year}.csv", '-o', f"{past_year}_grades.csv"], capture_output=True, text=True).stderr)    


if __name__ == "__main__":
    main()

