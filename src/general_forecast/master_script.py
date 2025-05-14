# %%
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
import argparse
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

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
    TABLES = [f'_{past_year}', f'_{curr_year}', f'actual_data_{past_year}_bad_otzar_only', f'actual_data_{curr_year}_bad_otzar_only']
    TO_EVAL = TABLES[0]

    print("Comencing program...")
    print("Close all relevent tables !!")
    print(subprocess.run(["python", "preprocess_data.py"], capture_output=True, text=True).stderr)
    print("Finished preprocessing data")
    print("Working on hashbarot...")
    print(subprocess.run(["python", "hashbarot_model.py", "--past_year",  past_year, "--curr_year",  curr_year, "--curr_month",  curr_month, "--months_back", months_back], capture_output=True, text=True).stderr)
    print("Finished hashbarot")
    print("Working on forcasting the rest...")


    def run_table(table):
        result = subprocess.run(["python", "run_notebook.py", "--path", table, "--past_year", past_year, "--curr_year", curr_year, "--curr_month", curr_month, "--months_back", months_back], capture_output=True, text=True)
    
        if result.stderr: print(f"Error in {table}:\n{result.stderr}")
        else:             print(f"Finished {table}")

    tables = [table for table in os.listdir() if table.endswith('data-preprocessed-by-posting-date.csv')]

    with ThreadPoolExecutor(max_workers = len(tables)) as executor:  executor.map(run_table, tables)


    for table_type in TABLES:

        files = [f for f in os.listdir() if (f.startswith('forcast') and f.endswith(table_type + '.csv')) or f.startswith(table_type)]

        forcasts = pd.concat([pd.read_csv(f) for f in files])

        months = forcasts.columns.difference([IND, COL])
        forcasts[VAL] = forcasts[months].sum(axis = 1)

        forcasts = forcasts.pivot_table(index = IND, columns = COL, values = VAL, aggfunc = 'sum')

        forcasts['sum'] = forcasts.select_dtypes(include = 'number').fillna(0).sum(axis = 1)

        if table_type == TO_EVAL: forcasts = forcasts['sum']
            
        forcasts.to_csv('ALL_' + table_type + '.csv')


    pd.concat([pd.read_csv(f, index_col = IND) for f in os.listdir() if f.startswith('full_actual')], axis = 1).fillna(0).sum(axis = 1).sort_index().to_csv(f'ALL_actual_data_{past_year}.csv')

    print(subprocess.run(["python", "evaluate.py", '-f', f"ALL__{past_year}.csv", '-t', f"ALL_actual_data_{past_year}.csv", '-o', f"{past_year}_grades.csv"], capture_output=True, text=True).stderr)    


if __name__ == "__main__":
    main()