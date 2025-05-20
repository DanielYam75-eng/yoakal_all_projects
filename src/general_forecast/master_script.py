import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
import argparse
import pandas as pd
import warnings
warnings.filterwarnings("ignore")




def main():

    parser = argparse.ArgumentParser(description="The Main Program")
    parser.add_argument("--hashbarot_data",  type=str,  required=True,                   help="Path to the hashbarot data")
    parser.add_argument("--main_data",       type=str,  required=True,                   help="Path to the main data")
    parser.add_argument("--past_year",       type=str,  required=True,                   help="Year to forecast")
    parser.add_argument("--curr_year",       type=str,  required=True,                   help="Current Year")
    parser.add_argument("--curr_month",      type=str,  required=True,                   help="Current Month")
    parser.add_argument("--months_back",     type=str,  required=False, default = '-1',  help="Month to train on")
    parser.add_argument("--experiment_mode", type=bool, required=False, default = False, help="Experiment mode")

    past_year = parser.parse_args().past_year
    months_back = parser.parse_args().months_back
    curr_year = parser.parse_args().curr_year
    curr_month = parser.parse_args().curr_month
    exp_mode = parser.parse_args().experiment_mode

    IND  = 'kvotzat otzar'
    COL  = 'kvuzat sahar'
    VAL  = 'anual'
    TABLES = [f'_{curr_year}'] if exp_mode else [f'_{past_year}', f'_{curr_year}', f'actual_data_{past_year}_bad_otzar_only', f'actual_data_{curr_year}_bad_otzar_only']
    TO_EVAL = TABLES[0]

    print("Comencing program...")
    print("Close all relevent tables !!")
    print(subprocess.run(["python", "preprocess_data.py", "--path", parser.parse_args().main_data, '--current-year', curr_year], capture_output=True, text=True).stderr)
    print("Finished preprocessing data")
    if not exp_mode:
        print("Working on hashbarot...")
        print(subprocess.run(["python", "hashbarot_model.py", "--path", parser.parse_args().hashbarot_data, "--past_year",  past_year, "--curr_year",  curr_year, "--curr_month",  curr_month, "--months_back", months_back], capture_output=True, text=True).stderr)
        print("Finished hashbarot")
    print("Working on forcasting the rest...")


    def run_table(table):
        name = table[len('result-'):-len('data-preprocessed-by-posting-date.csv') - 1]
        result = subprocess.run(["python", "run_notebook.py", "--path", table, "--type", name, "--past_year", past_year, "--curr_year", curr_year, "--curr_month", curr_month, "--months_back", months_back], capture_output=True, text=True)
    
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

        if table_type == TO_EVAL and not exp_mode: forcasts = forcasts['sum']

        if exp_mode:    
            forcasts.to_csv(r'Data\ALL_' + table_type + f'_{curr_year}_' + f'_{curr_month}_' + f'_{months_back}' + '.csv')
        else:
            forcasts.to_csv(r'Data\ALL_' + table_type + '.csv')


    if not exp_mode:
        pd.concat([pd.read_csv(f, index_col = IND) for f in os.listdir() if f.startswith('full_actual')], axis = 1).fillna(0).sum(axis = 1).sort_index().to_csv(rf'Data\ALL_actual_data_{past_year}.csv')

    if not exp_mode:
        print("Grading...")
        print(subprocess.run(["python", "evaluate.py", '-f', rf"Data\ALL__{past_year}.csv", '-t', rf"Data\ALL_actual_data_{past_year}.csv", '-o', rf"Data\{past_year}_grades.csv"], capture_output=True, text=True).stderr)
    print("Cleaning...")
    subprocess.run(["python", "clean.py"], capture_output=True, text=True)


if __name__ == "__main__":
    main()
    print("Done")