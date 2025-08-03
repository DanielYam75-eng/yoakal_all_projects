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


def main():

    parser = argparse.ArgumentParser(description="The Main Program")
    parser.add_argument("--hashbarot_data",  type=str,  required=True,                   help="Path to the hashbarot data")
    parser.add_argument("--main_data",       type=str,  required=True,                   help="Path to the main data")
    parser.add_argument("--past_year",       type=int,  required=True,                   help="Year to forecast")
    parser.add_argument("--curr_year",       type=int,  required=True,                   help="Current Year")
    parser.add_argument("--curr_month",      type=int,  required=True,                   help="Current Month")
    parser.add_argument("--months_back",     type=int,  required=False, default = -1,  help="Month to train on")
    parser.add_argument("--experiment_mode", type=bool, required=False, default = False, help="Experiment mode")
    parser.add_argument("--coin_type",      type=int,  required=True, default = 1, help="Coin Type")
    past_year = parser.parse_args().past_year
    months_back = parser.parse_args().months_back
    curr_year = parser.parse_args().curr_year
    curr_month = parser.parse_args().curr_month
    exp_mode = parser.parse_args().experiment_mode
    coin_type = parser.parse_args().coin_type

    IND  = 'kvotzat otzar'
    COL  = 'kvuzat sahar'
    VAL  = 'anual'
    TABLES = [f'_{curr_year}'] if exp_mode else [f'_{past_year}', f'_{curr_year}', f'actual_data_{past_year}_bad_otzar_only', f'actual_data_{curr_year}_bad_otzar_only']
    TO_EVAL = []

    print("Comencing program...")
    print("Close all relevent tables !!")
    preprocess_thread = threading.Thread(target=preprocess_main, args=(parser.parse_args().main_data, curr_year, coin_type))
    preprocess_thread.start()
    preprocess_thread.join()
    print("Finished preprocessing data")
    if not exp_mode:
        print("Working on hashbarot...")
        hashbarot_thread = threading.Thread(target=hashbarot_main, args=(parser.parse_args().hashbarot_data, past_year, curr_year, curr_month, months_back, coin_type))
        hashbarot_thread.start()
        hashbarot_thread.join()
        print("Finished hashbarot")
    print("Working on forcasting the rest...")


    def run_table(table):
        name = table[len('result-'):-len('data-preprocessed-by-posting-date.csv') - 1]
        run_notebook_thread = threading.Thread(target=run_notebook_main, args=(table, name, past_year, curr_year, curr_month, months_back, coin_type))
        run_notebook_thread.start()
        run_notebook_thread.join()

    tables = [table for table in os.listdir() if table.endswith('data-preprocessed-by-posting-date.csv')]

    with ThreadPoolExecutor(max_workers = len(tables)) as executor:  executor.map(run_table, tables)

    for table_type in TABLES:

        files = [f for f in os.listdir() if (f.startswith('forcast') and f.endswith(table_type + '.csv')) or f.startswith(table_type)]

        forcasts = pd.concat([pd.read_csv(f) for f in files])

        months = forcasts.columns.difference([IND, COL])
        if not exp_mode and table_type == TABLES[1]: forcasts.to_csv(r'Data\ALL_' + table_type + '_monthly' + '.csv', index = False)
        print(months[-12:])
        forcasts[VAL] = forcasts[months[-12:]].sum(axis = 1)

        forcasts = forcasts.pivot_table(index = IND, columns = COL, values = VAL, aggfunc = 'sum')

        forcasts['sum'] = forcasts.select_dtypes(include = 'number').fillna(0).sum(axis = 1)

        if table_type == TO_EVAL and not exp_mode: forcasts = forcasts['sum']

        if exp_mode: forcasts.to_csv(r'Data\ALL_' + table_type + f'_{curr_year}_' + f'_{curr_month}_' + f'_{months_back}' + '.csv')
        else:        forcasts.to_csv(r'Data\ALL_' + table_type + '.csv')


    if not exp_mode:
        pd.concat([pd.read_csv(f, index_col = IND) for f in os.listdir() if f.startswith('full_actual')], axis = 1).fillna(0).sort_index().to_csv(rf'Data\ALL_actual_data_{past_year}.csv')

    print("Cleaning...")
    clean_thread = threading.Thread(target=clean_main)
    clean_thread.start()
    clean_thread.join()


if __name__ == "__main__":
    main()
    print("Done")
