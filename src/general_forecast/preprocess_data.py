import argparse
import pandas as pd
import sys
from read_file import read

def main(path, current_year, coin_type):
    # %%
    data = read(path, sep='\t')


    # %%
    data = data.melt(id_vars=['financial_year', 'economy', 'expenditure_type', 'doc_type', 'fund_code', 'fingroup', 'law'], var_name='month', value_name='volume')

    # %%
    data.loc[data['volume'].astype(str).str.replace(',', '').str.replace(' ', '')== '-', 'volume'] = 0
    data['volume'] = data['volume'].astype(str).str.replace(',', '').str.replace(' ', '').astype(float)

    # %%
    data = data.dropna(subset=['financial_year'])

    # %%
    data.index = pd.to_datetime(data['financial_year'].astype(int).astype(str) + data['month'].astype(str), format='%Y%m') + pd.offsets.MonthEnd(0)

    # %%
    data.fillna({'volume': 0}, inplace=True)

    # %%
    data = data[~data['doc_type'].isin(['RE', 'ZY', 'ZF', 'ZH'])]

    data = data[~data['fund_code'].isin([1410,1405,1400,1406])]

    # %%
    data['type'] = 'rest'

    # %%
    if coin_type == 1:
        data.loc[data['doc_type'] == 'ZC', 'type'] = 'affilated_other'  # Should be before the salary specification
        data.loc[(data['doc_type'].isin(['KR', 'KG'])) & (data['law'] == 300), 'type'] = 'vehicles'  # should be before cor
        data.loc[(data['fund_code'] == 1411) & (data['doc_type'] == 'ZC'), 'type'] = 'career_salary'
        data.loc[(data['fund_code'] == 1409) & (data['doc_type'] == 'ZC'), 'type'] = 'drafted_salary'
        data.loc[(data['fund_code'].isin([1401, 1402])) & (data['doc_type'].isin(['ZC','ZW'])), 'type'] = 'pensions'
        data.loc[(data['fund_code'].isin([1408])) & (data['doc_type'] == 'ZC'), 'type'] = 'idf_workers_salary'
        data.loc[(data['fund_code'].isin([1413, 1414, 1415])) & (data['doc_type'].isin(['ZC', 'ZW'])), 'type'] = 'dd_workers_salary'
        data.loc[(data['fund_code'] == 1412) & (data['doc_type'] == 'ZC'), 'type'] = 'pre_draft_salary'
        data.loc[(data['fund_code'] == 1416) & (data['doc_type'] == 'ZC'), 'type'] = 'additional_drafted_service_salary'
        data.loc[(data['doc_type'].isin(['KR', 'KG'])) & (data['law'] == 302), 'type'] = 'overseas_transportation'
        data.loc[(data['doc_type'].isin(['KR', 'KG'])) & (data['law'] == 706), 'type'] = 'tariffs'
        data.loc[(data['doc_type'].isin(['KR', 'KG'])) & (data['law'] == 2900), 'type'] = 'insurance'
        data.loc[(data['doc_type'].isin(['KR', 'KG'])) & (data['law'] == 1316), 'type'] = 'special_compensation'
        data.loc[(data['law'] == 9800), 'type'] = 'special_research'
        data.loc[(data['fund_code'].isin([1423, 1425])), 'type'] = 'families'
        data.loc[(data['fund_code'].isin([1403])), 'type'] = 'commemoration'
        data.loc[data['doc_type'].isin(['ZD']), 'type'] = 'arnona'
        data.loc[data['expenditure_type'] == 1020, 'type'] = 'electricity'
        data.loc[data['expenditure_type'] == 1030, 'type'] = 'water'
        data.loc[data['doc_type'].isin(['KM']), 'type'] = 'KM'
        data.loc[data['doc_type'].isin(['KT']), 'type'] = 'KT'
        data.loc[data['doc_type'].isin(['SA']), 'type'] = 'SA'
        data.to_csv('result-data-preprocessed-by-posting-date_all.csv')
    if coin_type == 5:
        data.loc[(data['doc_type'] == 'ZW'), 'type'] = 'ZW'
        data.loc[(data['doc_type'] == 'ZC'), 'type'] = 'ZC'

        data.loc[(data['doc_type'].isin(['KR','KG'])) & (data['expenditure_type'].isin([2010,2030,2050,2015,2035, 2045])), 'type'] = 'travel-KRKG'
        data.loc[(data['doc_type'].isin(['KR','KG'])) & (data['fund_code'].isin(range(1400, 1479))), 'type'] = '14-KRKG'

        data.loc[data['doc_type'].isin(['SA']), 'type'] = 'SA'



    # %%
    frames : dict[str, pd.DataFrame] = {name : data[data['type'] == name] for name in data['type'].unique() }

    # %% 
    frames = { name : frames[name].groupby('fingroup').resample('ME').sum()['volume'] for name in frames }
    # %%
    frames = { name : frames[name].groupby([frames[name].index.get_level_values(0), frames[name].index.get_level_values(1).year]).cumsum() for name in frames }

    # %%
    for frame in frames.values():
        frame.index.set_names(['OTZAR_GROUP', 'DT'], inplace=True)

    # %%
    for name, frame in frames.items():
        if not frame.empty and f'{current_year}-01-31' in frame.index.levels[1]:
            frame.to_csv('result-' + name + '-data-preprocessed-by-posting-date.csv')
        else: print(f"Frame {name} is empty. No data to save.", file=sys.stderr)

if __name__ == "__main__":
    main()
