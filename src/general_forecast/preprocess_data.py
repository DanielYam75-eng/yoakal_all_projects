# %%
import argparse
import pandas as pd
import sys

# %%
parser = argparse.ArgumentParser(description="Forecasting script")
parser.add_argument("--path", type=str, required=True, help="Path to the CSV file")
parser.add_argument("--current-year", type=int, required=True, help="The year to which we create the forecast for")
data = pd.read_csv(r"Data\\" + parser.parse_args().path)
current_year = parser.parse_args().current_year

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
data = data[data['fund_code'] != 1410]


# %%
data['type'] = 'rest'

# %%
data.loc[data['doc_type'] == 'ZC', 'type'] = 'affilated_other'  # Should be before the salary specification
data.loc[(data['fund_code'].isin([1401, 1402, 1408, 1409, 1411, 1412, 1413, 1414, 1415, 1416, 1099, 1523])) & (data['doc_type'].isin(['ZC', 'ZW'])) & (data['expenditure_type'] == 3010), 'type'] = 'salary'
data.loc[(data['doc_type'].isin(['KR', 'KG'])) & (data['law'] == 300), 'type'] = 'vehicles'  # should be before cor
data.loc[(data['doc_type'].isin(['KR', 'KG'])) & (data['law'] == 302), 'type'] = 'overseas_transportation'
data.loc[(data['doc_type'].isin(['KR', 'KG'])) & (data['law'] == 706), 'type'] = 'tariffs'
data.loc[(data['doc_type'].isin(['KR', 'KG'])) & (data['law'] == 2900), 'type'] = 'insurance'
data.loc[(data['doc_type'].isin(['KR', 'KG'])) & (data['law'] == 1316), 'type'] = 'special_compensation'
data.loc[(data['law'] == 9800), 'type'] = 'special_research'

data.loc[(data['fund_code'].isin([1400, 1403, 1405, 1406, 1407, 1423, 1425])), 'type'] = 'cor'
data.loc[data['doc_type'].isin(['ZD']), 'type'] = 'arnona'
data.loc[data['expenditure_type'] == 1020, 'type'] = 'electricity'
data.loc[data['expenditure_type'] == 1030, 'type'] = 'water'
data.loc[data['doc_type'].isin(['KM']), 'type'] = 'KM'
data.loc[data['doc_type'].isin(['KT']), 'type'] = 'KT'
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
