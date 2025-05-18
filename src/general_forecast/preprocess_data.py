# %%
import argparse
import pandas as pd

# %%
parser = argparse.ArgumentParser(description="Forecasting script")
parser.add_argument("--path", type=str, required=True, help="Path to the CSV file")
data = pd.read_csv(r"Data\\" + parser.parse_args().path)

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
data['type'] = 'any'

# %%
data.loc[data['doc_type'] == 'ZC', 'type'] = 'affilated-other'  # Should be before the salary specification
data.loc[(data['fund_code'].isin([1401, 1402, 1408, 1409, 1411, 1412, 1413, 1414, 1415, 1416, 1099, 1523])) & (data['doc_type'].isin(['ZC', 'ZW'])) & (data['expenditure_type'] == 3010), 'type'] = 'salary'
data.loc[(data['doc_type'].isin(['KR', 'KG'])) & (data['law'] == '0300'), 'type'] = 'vehicles'  # should be before cor
data.loc[(data['doc_type'].isin(['KR', 'KG'])) & (data['law'] == '0302'), 'type'] = 'special-transportation'
data.loc[(data['doc_type'].isin(['KR', 'KG'])) & (data['law'] == '0706'), 'type'] = 'tariffs'
data.loc[(data['doc_type'].isin(['KR', 'KG'])) & (data['law'] == '2900'), 'type'] = 'insurance'
data.loc[(data['doc_type'].isin(['KR', 'KG'])) & (data['law'] == '1316'), 'type'] = 'special-compensation'
data.loc[(data['law'] == '9800'), 'type'] = 'special-research'

data.loc[(data['fund_code'].isin([1400, 1403, 1405, 1406, 1407, 1423, 1425])), 'type'] = 'cor'
data.loc[data['doc_type'].isin(['ZD']), 'type'] = 'arnona'
data.loc[data['expenditure_type'] == '1020', 'type'] = 'electricity'
data.loc[data['expenditure_type'] == '1030', 'type'] = 'water'
data.loc[data['doc_type'].isin(['KM']), 'type'] = 'market'
data.loc[data['doc_type'].isin(['KT']), 'type'] = 'KT'
data.loc[data['doc_type'].isin(['SA']), 'type'] = 'SA'




# %%
names = ['salary', 'cor', 'arnona', 'KM', 'KT', 'electricity', 'water', 'vehicles', 'overseas-transportation', 'tariffs', 'insurance', 'special-compensation', 'special-research', 'SA', 'rest']
frames = [
data[data['type'] == 'salary'],
data[data['type'] == 'cor'],
data[data['type'] == 'arnona'],
data[data['type'] == 'market'],
data[data['type'] == 'KT'],
data[data['type'] == 'electricity'],
data[data['type'] == 'water'],
data[data['type'] == 'vehicles'],
data[data['type'] == 'special-transportation'],
data[data['type'] == 'tariffs'],
data[data['type'] == 'insurance'],
data[data['type'] == 'special-compensation'],
data[data['type'] == 'special-research'],
data[data['type'] == 'SA'],
data[data['type'] == 'any']]

# %%
frames = [frame.groupby('fingroup').resample('ME').sum()['volume'] for frame in frames]

# %%
frames = [frame.groupby([frame.index.get_level_values(0), frame.index.get_level_values(1).year]).cumsum()  for frame in frames]

# %%
for frame in frames:
    frame.index.set_names(['OTZAR_GROUP', 'DT'], inplace=True)

# %%
for name, frame in zip(names, frames):
    if not frame.empty:
        frame.to_csv('result-' + name + '-data-preprocessed-by-posting-date.csv')
