# %%
import pandas as pd

# %%
data = pd.read_csv("entry_date_data.csv")

# %%
data = data.melt(id_vars=['financial_year', 'economy', 'expenditure_type', 'doc_type', 'fund_code', 'fingroup'], var_name='month', value_name='volume')

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
data = data[~data['doc_type'].isin(['RE', 'ZY', 'ZF'])]
data = data[data['fund_code'] != 1410]


# %%
data['type'] = 'any'

# %%
data.loc[(~data['fund_code'].isin([1400, 1403, 1405, 1406, 1407, 1423, 1425])) & (~data['doc_type'].isin(['KM', 'KT', 'ZH'])) & (data['expenditure_type'] == 3010), 'type'] = 'salary'
data.loc[(data['fund_code'].isin([1400, 1403, 1405, 1406, 1407, 1423, 1425])) & (~data['doc_type'].isin(['KM', 'KT', 'ZH'])), 'type'] = 'cor'
data.loc[data['doc_type'].isin(['KM']), 'type'] = 'market'
data.loc[data['doc_type'].isin(['KT']), 'type'] = 'KT'
data.loc[data['doc_type'].isin(['ZH']), 'type'] = 'ZH'
data.loc[data['expenditure_type'].astype(str).str.startswith('2'), 'type'] = 'nesah'


# %%
names = ['salary', 'cor', 'KM', 'KT', 'ZH', 'nesah', 'rest']
frames = [data[data['type'] == 'salary'],
data[data['type'] == 'cor'],
data[data['type'] == 'market'],
data[data['type'] == 'KT'],
data[data['type'] == 'ZH'],
data[data['type'] == 'nesah'],
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
    frame.to_csv('result-' + name + '-data-preprocessed-by-posting-date.csv')


