# %%
import pandas as pd

# %%
data = pd.read_csv("entry_date_data.csv")

# %%
data = data.melt(id_vars=['financial_year', 'economy', 'expenditure_type', 'doc_type', 'fund_code', 'fingroup'], var_name='month', value_name='volume')

# %%
data.head(10)

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
data.head(10)

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
data.head(10)

# %%
salary_data = data[data['type'] == 'salary']
cor_data = data[data['type'] == 'cor']
KM_data = data[data['type'] == 'market']
KT_data = data[data['type'] == 'KT']
ZH_data = data[data['type'] == 'ZH']
nesah_data = data[data['type'] == 'nesah']
rest_data = data[data['type'] == 'any']

# %%
salary_grouped = salary_data.groupby('fingroup').resample('ME').sum()['volume']
cor_grouped = cor_data.groupby('fingroup').resample('ME').sum()['volume']
KM_grouped = KM_data.groupby('fingroup').resample('ME').sum()['volume']
KT_grouped = KT_data.groupby('fingroup').resample('ME').sum()['volume']
ZH_grouped = ZH_data.groupby('fingroup').resample('ME').sum()['volume']
nesah_grouped = nesah_data.groupby('fingroup').resample('ME').sum()['volume']
rest_grouped = rest_data.groupby('fingroup').resample('ME').sum()['volume']

# %%
rest_data

# %%
rest_data[rest_data['financial_year'] == 2024].groupby('doc_type')['volume'].sum().sort_values(ascending=False).head(10)

# %%
frames = [salary_grouped, cor_grouped, KM_grouped, KT_grouped, ZH_grouped, nesah_grouped, rest_grouped]

# %%
frames = [frame.groupby([frame.index.get_level_values(0), frame.index.get_level_values(1).year]).cumsum()  for frame in frames]

# %%
for frame in frames:
    frame.index.set_names(['OTZAR_GROUP', 'DT'], inplace=True)

# %%
names = ['salary', 'cor', 'KM', 'KT', 'ZH', 'nesah', 'rest']

# %%
for name, frame in zip(names, frames):
    frame.to_csv('result-' + name + '-data-preprocessed-by-posting-date.csv')


