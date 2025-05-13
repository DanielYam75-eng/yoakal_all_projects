import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, root_mean_squared_error
import os
import pickle
import argparse


def multiyear_prediction(target_year, results, ground_truth, n_years=3):
    target_year = int(target_year)
    total = sum(results.loc[str(target_year - i)].sum() for i in range(0, n_years))
    return total - sum(ground_truth.loc[target_year - i] for i in range(1, n_years))

def parse(name):
    return name.split('_')[-1].split('.')[0]

def preprocess(table1, table2):
    table1 = table1.copy()
    table2 = table2.copy()
    table2 = table2.drop_duplicates(subset=['doc_id', 'fund_year'], keep='first').set_index(['doc_id', 'fund_year'])
    table2['start_year'] = pd.concat([pd.Series(table2.index.get_level_values('fund_year').astype(int), index=table2.index), pd.to_datetime(table2['doc_date'], format='%d.%m.%Y').dt.year], axis=1).max(axis=1)
    table2['N'] = table2.index.get_level_values('fund_year') - pd.to_datetime(table2['doc_date'], format='%d.%m.%Y').dt.year
    table1 = table1.fillna(0)
    table1 = table1.join(table2['start_year'], how='left', on=['doc_id', 'fund_year'])
    table1['invoice_year'] = table1['financial_year'] - table1['start_year']
    table1 = table1.set_index(['doc_id', 'fund_year', 'invoice_year'])
    table1 = table1[['invoice_volume']]
    table1 = table1.groupby(level=[0, 1, 2]).sum()
    table1 = table1.unstack()
    table1 = table1.fillna(0)
    table1 = table1.droplevel(0, axis=1)
    table2 = table2.join(pd.cut(np.log(table2.loc[table2['po_net_value'] > 0, 'po_net_value']), bins=10).rename('po_net_value_category'), how='left', on=['doc_id', 'fund_year'])
    table2['quarter'] = pd.to_datetime(table2['doc_date'], format='%d.%m.%Y').dt.quarter
    return table1, table2

class Trainer:
    def __init__(self, model, data, features_list=['po_type', 'N'], labels_list=[0, 1, 2, 3, 4]):
        self.data = data
        self.model = model
        self.X_train = None
        self.y_train = None
        self.features_list = features_list
        self.labels_list = labels_list
    
    def train_and_evaluate(self, split_year=2022):
        features_table, labels_table = self._create_features_and_labels_tables()
        self.model = self.split_and_fit(features_table, labels_table, split_year=split_year)
        rmse_in_sample, r2_in_sample = self.evaluate_in_sample()
        rmse_out_of_sample, r2_out_of_sample = self.evaluate_out_of_sample()
        return dict(rmse_in_sample=rmse_in_sample, r2_in_sample=r2_in_sample, rmse_out_of_sample=rmse_out_of_sample, r2_out_of_sample=r2_out_of_sample)
    
    def _create_features_and_labels_tables(self):
        labels_table = self.data.loc[:, self.labels_list].div(self.data['po_net_value'], axis=0).fillna(0)
        features_table = self.data.loc[:, self.features_list].astype('category')

        # This masks filters out anomalous rows.
        mask = (np.all(labels_table <= 1, axis=1)) & (np.all(labels_table >= 0, axis=1))
        features_table = features_table[mask]
        labels_table = labels_table[mask]
        return features_table, labels_table

    @staticmethod
    def _split(features_table, labels_table, split_year=2022):
        mask = features_table.index.get_level_values('fund_year') <= split_year
        return features_table[mask], labels_table[mask]
    
    def split_and_fit(self, split_year=2022):
        features_table, labels_table = self._create_features_and_labels_tables()
        X_train, y_train = self._split(features_table, labels_table, split_year)
        self.model.fit(X_train, y_train)
        self.X_train = X_train
        self.y_train = y_train
        return self.model
    
    def evaluate_in_sample(self):
        rmse = root_mean_squared_error(self.y_train, self.model.predict(self.X_train))
        r2 = r2_score(self.y_train, self.model.predict(self.X_train))
        return rmse, r2
    
    def evaluate_out_of_sample(self):
        rmse = root_mean_squared_error(self.y_test, self.model.predict(self.X_test))
        r2 = r2_score(self.y_test, self.model.predict(self.X_test))
        return rmse, r2


class Wrapper:
    def __init__(self, model, features, labels: pd.Index):
        self.model = model
        self.labels = labels
        self.features = features
    
    def _get_X(self, df, target_year):
        df = df.copy()
        df = df.reset_index(['doc_id', 'fund_year'])
        df['opening_year'] = pd.to_datetime(df['doc_date'], format='%d.%m.%Y').dt.year.astype(int)
        df['diffyear'] = target_year - df[['fund_year', 'opening_year']].max(axis=1)
        df['po_net_value'] = df['po_net_value']
        df = df.set_index(['doc_id', 'fund_year'])
        for feature in self.features:
            df[feature] = df[feature].astype('category')
        return df[self.features], df['diffyear']
    
    def _predict(self, X) -> pd.DataFrame:
        return pd.DataFrame(self.model.predict(X), index=X.index, columns=self.labels)
    
    def _get_results(self, y_pred, net_values, diff_years):
        y_pred = y_pred.stack()
        y_pred.index = y_pred.index.set_names('diffyear', level=2)
        y_pred.name = 'coefficient'
        results = pd.DataFrame(diff_years).join(y_pred, how='left', on=['doc_id', 'fund_year', 'diffyear']).fillna(0).join(net_values, how='left', on=['doc_id', 'fund_year'])
        results = results['coefficient'] * results['po_net_value']
        return results
    
    def predict(self, df, target_year):
        X, diffyear = self._get_X(df, target_year)
        df = df.copy()
        df = df.reset_index(['doc_id', 'fund_year'])
        y_pred = self._predict(X)
        net_values = df[['doc_id', 'fund_year', 'po_net_value']].set_index(['doc_id', 'fund_year'])['po_net_value']
        results = self._get_results(y_pred, net_values, diffyear)
        return results
        

def rmse(true, prediction):
    return np.sqrt(((true-prediction)**2).sum()) / len(true)


def log_rmse(true, prediction):
    return rmse(np.log(true), np.log(prediction))


def train_model(table1, table2, features, labels, model, split_year):
    data = table1.join(table2, how='left', on=['doc_id', 'fund_year'])
    trainer = Trainer(model, data, features_list=features, labels_list=table1.columns)
    return trainer.split_and_fit(split_year=split_year)


def read(path1, path2):
    table1 = pd.read_csv(path1)
    table1 = table1.fillna(0)
    table1['invoice_volume'] = table1['RE']
    table1 = table1.drop(columns=['RE', 'ZF', 'ZY', 'fingroup'])
    table2 = pd.read_csv(path2)
    table2['po_net_value'] = table2['po_net_value'].astype(str).str.replace(',', '').astype(float)
    table1, table2 = preprocess(table1, table2)
    table2['fund_code'] = table2['fund_code'].astype(str)
    table2.loc[:, 'fund_code'] = table2['fund_code'].str[-4:]
    table2.loc[:, 'doc_date'] = pd.to_datetime(table2['doc_date'], format='%d.%m.%Y')
    return table1, table2


def train():
    # Loading Data + Feature Engineering
    parser = argparse.ArgumentParser()
    parser.add_argument('paths', nargs=2)
    parser.add_argument('-o', '--output', nargs=1, required=True)
    args = parser.parse_args()
    path1 = args.paths[0]
    path2 = args.paths[1]
    target_model_path = args.output[0]
    table1, table2 = read(path1, path2)
    print("Preprocessing Finished")

    feature_list3 = ['po_type', 'fingroup', 'procurment_organization', 'N', 'po_net_value_category', 'quarter']
    xgboost_model3 = XGBRegressor(enable_categorical=True, tree_method='hist', n_estimators=1000, learning_rate=0.1, max_depth=3, random_state=42)
    xgboost_model3 = train_model(table1, table2, feature_list3, table1.columns, xgboost_model3, split_year=2021)
    print("Training Finished")
    pickle.dump(xgboost_model3, open(target_model_path, 'wb'))


def infer():
    parser = argparse.ArgumentParser()
    parser.add_argument('paths', nargs=2)
    parser.add_argument('-m', '--model', nargs=1, required=True)
    parser.add_argument('-y', '--year', nargs=1, required=True, type=int)
    parser.add_argument('-o', '--output', nargs=1, required=True)
    args = parser.parse_args()
    model_path = args.model[0]
    model = pickle.load(open(model_path, 'rb'))
    target_year = args.year[0]
    feature_list = ['po_type', 'fingroup', 'procurment_organization', 'N', 'po_net_value_category', 'quarter']
    path1 = args.paths[0]
    path2 = args.paths[1]
    output_path = args.output[0]
    table1, table2 = read(path1, path2)
    wrapper = Wrapper(model, feature_list, table1.columns)

    # Augmentation
    temp = table2[(table2['doc_date'] > pd.Timestamp('2024-05-13')) & (table2['doc_date'] < pd.Timestamp('2025-01-01'))]
    temp.loc[:, 'doc_date'] = temp['doc_date'] + pd.DateOffset(years=1)
    temp.index = temp.index.set_levels(temp.index.levels[0].astype(str) + 'N', level='doc_id')
    temp.index = temp.index.set_levels(temp.index.levels[1] + 1, level='fund_year')
    temp.loc[:, 'start_year'] = temp['start_year'] + 1
    augmented_table2 = pd.concat([table2, temp])
    print('Augmentation Finished')

    # Inference
    if not os.path.exists('results_new'):
        os.mkdir('results_new')
    wrapper = Wrapper(model, feature_list, table1.columns)
    (pd.DataFrame(wrapper.predict(augmented_table2, target_year), columns=['prediction'])
        .join(augmented_table2['fingroup'], on=['doc_id', 'fund_year'], how='left')
        .groupby('fingroup').sum()
        .to_csv(output_path))
