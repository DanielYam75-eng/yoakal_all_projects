from typing import Callable
from numpy import log
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, root_mean_squared_error
import argparse


IND    = 'MOF_class'
ACT    = 'ground_truth'
PRED   = 'prediction'
SLC    = 'group'
SCORE  = 'score'
METRIC = 'metric'
STD_CLUSTERS =  pd.DataFrame.from_dict({"0101/05": "medium std",
                                        "0101/06": "high std",
                                        "0101/13": "medium std",
                                        "0102/04": "high std",
                                        "0102/07": "high std",
                                        "0109/05": "medium std",
                                        "0200/04": "low std",
                                        "0200/07": "low std",
                                        "0201/07": "low std",
                                        "0201/08": "medium std",
                                        "0201/09": "low std",
                                        "0201/11": "high std",
                                        "0201/12": "low std",
                                        "0203/20": "high std",
                                        "0300/07": "low std",
                                        "0301/02": "high std",
                                        "0350/04": "low std",
                                        "0352/04": "high std",
                                        "0353/04": "medium std",
                                        "0354/04": "medium std",
                                        "0355/04": "low std",
                                        "0356/04": "low std",
                                        "0357/04": "medium std",
                                        "0358/04": "high std",
                                        "0358/23": "low std",
                                        "0359/04": "high std",
                                        "0360/04": "high std",
                                        "0360/05": "low std",
                                        "0400/02": "medium std",
                                        "0499/03": "high std",
                                        "0500/01": "low std",
                                        "0500/02": "medium std",
                                        "0500/07": "low std",
                                        "0500/20": "medium std",
                                        "0500/23": "medium std",
                                        "0550/04": "high std",
                                        "0551/01": "high std",
                                        "0553/04": "high std",
                                        "0554/04": "medium std",
                                        "0555/04": "high std",
                                        "0600/01": "low std",
                                        "0600/02": "medium std",
                                        "0600/03": "high std",
                                        "0600/07": "low std",
                                        "0600/20": "low std",
                                        "0600/23": "low std",
                                        "0650/01": "low std",
                                        "0650/02": "high std",
                                        "0650/03": "medium std",
                                        "0652/01": "high std",
                                        "0690/03": "high std",
                                        "0695/01": "medium std",
                                        "0695/03": "medium std",
                                        "0696/03": "medium std",
                                        "0697/01": "high std",
                                        "0697/03": "medium std",
                                        "0698/03": "high std",
                                        "0699/01": "high std",
                                        "0699/03": "high std",
                                        "0700/01": "low std",
                                        "0700/02": "high std",
                                        "0700/07": "low std",
                                        "0700/20": "medium std",
                                        "0700/23": "low std",
                                        "0701/01": "low std",
                                        "0702/03": "high std",
                                        "0750/02": "high std",
                                        "0750/03": "medium std",
                                        "0751/01": "medium std",
                                        "0800/01": "medium std",
                                        "0800/02": "high std",
                                        "0800/03": "medium std",
                                        "0800/07": "low std",
                                        "0800/20": "medium std",
                                        "0800/21": "medium std",
                                        "0800/23": "low std",
                                        "0802/04": "low std",
                                        "0803/02": "medium std",
                                        "0803/04": "medium std",
                                        "0850/20": "high std",
                                        "0852/01": "high std",
                                        "0852/03": "medium std",
                                        "0898/01": "high std",
                                        "0899/01": "high std",
                                        "0900/01": "low std",
                                        "0900/07": "low std",
                                        "0900/21": "low std",
                                        "1100/01": "low std",
                                        "1100/07": "high std",
                                        "1100/51": "low std",
                                        "1101/01": "high std",
                                        "1101/03": "high std",
                                        "1101/51": "low std",
                                        "1102/01": "low std",
                                        "1102/03": "high std",
                                        "1102/51": "medium std",
                                        "1103/01": "medium std",
                                        "1103/03": "high std",
                                        "1106/01": "high std",
                                        "1106/07": "medium std",
                                        "1151/01": "medium std",
                                        "1152/03": "medium std",
                                        "1153/51": "medium std",
                                        "1161/01": "medium std",
                                        "1161/51": "low std",
                                        "1172/01": "medium std",
                                        "1173/01": "high std",
                                        "1200/04": "low std",
                                        "1200/07": "low std",
                                        "1214/05": "low std",
                                        "1300/01": "low std",
                                        "1300/03": "high std",
                                        "1300/07": "low std",
                                        "1360/03": "low std",
                                        "1400/02": "medium std",
                                        "1400/03": "high std",
                                        "1400/20": "low std",
                                        "1401/20": "low std",
                                        "1402/05": "high std",
                                        "1403/04": "high std",
                                        "1405/01": "high std",
                                        "1407/05": "medium std",
                                        "1411/02": "high std",
                                        "1411/04": "medium std",
                                        "1412/05": "high std",
                                        "1414/04": "medium std",
                                        "1449/03": "high std",
                                        "1499/03": "high std",
                                        "1500/30": "high std",
                                        "1500/32": "low std",
                                        "1500/34": "medium std",
                                        "1500/35": "low std",
                                        "1500/36": "low std",
                                        "1500/37": "low std",
                                        "1600/15": "low std",
                                        "1600/31": "low std",
                                        "1600/32": "low std",
                                        "1600/33": "medium std",
                                        "1701/04": "low std",
                                        "1701/07": "medium std",
                                        "1702/01": "low std",
                                        "1703/05": "low std",
                                        "1704/04": "low std",
                                        "1704/07": "medium std",
                                        "1705/05": "high std",
                                        "1705/31": "high std",
                                        "1706/05": "medium std",
                                        "1707/05": "medium std",
                                        "1709/02": "high std",
                                        "1801/04": "low std",
                                        "1801/07": "medium std",
                                        "1900/05": "high std",
                                        "1900/10": "low std",
                                        "1901/05": "high std",
                                        "1902/05": "medium std",
                                        "1903/05": "high std",
                                        "2000/04": "high std",
                                        "2000/05": "medium std",
                                        "2000/07": "medium std",
                                        "2001/05": "low std",
                                        "2001/07": "medium std",
                                        "2200/04": "low std",
                                        "2200/07": "high std",
                                        "2203/04": "medium std",
                                        "2258/20": "medium std",
                                        "2600/14": "medium std",
                                        "2600/15": "low std",
                                        "2800/07": "high std",
                                        "2800/20": "low std",
                                        "2803/26": "low std",
                                        "2803/28": "low std",
                                        "2900/02": "medium std",
                                        "2901/02": "medium std",
                                        "2901/20": "high std",
                                        "2903/20": "high std",
                                        "2904/02": "medium std",
                                        "2959/03": "high std"},
                                        orient = 'index',
                                        columns = ['cluster'])


def get_scores(data: pd.DataFrame, metric: Callable, slicing: str = IND) -> pd.Series:
    '''
    get_scores the prephormance of the model.
    data: could be the origonal or any processed version.
    metric: The get_scoresing metric itself.
    slicing: A var for grouping, by default the index (no grouping).
    '''
    scores = data.groupby(slicing).agg(lambda g : metric(data.loc[g.index, ACT], data.loc[g.index, PRED]))[ACT]
    scores = scores.rename(SCORE).reset_index()
    scores[METRIC] = metric.__name__
    return scores


def log_rmse(y_true, y_pred):
    y_true = log(y_true)
    y_pred = log(y_pred)
    df = pd.merge(y_true, y_pred, left_index = True, right_index = True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna()
    y_true = df[ACT]
    y_pred = df[PRED]
    if len(y_true) == 0 or len(y_pred) == 0:
        return np.nan
    return root_mean_squared_error(y_true, y_pred)


def compare(y_true, y_pred, y_alt, compare_with_alternative):
    '''
    Compare the performance of the model with an alternative.
    data: The original data.
    alternative: The alternative data.
    '''
    if compare_with_alternative:
        mutual_index = y_true.index.intersection(y_pred.index).intersection(y_alt.index)
        y_true = y_true.loc[mutual_index]
        y_pred = y_pred.loc[mutual_index]
        y_alt = y_alt.loc[mutual_index]

        return (np.abs(y_true - y_pred) < np.abs(y_true - y_alt)).mean()
    else:
        return np.nan


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--forecast", nargs=1, required=True)
    parser.add_argument("-t", "--truth",    nargs=1, required=True)
    parser.add_argument("-o", "--output",   nargs=1, required=True)
    parser.add_argument("-a", "--alternative",    nargs=1, required=False)
    args = parser.parse_args()


    def load_data(file_path: str, col_name: str) -> pd.DataFrame:
        data = pd.read_csv(file_path, index_col= 0)
        data = data.reindex(STD_CLUSTERS.index, fill_value=0)
        data.index.rename(IND, inplace=True)
        data.columns = [col_name]
        return data


    data = pd.merge(load_data(args.forecast[0], PRED),
                    load_data(args.truth[0], ACT),
                    left_index = True, right_index = True, how = 'outer').fillna(0)


    if args.alternative is None:
        compare_with_alternative = False
        alternative = pd.DataFrame(columns=[ACT, PRED])
    else:
        compare_with_alternative = True
        alternative = load_data(args.alternative[0], PRED)



    frames = []

    def comparison(y_true, y_pred):
        return compare(y_true, y_pred, alternative[PRED], compare_with_alternative)

    for metric in r2_score, root_mean_squared_error, log_rmse, comparison:
        data[SLC] = 'all'
        frames.append(get_scores(data, metric, SLC))

        data[SLC] = '2600/14'
        frames.append(get_scores(data.loc[['2600/14']], metric, SLC))

        data[SLC] = 'TOP 30'
        frames.append(get_scores(data.nlargest(30, ACT), metric, SLC))

        data[SLC] = data.index.str.split('/').str[0]
        frames += [get_scores(data[data.index.str.startswith('1500')], metric, SLC),
                   get_scores(data[data.index.str.startswith('1600')], metric, SLC)]

        data[SLC] = data.index.str.split('/').str[-1]
        frames.append(get_scores(data.groupby(SLC).filter(lambda g : len(g) > 1), metric, SLC))

        mutual_index = data.index.intersection(STD_CLUSTERS.index)
        data[SLC] = STD_CLUSTERS.loc[mutual_index]
        frames.append(get_scores(data, metric, SLC))

    results = pd.concat(frames, ignore_index = True)
    results = results.set_index(['group', 'metric']).unstack()
    results = results.droplevel(0, axis=1)
    results.to_csv(args.output[0], index = True)

if __name__ == '__main__':
    main()
