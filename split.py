import pandas as pd


PATH = "../data/deep learning project data/EMFs"
VARS = ["coin"]


for names, data in pd.read_csv(PATH + ".csv").groupby(VARS):
    data.drop(columns = VARS).to_csv(PATH + "-" + "-".join(map(str, names)) + ".csv", index = False)