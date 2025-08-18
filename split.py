import pandas as pd


PATH = "NN-data-5"
VARS = ["article", "coin"]


for names, data in pd.read_csv(PATH + ".csv").groupby(VARS):
    data.drop(columns = VARS).to_csv(PATH + "-" + "-".join(map(str, names)) + ".csv")