import pandas as pd

data = pd.read_csv("data/new_data.csv", index_col=0)

print(data.head())
