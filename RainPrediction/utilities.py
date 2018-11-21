import pandas as pd
import numpy as np

def prune_data(data_path):

    data = pd.read_csv(data_path, sep=" ")

    month = [0] * 4137

    for i in range(0, 4137):
        month[i] = data['1:Date'][i].split("/")[1]

        data.loc[i,('1:Date')] = data['1:Date'][i].split("/")[0]

        data.loc[i,('2:Time')] = int(data['2:Time'][i].split(":")[0])*60 + (int(data['2:Time'][i].split(":")[1]))

    data['1:Date'] = pd.to_numeric(data['1:Date'])

    data['2:Time'] = pd.to_numeric(data['2:Time'])

    mon = np.array(month)

    data['25:Month'] = pd.Series(mon, index=data.index)

    data['index'] = data.index

    print(data.head())

    # data.to_csv("data/new_data.csv")

if __name__ == "__main__":

    prune_data("data/NEW-DATA.txt")
