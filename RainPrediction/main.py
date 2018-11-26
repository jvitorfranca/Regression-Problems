import autosklearn.regression as asc
import sklearn as sk
import pandas as pd
import utilities as ut


data = pd.read_csv("data/new_data.csv", index_col=0)

features = data.drop('12:Precipitacion', axis=1)

target = data['12:Precipitacion']

if __name__ == "__main__":

    runs = 5

    for i in range(0, runs):

        ut.classify(features, target, 60*5, str(i + 1)+'pred')
