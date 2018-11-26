import autosklearn.regression as asc
import sklearn as sk
import pandas as pd
import numpy as np
import pickle

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

    data.to_csv("data/new_data.csv")

def mean_absolute_percentage_error(y_true, y_pred):

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def classify(features, targets, time, name):

    X_train, X_test, y_train, y_test = \
        sk.model_selection.train_test_split(features, targets, shuffle=False)

    classifier = asc.AutoSklearnRegressor(
        time_left_for_this_task=time+10,
        per_run_time_limit=time,
        initial_configurations_via_metalearning=0,
        resampling_strategy='cv',
        resampling_strategy_arguments={'folds': 5},
    )

    classifier.fit(X_train.copy(), y_train.copy(), dataset_name='engines')

    classifier.refit(X_train.copy(), y_train.copy())

    pickle.dump(classifier, open('obj/' + name + '_obj.sav', 'wb'))

    predictions = classifier.predict(X_test)

    evs = sk.metrics.explained_variance_score(y_test, predictions)
    mae = sk.metrics.mean_absolute_error(y_test, predictions)
    mse = sk.metrics.mean_squared_error(y_test, predictions)
    mdae = sk.metrics.median_absolute_error(y_test, predictions)
    r2 = sk.metrics.r2_score(y_test, predictions)
    rmse = np.sqrt(sk.metrics.mean_squared_error(y_test, predictions))

    with open("results/" + name, 'a') as arch:
        arch.write(classifier.show_models() + "\n\n\n")
        arch.write("EVS\t\tMAE\t\tMSE\t\tMDAE\t\tR2\t\tRMSE\n")
        arch.write("{:2f}\t{:2f}\t{:2f}\t{:2f}\t{:2f}\t{:2f}\n".format(evs,mae,mse,mdae,r2,rmse))

if __name__ == "__main__":

    prune_data("data/NEW-DATA.txt")
