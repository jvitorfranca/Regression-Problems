import autosklearn.regression as asc
import sklearn as sk
import pandas as pd
import numpy as np
import requests, zipfile
import os
import pickle
from random import randint


try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO


def download_data():
    if 'train_FD004.txt' not in os.listdir('data'):
        print('Downloading Data...')
        # Download the data
        r = requests.get("https://ti.arc.nasa.gov/c/6/", stream=True)
        z = zipfile.ZipFile(StringIO.StringIO(r.content))
        z.extractall('data')
    else:
        print('Using previously downloaded data')

def load_data(data_path):
    operational_settings = ['operational_setting_{}'.format(i + 1) for i in range(3)]
    sensor_columns = ['sensor_measurement_{}'.format(i + 1) for i in range(26)]
    cols = ['engine_no', 'time_in_cycles'] + operational_settings + sensor_columns
    data = pd.read_csv(data_path, sep=' ', header=-1, names=cols)
    data = data.drop(cols[-5:], axis=1)
    data['index'] = data.index
    data.index = data['index']
    data['time'] = pd.date_range('1/1/2000', periods=data.shape[0], freq='600s')
    print('Loaded data with:\n{} Recordings\n{} Engines'.format(
        data.shape[0], len(data['engine_no'].unique())))

    return data

def new_labels(data, labels):
    ct_ids = []
    ct_times = []
    ct_labels = []
    data = data.copy()
    data['RUL'] = labels
    gb = data.groupby(['engine_no'])
    for engine_no_df in gb:
        instances = engine_no_df[1].shape[0]
        r = randint(5, instances - 1)
        ct_ids.append(engine_no_df[1].iloc[r,:]['engine_no'])
        ct_times.append(engine_no_df[1].iloc[r,:]['time'])
        ct_labels.append(engine_no_df[1].iloc[r,:]['RUL'])
    ct = pd.DataFrame({'engine_no': ct_ids,
                       'cutoff_time': ct_times,
                       'RUL': ct_labels})
    ct = ct[['engine_no', 'cutoff_time', 'RUL']]
    ct.index = ct['engine_no']
    ct.index = ct.index.rename('index')

    return ct

def make_cutoff_times(data):
    gb = data.groupby(['engine_no'])
    labels = []

    for engine_no_df in gb:
        instances = engine_no_df[1].shape[0]
        label = [instances - i - 1 for i in range(instances)]
        labels += label

    return new_labels(data, labels)

def classify(features, targets, time, name):
    X_train, X_test, y_train, y_test = \
        sk.model_selection.train_test_split(features, targets)

    classifier = asc.AutoSklearnRegressor(
        time_left_for_this_task=time+10,
        per_run_time_limit=time,
        initial_configurations_via_metalearning=0
    )

    classifier.fit(X_train.copy(), y_train.copy(), dataset_name='engines')

    pickle.dump(classifier, open('obj/' + name + '_obj.sav', 'wb'))

    predictions = classifier.predict(X_test)

    evs = sk.metrics.explained_variance_score(y_test, predictions)
    mae = sk.metrics.mean_absolute_error(y_test, predictions)
    mse = sk.metrics.mean_squared_error(y_test, predictions)
    msle = sk.metrics.mean_squared_log_error(y_test, predictions)
    mdae = sk.metrics.median_absolute_error(y_test, predictions)
    r2 = sk.metrics.r2_score(y_test, predictions)
    rmse = np.sqrt(sk.metrics.mean_squared_error(y_test, predictions))

    with open("results/" + name, 'a') as arch:
        arch.write(classifier.show_models() + "\n\n\n")
        arch.write("EVS\t\tMAE\t\tMSE\t\tMSLE\t\tMDAE\t\tR2\t\tRMSE\n")
        arch.write("{:2f}\t{:2f}\t{:2f}\t{:2f}\t{:2f}\t{:2f}\t{:2f}\n".format(evs,mae,mse,msle,mdae,r2,rmse))
