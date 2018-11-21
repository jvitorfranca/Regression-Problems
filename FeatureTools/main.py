import pandas as pd
import utilities as ut
import ft
import pickle

features = pickle.load(open("obj/feature_model", 'rb'))

targets = pickle.load(open("obj/targets_model", 'rb'))

runs = 30

for i in range(runs):
    ut.classify(features, targets, 60*60, str(i + 1)+'pred')
