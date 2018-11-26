import autosklearn.classification as asc
import autosklearn as asc_t
import sklearn as sk
import pandas as pd
import numpy as np
import scores
import pickle

def main():

    data = pd.read_csv("data/base_Intervalo.csv")

    data = scores.to_numeric(data)

    runs = 5

    time = 60*5

    for i in range(0, runs):

        X_train, X_test, Y_train, Y_test = sk.model_selection.train_test_split(data.drop('rain', axis=1), data['rain'], test_size=0.3)

        # Choosing kappa as the metric
        kappas = asc_t.metrics.make_scorer('kappa', sk.metrics.cohen_kappa_score)

        automl = asc.AutoSklearnClassifier(
                    time_left_for_this_task=time+10,
                    per_run_time_limit=time,
                    initial_configurations_via_metalearning=0
                    )

        automl.fit(X_train, Y_train, metric=kappas)

        scores.predict_and_save(automl, X_test, Y_test, verbose=True, file=str(i+1)+"prediction.txt")

        # Save the model into binary code
        filename = str(i + 1)+'model.sav'
        pickle.dump(automl, open(filename, 'wb'))

if __name__ == "__main__":
    main()
