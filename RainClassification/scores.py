import sklearn as sk
import autosklearn as asc
import pandas as pd

def predict_and_save(classifier, X_test, Y_test, verbose=False, file=None):
    if(verbose):
        print('Classification results...')

    predictions = classifier.predict(X_test)
    # print(classification_report_imbalanced(Y_test, predictions))
    cm = sk.metrics.confusion_matrix(Y_test, predictions)

    tn, fp, fn, tp = cm.ravel()
    pos = tp + fn + 0.0
    neg = fp + tn + 0.0

    acc = float(tp + tn)/float(pos + neg)
    prec = float(tp)/float(tp + fp)
    sens = float(tp)/float(tp + fn)
    spec = float(tn)/float((tn + fp))
    fscore = float(2*tp)/float(2*tp + fp + fn)

    kappa = sk.metrics.cohen_kappa_score(Y_test, predictions)

    if(verbose):
        print("Acc\t\tPrec\t\tSens\t\tSpec\t\tFscore\t\tKappa\t\tTP\tFN\tFP\tTN")
        print("{:2f}\t{:2f}\t{:2f}\t{:2f}\t{:2f}\t{:2f}\t{:d}\t{:d}\t{:d}\t{:d}".format(acc,prec,sens,spec,fscore,kappa,tp,fn,fp,tn))

    if(file != None):
        with open(file, "a") as arch:
            arch.write(classifier.show_models() + "\n\n\n")
            arch.write("{:2f}\t{:2f}\t{:2f}\t{:2f}\t{:2f}\t{:2f}\t{:d}\t{:d}\t{:d}\t{:d}\n".format(acc,prec,sens,spec,fscore,kappa,tp,fn,fp,tn))

def to_numeric(data):

    for i in range(0, 828):
        data.loc[i,('period')] = int(data['period'][i].split(":")[0])*60 + (int(data['period'][i].split(":")[1]))

    data['period'] = pd.to_numeric(data['period'])

    return data
