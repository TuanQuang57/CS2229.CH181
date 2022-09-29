import os
from category_encoders.leave_one_out import LeaveOneOutEncoder
from catboost.utils import read_cd
import numpy as np
from catboost import Pool

def load_regression_dataset(name):
    if name == "YearPredictionMSD":
        # https://archive.ics.uci.edu/ml/datasets/YearPredictionMSD
        data = np.loadtxt("datasets/" + name + ".txt", delimiter=",")
        n_splits = 1
        index_features = [i for i in range(1, 91)]
        index_target = 0
    else:
        # repository with all UCI datasets
        url = "https://raw.githubusercontent.com/yaringal/DropoutUncertaintyExps/master/UCI_Datasets/" + name + "/data/"
        data = np.loadtxt(url + "data.txt")
        n_splits = int(np.loadtxt(url + "n_splits.txt"))
        index_features = [int(i) for i in np.loadtxt(url + "index_features.txt")]
        index_target = int(np.loadtxt(url + "index_target.txt"))

    X = data[:, index_features]  # features
    y = data[:, index_target]  # target

    # prepare data for all train/test splits
    index_train = []
    index_test = []
    for i in range(n_splits):
        if name == "YearPredictionMSD":
        # default split for this dataset, see https://archive.ics.uci.edu/ml/datasets/YearPredictionMSD
            index_train.append([i for i in range(463715)])
            index_test.append([i for i in range(463715, 515345)])
        else:
            index_train.append([int(i) for i in np.loadtxt(url + "index_train_" + str(i) + ".txt")])
            index_test.append([int(i) for i in np.loadtxt(url + "index_test_" + str(i) + ".txt")])

    return X, y, index_train, index_test, n_splits


def make_train_val_test(X, y, index_train, index_test, fold):
    # train_all consists of all train instances
    X_train_all = X[index_train[fold], :]
    y_train_all = y[index_train[fold]]

    X_test = X[index_test[fold], :]
    y_test = y[index_test[fold]]

    # for parameter tuning we use 20% of train dataset for validation
    num_training_examples = int(0.8 * X_train_all.shape[0])
    X_train = X_train_all[0:num_training_examples, :]
    y_train = y_train_all[0:num_training_examples]
    X_validation = X_train_all[num_training_examples:, :]
    y_validation = y_train_all[num_training_examples:]

    return X_train_all, y_train_all, X_train, y_train, X_validation, y_validation, X_test, y_test


