import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import math

DATASET = ["IRIS","WINE","BREAST","ECOLI","BALANCE",
           "GLASS","PIMA","POST", "BANANA", "PARKINSON"]

RANDOM_SEEDS = [1679, 2528, 1200, 236, 28, 18, 0, 1, 6, 4491]

def __main__():

    for i in range(len(DATASET)):

        X, y = load_data(DATASET[i])

        acc, fscore, errors = start(X, y, RANDOM_SEEDS[i])

        acc = np.array(acc) * 100
        fscore = np.array(fscore) * 100

        print("\n", DATASET[i])
        print("Accuracies: %.2f %.2f" % (acc[0], acc[1]))
        print("F1-scores: %.2f %.2f" % (fscore[0], fscore[1]))
        print("Errors: %d %d" % (errors[0], errors[1]))


def start(X, y, seed):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=seed)

        classes = np.array(list(set(y)))
        nb_classes = classes.shape[0]
        avg = 'binary'
        if nb_classes > 2:
            avg = 'weighted'

        model1 = KNeighborsClassifier(n_neighbors=10)
        model1.fit(X_train, y_train)

        model2 = RandomForestClassifier(n_estimators=10, random_state=seed+1)
        model2.fit(X_train, y_train)

        acc1 = model1.score(X_test, y_test)
        acc2 = model2.score(X_test, y_test)

        fscore1 = f1_score(y_test, model1.predict(X_test), average=avg)
        fscore2 = f1_score(y_test, model2.predict(X_test), average=avg)

        error1 = np.where(model1.predict(X_test) != y_test)[0].shape[0]
        error2 = np.where(model2.predict(X_test) != y_test)[0].shape[0]

        return [acc1, acc2], [fscore1, fscore2], [error1, error2]

def load_data(dataset):
    if dataset == "IRIS":
        X = np.load("datasets/iris/X.npy").astype(float)
        y = np.load("datasets/iris/y.npy").astype(str)
        y = preprocessing.LabelEncoder().fit(y).transform(y)
        y = np.array(y)
    elif dataset == "WINE":
        X = np.load("datasets/wine/X.npy").astype(float)
        y = np.load("datasets/wine/y.npy").astype(str)
        y = preprocessing.LabelEncoder().fit(y).transform(y)
        y = np.array(y)
    elif dataset == "BREAST":
        X = np.load("datasets/breast/X.npy").astype(float)
        y = np.load("datasets/breast/y.npy").astype(str)
        y = preprocessing.LabelEncoder().fit(y).transform(y)
        y = np.array(y)
    elif dataset == 'ECOLI':
        X = np.genfromtxt('datasets/ecoli/ecoli.data',delimiter=',', usecols = [1,2,3,4,5,6,7])
        y = np.genfromtxt('datasets/ecoli/ecoli.data',delimiter=',', usecols = [8], dtype=str)
        y = preprocessing.LabelEncoder().fit(y).transform(y)
        y = np.array(y)
    elif dataset == 'BALANCE':
        X = np.genfromtxt('datasets/balance/balance.data',delimiter=',', usecols = [1,2,3,4])
        y = np.genfromtxt('datasets/balance/balance.data',delimiter=',', usecols = [0], dtype=str)
        y = preprocessing.LabelEncoder().fit(y).transform(y)
        y = np.array(y)
    elif dataset == 'GLASS':
        X = np.genfromtxt('datasets/glass/glass.data',delimiter=',', usecols = [1,2,3,4,5,6,7,8,9])
        y = np.genfromtxt('datasets/glass/glass.data',delimiter=',', usecols = [10])
        y = preprocessing.LabelEncoder().fit(y).transform(y)
        y = np.array(y)
    elif dataset == "PIMA":
        X = np.load("datasets/pima/X.npy").astype(float)
        y = np.load("datasets/pima/y.npy").astype(str)
        y = preprocessing.LabelEncoder().fit(y).transform(y)
        y = np.array(y)
    elif dataset == 'POST':
        X = np.zeros((86, 8))
        x_temp = np.genfromtxt('datasets/post-operative/post-operative.data',delimiter=',', usecols = [0, 1, 2, 3, 4, 5, 6], dtype=str)
        x_temp = np.where((x_temp == 'high') | (x_temp == 'good') | (x_temp == 'stable'), 2, x_temp)
        x_temp = np.where((x_temp == 'mid') | (x_temp == 'fair') | (x_temp == 'mod-stable'), 1, x_temp)
        x_temp = np.where((x_temp == 'low') | (x_temp == 'poor') | (x_temp == 'unstable'), 0, x_temp)
        x_temp = np.where((x_temp == 'excellent'), 3, x_temp)
        X[:,0:7] = x_temp
        X[:,7] = np.genfromtxt('datasets/post-operative/post-operative.data',delimiter=',', usecols = [7])
        y = np.genfromtxt('datasets/post-operative/post-operative.data',delimiter=',', usecols = [8], dtype=str)
        y = preprocessing.LabelEncoder().fit(y).transform(y)
        y = np.array(y)
    elif dataset == 'BANANA':
        X = np.load("datasets/banana/X.npy").astype(float)
        y = np.load("datasets/banana/y.npy").astype(str)
        y = preprocessing.LabelEncoder().fit(y).transform(y)
        y = np.array(y)
    elif dataset == "PARKINSON":
        X = np.genfromtxt('datasets/parkinson/parkinsons.data',delimiter=',', usecols = [i for i in range(1, 17)] + [i for i in range(18, 24)])
        y = np.genfromtxt('datasets/parkinson/parkinsons.data',delimiter=',', usecols = [17], dtype=str)
        y = preprocessing.LabelEncoder().fit(y).transform(y)
        y = np.array(y)

    # Scale data
    X = preprocessing.StandardScaler().fit_transform(X)

    return X, y

__main__()