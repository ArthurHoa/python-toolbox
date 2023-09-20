import numpy as np
import matplotlib.pyplot as py
from perceptron import Perceptron
from sklearn import preprocessing
from sklearn.datasets import load_breast_cancer


LEARNING_RATE = 0.0001
MAX_ITER = 100

def __main__():
    X, y = load_breast_cancer(return_X_y=True)

    X = preprocessing.StandardScaler().fit_transform(X)

    classifier = Perceptron(LEARNING_RATE, MAX_ITER)

    classifier.fit(X, y)

    print_error(classifier.get_errors(), classifier.get_accuracies())


def print_error(errors, accuracies):
    py.plot(range(errors.shape[0]), accuracies, label="Accuracy")
    py.plot(range(errors.shape[0]), errors, label="Loss")
    py.legend()
    py.show()


__main__()