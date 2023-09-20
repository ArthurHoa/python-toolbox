import numpy as np
import matplotlib.pyplot as py
from neural_net import NN
from sklearn import preprocessing
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

LEARNING_RATE = 0.1
MAX_ITER = 10000

def __main__():
    X, y = load_breast_cancer(return_X_y=True)

    X = preprocessing.StandardScaler().fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    classifier = NN(LEARNING_RATE, MAX_ITER)
    classifier.add_layer(10)
    classifier.add_layer(10)
    classifier.add_final_layer(2)

    print(X.shape)
    classifier.fit(np.array(X_train), np.array(y_train), X_test, y_test)

    print_error(classifier.get_errors(), classifier.get_accuracies())


def print_error(errors, accuracies):
    fig, ax1 = py.subplots()

    ax2 = ax1.twinx()
    ax1.plot(range(len(errors)), errors, 'g-')
    ax2.plot(range(len(accuracies)), accuracies, 'b-')

    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('errors', color='g')
    ax2.set_ylabel('accuracies', color='b')

    py.show()


__main__()