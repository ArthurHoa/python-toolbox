import numpy as np
import matplotlib.pyplot as py
from neural_net import NN
from keras.datasets import mnist

LEARNING_RATE = 0.5
MAX_ITER = 500

def __main__():
    (train_X, train_y), (test_X, test_y) = mnist.load_data()

    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

    indexes = np.where((train_y == 0) | (train_y == 1))[0]
    train_X = train_X[indexes]
    train_y = train_y[indexes]

    indexes = np.where((test_y == 0) | (test_y == 1))[0]
    test_X = train_X[indexes]
    test_y = train_y[indexes]

    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

    train_X = train_X.reshape((train_X.shape[0], train_X.shape[1] * train_X.shape[2]))
    test_X = test_X.reshape((test_X.shape[0], test_X.shape[1] * test_X.shape[2]))

    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

    classifier = NN(LEARNING_RATE, MAX_ITER)
    classifier.add_layer(10)
    classifier.add_layer(10)
    classifier.add_final_layer(2)

    classifier.fit(np.array(train_X), np.array(train_y), test_X, test_y)

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