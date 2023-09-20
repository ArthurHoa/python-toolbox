import numpy as np
import matplotlib.pyplot as py
from sklearn.metrics import accuracy_score

class NN():
    def __init__(self, learning_rate, max_iter):
        self._network_weights = []
        self._network_outputs = []
        self._learning_rate = learning_rate
        self._max_iter = max_iter
        self._fitted = False

    def add_layer(self, neurons_number):
        layer = np.ones(neurons_number + 1)
        self._network_outputs.append(layer)

    def add_final_layer(self, neurons_number):
        layer = np.ones(neurons_number)
        self._network_outputs.append(layer)

    def fit(self, X, y, X_test, y_test, min_error=0.01):

        # Size of the perceptron
        self._size = X.shape[0]

        self._X_test = X_test
        self._y_test = y_test

        # Init bias
        self._X = np.ones((X.shape[0], X.shape[1] + 1))
        self._X[:,1:] = X
        
        self._init_weights()

        # Init minimum error
        self.min_error = min_error

        # Save classes
        self._y = y

        # Train errors
        self._errors = []
        self._accuracies = []

        # Train perceptron
        print("Training Perceptron...")
        
        # Feed forward

        for i in range(self._max_iter):
            self._feed_forward()
            self._update_weights()

            if self._errors[-1] < min_error:
                break

        self._fitted = True
    
    def get_weights(self):
        if self._fitted:
            return self._network_weights
        
        print("Please first, fit the model with data")

    def get_errors(self):
        if self._fitted:
            return np.array(self._errors)
        
        print("Please first, fit the model with data")

    def get_accuracies(self):
        if self._fitted:
            return np.array(self._accuracies)
        
        print("Please first, fit the model with data")

    def get_accuracy(self):
        temp = self._sigmoid(np.matmul(self.X, self.weights))
        good_pred = ((temp > 0.5) == (self.y > 0.5)).sum()
        return good_pred / self.X.shape[0]

    def _init_weights(self):        
        size = self._X.shape[1]
        for layer in self._network_outputs:
            weigths = (np.random.rand(len(layer), size) - 0.5) / 10
            self._network_weights.append(weigths)
            size = len(layer)

    def _feed_forward(self):
        self._network_outputs[0] = self._sigmoid(np.matmul(self._X, self._network_weights[0].T))
        self._network_outputs[0][:,0] = 1
        
        for i in range(1, len(self._network_outputs)):
            self._network_outputs[i] = self._sigmoid(np.matmul(self._network_outputs[i-1], self._network_weights[i].T))

            if i != len(self._network_outputs) - 1:
                self._network_outputs[i][:,0] = 1

    def _predict(self, X):
        X_test = np.ones((X.shape[0], X.shape[1] + 1))
        X_test[:,1:] = X

        layer = self._sigmoid(np.matmul(X_test, self._network_weights[0].T))
        layer[:,0] = 1

        for i in range(1, len(self._network_outputs)):
            layer = self._sigmoid(np.matmul(layer, self._network_weights[i].T))

            if i != len(self._network_outputs) - 1:
                layer[:,0] = 1

        return layer

    def _update_weights(self):
        Delta_L = [self._network_outputs[-1][:,0] - self._y,  self._network_outputs[-1][:,1] - (1 - self._y)]
        
        # Get error
        self._errors.append(np.sum(np.power(Delta_L, 2)) / 2)
        # Get accuracies
        self._accuracies.append(accuracy_score(self._y_test, 1 - np.argmax(self._predict(self._X_test), axis=1)))

        Delta_L = np.multiply(np.array(Delta_L).T, np.multiply(self._network_outputs[-1], 1 - self._network_outputs[-1]))
        Delta_l_buffer = Delta_L

        self._network_weights[-1] = self._network_weights[-1] - (self._learning_rate / Delta_l_buffer.shape[0]) * np.dot(np.array(Delta_l_buffer).T, np.array(self._network_outputs[-2]))

        for i in reversed(range(len(self._network_weights))):
            if i > 1:
                Delta_l_buffer = np.multiply(np.matmul(Delta_l_buffer, self._network_weights[i]), np.multiply(self._network_outputs[i-1], 1 - self._network_outputs[i-1]))
                self._network_weights[i - 1] = self._network_weights[i - 1] - (self._learning_rate / Delta_l_buffer.shape[0]) * np.dot(np.array(Delta_l_buffer).T, np.array(self._network_outputs[i-2]))

            elif i == 1:
                Delta_l_buffer = np.multiply(np.matmul(Delta_l_buffer, self._network_weights[i]), np.multiply(self._network_outputs[i-1], 1 - self._network_outputs[i-1]))
                self._network_weights[i - 1] = self._network_weights[i - 1] - (self._learning_rate / Delta_l_buffer.shape[0]) * np.dot(np.array(Delta_l_buffer).T, np.array(self._X))

    def _sigmoid(self, X):
        return 1 / (1 + np.exp(-X))