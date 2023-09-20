import numpy as np
import matplotlib.pyplot as py

class Perceptron():
    def __init__(self, learning_rate, max_iter):
        self._learning_rate = learning_rate
        self._max_iter = max_iter
        self._fitted = False

    def fit(self, X, y, min_error=0.001):

        # Size of the perceptron
        self.size = X.shape[1] + 1

        # Init weights
        self.weights = np.random.random(self.size) / 10

        # Init bias
        self.X = np.ones((X.shape[0], self.size))
        self.X[:,1:] = X
        
        # Init minimum error
        self.min_error = min_error

        # Save classes
        self.y = y

        # Train errors
        error = self.get_error()
        self._errors = []
        self._errors.append(error)
        self._accuracies = []
        self._accuracies.append(self.get_accuracy())

        # Train perceptron
        print("Training Perceptron...")
        i = 0

        for i in range(self._max_iter):
            if error < self.min_error:
                break
            
            # Update weights
            self._update_weights()

            error = self.get_error()
            self._errors.append(error)
            self._accuracies.append(self.get_accuracy())

        if error < self.min_error:
            print("Sparation reached at iterations = ", i + 1)
        else:
            print("Total iterations reached :", i + 1)

        self._fitted = True
    
    def get_weights(self):
        if self._fitted:
            return self.weights
        
        print("Please first, fit the model with data")

    def get_errors(self):
        if self._fitted:
            return np.array(self._errors)
        
        print("Please first, fit the model with data")

    def get_accuracies(self):
        if self._fitted:
            return np.array(self._accuracies)
        
        print("Please first, fit the model with data")

    def get_error(self):
        return np.sum((self._sigmoid(np.matmul(self.X, self.weights)) - self.y)**2) / self.X.shape[0]

    def get_accuracy(self):
        temp = self._sigmoid(np.matmul(self.X, self.weights))
        good_pred = ((temp > 0.5) == (self.y > 0.5)).sum()
        return good_pred / self.X.shape[0]

    def _update_weights(self):

        # Compute gradient - Sigmoid function
        sig = self._sigmoid(np.matmul(self.X, self.weights))
        grad = np.matmul(self.X.T,(sig - self.y) * sig * (1 - sig))

        # Compute gradient - Linear function
        # grad = np.matmul(self.X.T,(np.matmul(self.X, self.weights) - self.y))

        # Update weights
        self.weights = self.weights - self._learning_rate * grad

    def _sigmoid(self, X):
        return 1 / (1 + np.exp(-X))