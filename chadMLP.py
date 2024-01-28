import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class MultiLayerPerceptron:
    def __init__(self,nFeats, hlnNeur, nClasses, h_act='relu', o_act='relu', alpha=0.01):
        self.nFeats = nFeats
        self.hlnNeur = hlnNeur
        self.nClasses = nClasses
        self.layers = [nFeats, hlnNeur, nClasses]
        self.layer_activations = [h_act,o_act]
        self.neurons = [np.zeros(i) for i in self.layers]
        self.biases = [np.random.uniform(-0.5, 0.5, size=i) for i in self.layers[1:]]
        self.weights = [np.random.uniform(-0.5, 0.5, size=(self.layers[i], self.layers[i-1])) for i in range(1, 3)]
        self.activations = [self.get_activation_fn(act) for act in self.layer_activations]
        self.alpha = alpha
        self.fitness = 0
        self.cost = 0
        self.delta_biases = [np.zeros_like(bias) for bias in self.biases]
        self.delta_weights = [np.zeros_like(weight) for weight in self.weights]
        self.delta_count = 0

    def sigmoid(self, x, derivative=False):
        if derivative:
            return x * (1 - x)
        return 1 / (1 + np.exp(-x))

    def tanh(self, x, derivative=False):
        if derivative:
            return 1 - x**2
        return np.tanh(x)

    def relu(self, x, derivative=False):
        if derivative:
            return np.where(x <= 0, 0, 1)
        return np.maximum(0, x)

    def leakyrelu(self, x, derivative=False):
        if derivative:
            return np.where(x <= 0, 0.01, 1)

    def get_activation_fn(self, activation):
        if activation == "sigmoid":
            return self.sigmoid
        elif activation == "tanh":
            return self.tanh
        elif activation == "relu":
            return self.relu
        elif activation == "leakyrelu":
            return self.leakyrelu
        else:
            return self.relu

    def activate(self, value, layer):
        return self.activations[layer](value)

    def activate_derivative(self, value, layer):
        return self.activations[layer](value, derivative=True)

    def feed_forward(self, inputs):
        self.neurons[0] = inputs
        for i in range(1, len(self.layers)):
            layer = i - 1
            for j in range(self.layers[i]):
                value = np.dot(self.weights[i-1][j], self.neurons[i-1]) + self.biases[i-1][j]
                self.neurons[i][j] = self.activate(value, layer)
        return self.neurons[-1]
    
    def backpropagate(self, inputs, expected):
        output = self.feed_forward(inputs)
        self.cost = np.sum((output - expected)**2) / 2

        gamma = [np.zeros(layer) for layer in self.layers]
        layer = len(self.layers) - 2

        gamma[-1] = (output - expected) * self.activate_derivative(output, layer)

        self.biases[-1] -= gamma[-1] * self.alpha

        for i in range(len(self.biases[-1])):
            self.weights[-1][i] -= gamma[-1][i] * self.neurons[-2] * self.alpha

        for i in range(len(self.layers) - 2, 0, -1):
            layer = i - 1
            gamma[i] = np.dot(gamma[i + 1], self.weights[i]) * self.activate_derivative(self.neurons[i], layer)

            self.biases[i - 1] -= gamma[i] * self.alpha

            for j in range(len(self.neurons[i - 1])):
                self.weights[i - 1][:, j] -= gamma[i] * self.neurons[i - 1][j] * self.alpha

    def fit(self, X, y, epochs=100):
        y = y.astype(int)
        for _ in range(epochs):
            for i in range(len(X)):
                self.backpropagate(X[i], np.eye(self.layers[2])[y[i]])
    
    def predict(self, X):
        predictions = []
        for i in range(len(X)):
            predictions.append(np.argmax(self.feed_forward(X[i])))
        return predictions
    
df = pd.read_csv('data/cleaned-eighthr-ozone-day-dataset.data', header=None)
iris = datasets.load_iris()
cancer = datasets.load_breast_cancer()

# X = iris.data
# y = iris.target

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# X = iris.data
# y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlp = MultiLayerPerceptron(nFeats=X.shape[1], hlnNeur=3, nClasses=np.unique(y).shape[0], h_act='sigmoid',o_act='sigmoid', alpha=0.01)
mlp.fit(X_train, y_train, epochs=100)
y_pred = mlp.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Predictions:", y_pred)
print("Actual:", y_test)