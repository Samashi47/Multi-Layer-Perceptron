import numpy as np

class MLPClassifier:
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
