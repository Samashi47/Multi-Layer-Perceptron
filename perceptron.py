import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.datasets import load_breast_cancer

class Perceptron:
    def __init__(self, epochs=100, alpha=0.1):
        self.epochs = epochs
        self.alpha = alpha
        self.Ws = None
        self.b = None
        
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
    
    def w_update(self, x, y):
        y_hat = self.sigmoid(np.dot(x , self.Ws) + self.b)
        self.Ws += self.alpha * np.dot(y - y_hat, x)
        self.b += self.alpha * (y - y_hat)
        
    def fit(self, X, y):
        nSamples, nFeats  = np.shape(X)
        self.Ws = np.zeros(nFeats)
        self.b = 0
        
        for _ in range(self.epochs):
            for i in range(nSamples):
                self.w_update(X[i], y[i])
    
    def predict(self, X):
        self.sigmoid(np.dot(X, self.Ws) + self.b)
        return [1 if i > 0.5 else 0 for i in self.sigmoid(np.dot(X, self.Ws) + self.b)]
    
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

perceptron = Perceptron(epochs=2000, alpha=0.01)
perceptron.fit(X_train, y_train)
y_pred = perceptron.predict(X_test)

print(f"Accuracy: {np.round(accuracy_score(y_test, y_pred),2)*100}%")
print(confusion_matrix(y_test, y_pred))
