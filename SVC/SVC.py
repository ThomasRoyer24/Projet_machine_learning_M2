import numpy as np


class SVC:
    def __init__(self, lr=0.001, lambda_=0.001, n_iterations=1000):
        self.w = None
        self.b = None
        self.lr = lr
        self.lambda_ = lambda_
        self.n_iterations = n_iterations

    def init_parameters(self, X):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0


    def gradient_descent(self, X, y):
        y_ = np.where(y <= 0, -1, 1)
        for i,x in enumerate(X):
            if  y_[i]* (np.dot(x, self.w) - self.b) >= 1:
                dw = 2 * self.lambda_ * self.w
                db = 0
            else:
                dw = 2 * self.lambda_ * self.w - np.dot(x, y_[i])
                db = y_[i]
            self.update_parameters(dw, db)
    
    def update_parameters(self, dw, db):
        self.w = self.w - self.lr * dw
        self.b = self.b - self.lr * db

    def fit(self, X, y):
        self.init_parameters(X)
        for iteration in range(self.n_iterations):
            self.gradient_descent(X, y)

            #if iteration % 100 == 0:
                #print(f"Iteration {iteration}: w = {self.w}, b = {self.b}")

    def predict(self, X):
        output = np.dot(X, self.w) - self.b
        label_signs = np.sign(output)
        y_pred = np.where(label_signs <= -1, 0, 1)
        return y_pred

    def accuracy(self, y_true, y_pred):
        total_samples = len(y_true)
        correct_predictions = np.sum(y_true == y_pred)
        return correct_predictions / total_samples
    
    def predict_proba(self, X):
        decision_values = np.dot(X, self.w) - self.b

        probabilities = 1 / (1 + np.exp(-decision_values))  
        return np.vstack((1 - probabilities, probabilities)).T
        

