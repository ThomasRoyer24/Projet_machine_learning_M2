from DecisionTreeClassifier import DecisionTreeClassifierScratch
import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score


class RandomForestClassifierScratch:

    def __init__(self, n_estimators=10, max_depth=10, min_samples_split=2, n_features=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = []


    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_estimators):
            tree = DecisionTreeClassifierScratch(max_depth=self.max_depth, min_samples_split=self.min_samples_split, n_features=self.n_features)
            X_sample, y_sample = self._samples(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def _samples(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]


    def _most_common_label(self, y):
         counter = Counter(y)
         return counter.most_common(1)[0][0]


    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        tree_predictions = np.swapaxes(predictions, 0, 1)
        return np.array([self._most_common_label(pred) for pred in tree_predictions])


    # POUR FAIRE LE GRID SEARCH
    def score(self, X, y):
        predictions = self.predict(X)
        return accuracy_score(y, predictions)


    def get_params(self, deep=True):
        return {
            "n_estimators": self.n_estimators,
            "min_samples_split": self.min_samples_split,
            "max_depth": self.max_depth,
            "n_features": self.n_features
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self