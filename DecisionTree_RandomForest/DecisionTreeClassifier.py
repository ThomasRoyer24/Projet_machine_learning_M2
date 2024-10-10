import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score


# Lire les commentaires sur l'arbre de décision pour la régression car cette classe (pour la classification) est très similaire
# Ajout de commentaires ici sur les différences

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None

    
class DecisionTreeClassifierScratch():
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        self.root = self._expand_tree(X, y)
    
    def _expand_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))
        # si condition arrete atteinte, noeud feuille avec cible en plus grande proportion
        # en effet, la feuille peut ne pas être pure si dataset immense, on a donc d'autres conditions d'arret
        # exemple : si 3 instances A,A,B alors retourne A
        if(depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)
        best_feature, best_threshold = self._best_split(X, y, feat_idxs)
        left_idxs, right_idxs = self._split(X[:, best_feature], best_threshold)
        left = self._expand_tree(X[left_idxs, :], y[left_idxs], depth+1)
        right = self._expand_tree(X[right_idxs, :], y[right_idxs], depth+1)
        return Node(best_feature, best_threshold, left, right)


    # trouve la meilleure division en terme de gain d'information parmi les features selectionnnees
    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_threshold = None, None
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            for t in thresholds:
                # calcule le gain d'information
                gain = self._information_gain(y, X_column, t)

                if(gain > best_gain):
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = t

        return split_idx, split_threshold

    # calcule le gain pour une division donnée
    def _information_gain(self, y, X_column, thresh):
        parent_entropy = self._entropy(y)
        left_idxs, right_idxs = self._split(X_column, thresh)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        # on calcule l'entropie des enfants
        n = len(y)
        n_left, n_right = len(left_idxs), len(right_idxs)
        left_entropy, right_entropy = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_left/n)*left_entropy+(n_right/n)*right_entropy

        # Gain d'information
        information_gain = parent_entropy-child_entropy
        return information_gain

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    # entropie d'un ensemble de label (pureté des samples)
    def _entropy(self, y):
        hist = np.bincount(y) # cree un tableau avec le nombre d'occurences de la valeur i à l'index i
        ps = hist/len(y)
        return -np.sum([p*np.log2(p) for p in ps if p > 0])

    # trouve label le plus frequent dans un set de samples (utilisé pour determiner la valeur d'une feuille)
    def _most_common_label(self, y):
         counter = Counter(y)
         return counter.most_common(1)[0][0]

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        
        return self._traverse_tree(x, node.right)


    # POUR FAIRE LE GRID SEARCH
    def score(self, X, y):
        predictions = self.predict(X)
        return accuracy_score(y, predictions)

    def get_params(self, deep=True):
        return {
            "min_samples_split": self.min_samples_split,
            "max_depth": self.max_depth,
            "n_features": self.n_features
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
