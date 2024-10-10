import numpy as np


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature # quelle feature est utilisée pour diviser le noeud
        self.threshold = threshold # seuil pour la division
        self.left = left # sous arbre gauche
        self.right = right # sous arbre droit
        self.value = value # valeur de prediction de la feuille

    def is_leaf(self):
        return self.value is not None # si valeur non nulle alors est une feuille


class DecisionTreeRegressorScratch:
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        self.min_samples_split = min_samples_split # nb minimum de samples requis pour diviser le noeud
        self.max_depth = max_depth # profondeur max de l'arbre
        self.n_features = n_features # nombre de caractéristiques à etudier pour chaque division
        self.root = None # noeud racine de l'arbre

    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features) # nb de features à utiliser pour la division
        self.root = self._expand_tree(X, y)

    def _expand_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        # si condition arrete atteinte, noeud feuille avec valeur moyenne des cibles
        if depth >= self.max_depth or n_samples < self.min_samples_split:
            return Node(value=self._mean_value(y))

        # calcul de la meilleure division
        feat_idxs = np.random.choice(n_features, self.n_features, replace=False) # selection aleatoire d'un sous ensemble de caracteristiques
        best_feature, best_threshold = self._best_split(X, y, feat_idxs) # determine la meilleure feature et le meilleur seuil de division
        if best_feature is None:  # aucune division valide trouvée
            return Node(value=self._mean_value(y)) # on cree une feuille avec valeur moyenne des labels

        # division des donnees en sous ensembles gauche et droit
        left_idxs, right_idxs = self._split(X[:, best_feature], best_threshold)
        # construction des sous arbres gauche et droit
        left = self._expand_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._expand_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(best_feature, best_threshold, left, right)

    # trouve la meilleure division en terme de MSE parmi les features selectionnnees
    def _best_split(self, X, y, feat_idxs):
        best_mse = float('inf')
        split_idx, split_threshold = None, None
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx] # valeurs de la feature considérée
            thresholds = np.unique(X_column) # seuils uniques possibles pour la feature
            for t in thresholds:
                mse = self._calculate_mse(y, X_column, t) # calcul MSE pour chaque seuil

                if mse < best_mse: # si meilleure, on met à jour
                    best_mse = mse
                    split_idx = feat_idx
                    split_threshold = t

        return split_idx, split_threshold


    # calcule la MSE pour une division donnée
    def _calculate_mse(self, y, X_column, thresh):
        left_idxs, right_idxs = self._split(X_column, thresh) # divise les indices en fonction du seuil
        if len(left_idxs) == 0 or len(right_idxs) == 0: # si un groupe est vide, on renvoie infini
            return float('inf')

        # MSE des deux sous ensembles
        mse_left = np.mean((y[left_idxs] - np.mean(y[left_idxs])) ** 2)
        mse_right = np.mean((y[right_idxs] - np.mean(y[right_idxs])) ** 2)
        # MSE globale
        return (len(left_idxs) * mse_left + len(right_idxs) * mse_right) / len(y)

    # divise une colonne en deux en fonction du seuil
    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten() # liste des indices des valeurs qui respectent le seuil
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        # renvoie les indices des samples des deux sous ensembles
        return left_idxs, right_idxs

    # moyenne des cibles
    def _mean_value(self, y):
        return np.mean(y)

    # predit les cibles pour un ensemble X en traversant l'arbre
    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    # traverse l'arbre à partir d'un noeud donné jusqu'a une feuille
    def _traverse_tree(self, x, node):
        # renvoie la valeur si le noeud est une feuille
        if node.is_leaf():
            return node.value

        # sinon, compare valeur feature et seuil et parcours de l'arbre gauche ou droit
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)

        return self._traverse_tree(x, node.right)


    # POUR FAIRE LE CROSS VAL SCORE (voir la performance), j'ajoute ces deux fonctions sinon ne marche pas
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