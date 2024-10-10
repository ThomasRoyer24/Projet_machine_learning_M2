import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model
import time

def compute_rmse(theta, X, y): # Calcul de la RMSE
    m = len(y)
    predictions = X @ theta
    return np.sqrt((1 / m) * np.sum((predictions - y) ** 2))

def linear_reg(n_iterations, X_normalized, y, learning_rate=0.1):
    m, n = X_normalized.shape  # m = nombre d'échantillons, n = nombre de caractéristiques
    theta = np.zeros(n)
    rmse_history = []

    for _ in range(n_iterations):
        predictions = X_normalized @ theta
        errors = predictions - y

        theta -= (learning_rate / m) * (X_normalized.T @ errors)

        rmse = compute_rmse(theta, X_normalized, y)
        rmse_history.append(rmse)

    print("Coefficients:", theta)
    return theta, rmse_history

def linear_reg_lasso(param, n_iterations, X_normalized, y,learning_rate=0.1):
    m, n = X_normalized.shape
    print("m:", m)
    print("n:", n)
    theta = np.zeros(n) # init des coefficients à 0
    rmse_history = []

    for _ in range(n_iterations):
        predictions = X_normalized @ theta
        errors = predictions - y

        theta -= (learning_rate / m) * (X_normalized.T @ errors)  # Gradient descent

        for j in range(len(theta)): # Lasso L1
            if j > 0:
                if theta[j] > 0:
                    theta[j] -= learning_rate * param
                elif theta[j] < 0:
                    theta[j] += learning_rate * param

        rmse = compute_rmse(theta, X_normalized, y) # calcul de la rmse
        rmse_history.append(rmse)

    print("Coefficients avec Lasso:", theta)
    return theta, rmse_history

def print2d(rmse_history, rmse_history_skit, filename="rmse_comparison.png"):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)  # 1
    plt.plot(rmse_history, color='blue')
    plt.xlabel("Itérations")
    plt.ylabel("RMSE")
    plt.title("RMSE from scratch")

    plt.subplot(1, 2, 2)  # 2
    plt.plot(rmse_history_skit, color='red')
    plt.xlabel("Itérations")
    plt.ylabel("RMSE")
    plt.title("RMSE scikit-learn")

    plt.tight_layout()

    #plt.savefig(filename)

    plt.show()
def compare_predictions(X, y, theta_lasso, theta_skit,filename='comparison.png'):
    predictions_scratch = X @ theta_lasso
    predictions_skit = X @ theta_skit
    plt.figure(figsize=(10, 6))

    #plt.scatter(range(len(y)), y, color='blue', label="Valeurs réelles", alpha=0.6)
    plt.plot(range(len(y)), predictions_scratch, color='red', label="Prédictions from scratch", linestyle='--',alpha=0.8)
    plt.plot(range(len(y)), predictions_skit, color='green', label="Prédictions scikit-learn", linestyle='-.',alpha=0.8)

    plt.xlabel("X")
    plt.ylabel("pred")
    plt.title("comparaison")
    plt.legend()
    plt.tight_layout()
    plt.show()
    #plt.savefig(filename)


# nettoyer les données
data = pd.read_csv('ozone_complet.txt', sep=';')
data = data.drop(columns=['maxO3v']) # enlever la colonne maxO3v
data = data.dropna()

y = data['maxO3'] # variable à prédire
X = data.drop(columns=['maxO3'])
X = X.to_numpy()
y = y.to_numpy()

n_iterations = 1000

print("X_normalized contains NaN:", np.isnan(X).any())
print("y contains NaN:", np.isnan(y).any())

X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_normalized = (X - X_mean) / X_std # normalisation

print("X_normalized contains NaN:", np.isnan(X_normalized).any()) # vérifier si la normalisation a introduit des NaN
print("y contains NaN:", np.isnan(y).any())


def lasso_skit(X_normalized, y):
    clf = linear_model.Lasso(alpha=0.1, warm_start=True, max_iter=1)
    rmse_history_skit = []
    # pour garder en mémoire la RMSE pendant le training
    for i in range(n_iterations):
        clf.fit(X_normalized, y)
        predictions = clf.predict(X_normalized)
        m = len(y)
        rmse = np.sqrt((1 / m) * np.sum((predictions - y) ** 2))
        rmse_history_skit.append(rmse)
    return clf.coef_, rmse_history_skit


theta_skit, rmse_history_skit = lasso_skit(X_normalized, y) # scikit-learn

start = time.time()
theta_lasso, rmse_history_lasso = linear_reg_lasso(0.1, n_iterations, X_normalized, y) # from scratch
end = time.time()
print("Time for scratch:", end - start)

start = time.time()
clf = linear_model.Lasso(alpha=0.1, max_iter=n_iterations) # scikit-learn avec temps
clf.fit(X_normalized, y)
end = time.time()
print("Time for scikit-learn:", end - start)

# afficher rmse
print2d(rmse_history_lasso, rmse_history_skit, "rmse_comparison.png")
# afficher predictions
compare_predictions(X_normalized, y, theta_lasso, theta_skit, filename="comparison.png")


### Optimisation des paramètres

from sklearn.model_selection import train_test_split

def evaluate_model_scratch(X_train, y_train, X_val, y_val, alpha, learning_rate, n_iterations):
    theta, rmse_history = linear_reg_lasso(alpha, n_iterations, X_train, y_train, learning_rate)
    rmse_val = compute_rmse(theta, X_val, y_val)
    return rmse_val
def evaluate_model_sklearn(X_train, y_train, X_val, y_val, alpha, n_iterations):
    clf = linear_model.Lasso(alpha=alpha, max_iter=n_iterations)
    clf.fit(X_train, y_train)
    predictions_val = clf.predict(X_val)
    rmse_val = np.sqrt(np.mean((predictions_val - y_val) ** 2))
    return rmse_val

def grid_search(X, y):
    # split dataset
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # different valeurs de paramètre possible
    alphas = [0.001, 0.01, 0.1, 0.2, 0.5]
    learning_rates = [0.001, 0.01, 0.1, 0.2, 0.5]
    n_iterations_list = [10, 100, 500, 1000, 5000]

    best_params_scratch = None
    best_rmse_scratch = float('inf')
    best_params_sklearn = None
    best_rmse_sklearn = float('inf')

    # from scratch
    for alpha in alphas:
        for learning_rate in learning_rates:
            for n_iterations in n_iterations_list:
                rmse_scratch = evaluate_model_scratch(X_train, y_train, X_val, y_val, alpha, learning_rate, n_iterations)
                if rmse_scratch < best_rmse_scratch:
                    best_rmse_scratch = rmse_scratch
                    best_params_scratch = {
                        'alpha': alpha,
                        'learning_rate': learning_rate,
                        'n_iterations': n_iterations
                    }

    #scikit-learn
    for alpha in alphas:
        for n_iterations in n_iterations_list:
            rmse_sklearn = evaluate_model_sklearn(X_train, y_train, X_val, y_val, alpha, n_iterations)
            if rmse_sklearn < best_rmse_sklearn:
                best_rmse_sklearn = rmse_sklearn
                best_params_sklearn = {
                    'alpha': alpha,
                    'n_iterations': n_iterations
                }
    return best_params_scratch, best_params_sklearn


#a, b = grid_search(X_normalized, y)
#print("Best parameters for scratch model:", a)
#print("Best parameters for scikit-learn model:", b)


