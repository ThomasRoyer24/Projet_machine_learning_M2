import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model
import time
import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
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


def linear_reg_lasso(param, n_iterations, X_normalized, y, learning_rate=0.1):
    m, n = X_normalized.shape
    theta = np.zeros(n)
    rmse_history = []

    for _ in range(n_iterations):
        predictions = X_normalized @ theta
        errors = predictions - y

        theta -= (learning_rate / m) * (X_normalized.T @ errors)

        for j in range(1, len(theta)):
            if theta[j] > 0:
                theta[j] -= learning_rate * param
            elif theta[j] < 0:
                theta[j] += learning_rate * param

        # Calcul du RMSE pour chaque itération
        rmse = compute_rmse(theta, X_normalized, y)
        rmse_history.append(rmse)

    print("Coefficients avec Lasso (incluant l'intercept):", theta)
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
data = pd.read_csv('Hitters_train.csv', sep=',')
data = data.drop(columns=['Unnamed: 0'])
data = data.dropna()

from sklearn.preprocessing import OrdinalEncoder
enc = OrdinalEncoder()
data["League"] = enc.fit_transform(data[["League"]])
data["Division"] = enc.fit_transform(data[["Division"]])
data["NewLeague"] = enc.fit_transform(data[["NewLeague"]])

y = data['Salary']
X = data.drop(columns=['Salary'])
X = X.to_numpy()
y = y.to_numpy()

n_iterations = 20000

X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_normalized = (X - X_mean) / X_std
X_normalized = np.c_[np.ones(X_normalized.shape[0]), X_normalized]


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

start = time.time()
theta_lasso, rmse_history_lasso = linear_reg_lasso(0.1, n_iterations, X_normalized, y)
end = time.time()

print("Time for scratch:", end - start)
print("RMSE from scratch:", rmse_history_lasso[-1])


## Prédictions

data = pd.read_csv('Hitters_test.csv', sep=',')
data = data.drop(columns=['Unnamed: 0'])
data = data.dropna()

from sklearn.preprocessing import OrdinalEncoder
enc = OrdinalEncoder()
data["League"] = enc.fit_transform(data[["League"]])
data["Division"] = enc.fit_transform(data[["Division"]])
data["NewLeague"] = enc.fit_transform(data[["NewLeague"]])

y = data['Salary']
X = data.drop(columns=['Salary'])
X = X.to_numpy()
y = y.to_numpy()

X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_normalized = (X - X_mean) / X_std
X_normalized = np.c_[np.ones(X_normalized.shape[0]), X_normalized]
predictions_scratch = X_normalized @ theta_lasso

plt.figure(figsize=(10, 6))

plt.plot(range(len(y)), predictions_scratch, color='red', label="Prédictions from scratch", linestyle='--',alpha=0.8)
plt.plot(range(len(y)), y, color='green', label="y", linestyle='-.',alpha=0.8)

plt.xlabel("X")
plt.ylabel("pred salary")
plt.title("comparaison")
plt.legend()
plt.tight_layout()
plt.show()




