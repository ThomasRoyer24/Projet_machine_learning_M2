from RandomForestRegressor import RandomForestRegressorScratch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import time


def load_and_preprocess_ozone(path):

    # Charger la base de données depuis un fichier txt
    data = pd.read_csv(path, delimiter=';', quotechar='"')
    data = data.drop(columns=['maxO3v'])
    data = data.dropna()

    return data


def evaluate_model(model, X_train, y_train, X_test, y_test):
    tab = cross_val_score(model, X_train, y_train, scoring="neg_mean_squared_error", cv=10)
    tab = np.sqrt(tab * -1)
    print("Valeurs RMSE du tableau :", tab)
    print("Moyenne :", np.mean(tab))
    print("Ecart-type :", np.std(tab))
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print("Valeur RMSE finale sur le test set : ", np.sqrt(mean_squared_error(y_test, predictions)))
    print("Coefficient de determination sur le test set : %.2f" % r2_score(y_test, predictions))


path = r"C:\Users\alexa\OneDrive - yncréa\ISEN\M2\Machine learning\Data-20241001\ozone_complet.txt"
data = load_and_preprocess_ozone(path)
print(data.info(verbose=True))



X = np.array(data.drop("maxO3", axis=1))

# for column in X.columns:
#     plt.figure(figsize=(10, 4))
#     plt.hist(X[column], bins=30, edgecolor='k')
#     plt.title(f'Histogramme de {column}')
#     plt.xlabel(column)
#     plt.ylabel('Fréquence')
#     plt.show()


y = np.array(data["maxO3"])


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)

scaler = MinMaxScaler()
X_train_standardized = scaler.fit_transform(X_train)
X_test_standardized = scaler.transform(X_test)

rfr_scratch = RandomForestRegressorScratch()
start_time = time.time()
evaluate_model(rfr_scratch, X_train_standardized, y_train, X_test_standardized, y_test)
end_time = time.time()
print(f"Temps d'exécution SCRATCH : {end_time - start_time} secondes\n")

rfr_sklearn = RandomForestRegressor()
start_time = time.time()
evaluate_model(rfr_sklearn, X_train_standardized, y_train, X_test_standardized, y_test)
end_time = time.time()
print(f"Temps d'exécution SKLEARN : {end_time - start_time} secondes")