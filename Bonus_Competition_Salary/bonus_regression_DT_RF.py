from DecisionTreeRegressor import DecisionTreeRegressorScratch
from RandomForestRegressor import RandomForestRegressorScratch
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import time


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


train_path = r"C:\Users\alexa\OneDrive - yncréa\ISEN\M2\Machine learning\BONUS\Hitters_train.csv"
data_train = pd.read_csv(train_path, delimiter=',', quotechar='"')
data_train = data_train.drop(columns=['Unnamed: 0'])
data_train = data_train.dropna()
enc = OrdinalEncoder()
data_train["League"] = enc.fit_transform(data_train[["League"]])
data_train["Division"] = enc.fit_transform(data_train[["Division"]])
data_train["NewLeague"] = enc.fit_transform(data_train[["NewLeague"]])

X_train = np.array(data_train.drop("Salary", axis=1))
y_train = np.array(data_train["Salary"])

test_path = r"C:\Users\alexa\OneDrive - yncréa\ISEN\M2\Machine learning\BONUS\Hitters_test.csv"
data_test = pd.read_csv(test_path, delimiter=',', quotechar='"')
data_test = data_test.drop(columns=['Unnamed: 0'])
data_test = data_test.dropna()
enc = OrdinalEncoder()
data_test["League"] = enc.fit_transform(data_test[["League"]])
data_test["Division"] = enc.fit_transform(data_test[["Division"]])
data_test["NewLeague"] = enc.fit_transform(data_test[["NewLeague"]])

X_test = np.array(data_test.drop("Salary", axis=1))
y_test = np.array(data_test["Salary"])


scaler = MinMaxScaler()
X_train_standardized = scaler.fit_transform(X_train)
X_test_standardized = scaler.transform(X_test)

dtr_scratch = DecisionTreeRegressorScratch()
start = time.time()
evaluate_model(dtr_scratch, X_train_standardized, y_train, X_test_standardized, y_test)
end = time.time()
print(f"Temps d'exécution SCRATCH : {end - start} secondes\n")

dtr_sklearn = DecisionTreeRegressor()
start = time.time()
evaluate_model(dtr_sklearn, X_train_standardized, y_train, X_test_standardized, y_test)
end = time.time()
print(f"Temps d'exécution SKLEARN : {end - start} secondes\n")

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