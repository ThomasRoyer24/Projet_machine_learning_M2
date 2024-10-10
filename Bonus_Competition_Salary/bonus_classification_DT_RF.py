from DecisionTreeClassifier import DecisionTreeClassifierScratch
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from RandomForestClassifier import RandomForestClassifierScratch
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, matthews_corrcoef, f1_score, accuracy_score
import time

train_path = r"C:\Users\alexa\OneDrive - yncréa\ISEN\M2\Machine learning\BONUS\Hitters_train.csv"
data_train = pd.read_csv(train_path, delimiter=',', quotechar='"')
data_train = data_train.drop(columns=['Unnamed: 0'])
data_train = data_train.dropna()
data_train['Salary'] = data_train['Salary'].apply(lambda x: 'High' if x > 425 else 'Low')
enc = OrdinalEncoder()
data_train["League"] = enc.fit_transform(data_train[["League"]])
data_train["Division"] = enc.fit_transform(data_train[["Division"]])
data_train["NewLeague"] = enc.fit_transform(data_train[["NewLeague"]])
data_train["Salary"] = enc.fit_transform(data_train[["Salary"]])
data_train['Salary'] = data_train['Salary'].astype('int64')

X_train = np.array(data_train.drop("Salary", axis=1))
y_train = np.array(data_train["Salary"])

test_path = r"C:\Users\alexa\OneDrive - yncréa\ISEN\M2\Machine learning\BONUS\Hitters_test.csv"
data_test = pd.read_csv(test_path, delimiter=',', quotechar='"')
data_test = data_test.drop(columns=['Unnamed: 0'])
data_test = data_test.dropna()
data_test['Salary'] = data_test['Salary'].apply(lambda x: 'High' if x > 425 else 'Low')
enc = OrdinalEncoder()
data_test["League"] = enc.fit_transform(data_test[["League"]])
data_test["Division"] = enc.fit_transform(data_test[["Division"]])
data_test["NewLeague"] = enc.fit_transform(data_test[["NewLeague"]])
data_test["Salary"] = enc.fit_transform(data_test[["Salary"]])
data_test['Salary'] = data_test['Salary'].astype('int64')

X_test = np.array(data_test.drop("Salary", axis=1))
y_test = np.array(data_test["Salary"])

def gridSearch(estimator, parameters, k_fold, x_train, y_train):
    grid_search = GridSearchCV(estimator, parameters, cv=k_fold, n_jobs=-1, verbose=3)
    grid_search.fit(x_train, y_train)
    return grid_search.best_params_, grid_search.best_score_, grid_search.best_estimator_


def printResults(best_params, best_score):
    print("Meilleurs paramètres trouvés : ", best_params)
    print("Précision moyenne sur l'ensemble de train : ", best_score)



def evaluate_model(model, parameters, k_fold, X_train, y_train, X_test, y_test):
    best_params, best_score, best_estimator = gridSearch(model, parameters, k_fold, X_train, y_train)
    printResults(best_params, best_score)
    predictions = best_estimator.predict(X_test)
    print("Résultats sur ensemble de test avec les meilleurs paramètres")
    print(classification_report(y_test, predictions))
    print("Matthews correlation coefficient : ", matthews_corrcoef(y_test, predictions))
    print("F1-score : ", f1_score(y_test, predictions))
    print("Précision : {:.4f}%".format(accuracy_score(y_test, predictions) * 100))


dtc_sklearn = DecisionTreeClassifier(random_state=42)
dtc_scratch = DecisionTreeClassifierScratch()
rf_sklearn = RandomForestClassifier(random_state=42)
rf_scratch = RandomForestClassifierScratch()

parameters = {
            'max_depth': [10, 50, 100, 200, 500],
            'min_samples_split': [2, 5, 7]
}

k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("SKLEARN")
start = time.time()
evaluate_model(dtc_sklearn, parameters, k_fold, X_train, y_train, X_test, y_test)
end = time.time()
print(f"Temps d'exécution SKLEARN : {end - start} secondes\n")

print("SCRATCH")
start = time.time()
evaluate_model(dtc_scratch, parameters, k_fold, X_train, y_train, X_test, y_test)
end = time.time()
print(f"Temps d'exécution SCRATCH : {end - start} secondes\n")

print("SKLEARN")
start_time = time.time()
evaluate_model(rf_sklearn, parameters, k_fold, X_train, y_train, X_test, y_test)
end_time = time.time()
print(f"Temps d'exécution SKLEARN : {end_time - start_time} secondes")
print("SCRATCH")
start_time = time.time()
evaluate_model(rf_scratch, parameters, k_fold, X_train, y_train, X_test, y_test)
end_time = time.time()
print(f"Temps d'exécution SCRATCH : {end_time - start_time} secondes")
