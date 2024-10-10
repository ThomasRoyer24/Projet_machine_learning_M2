import pandas as pd
import numpy as np
from RandomForestClassifier import RandomForestClassifierScratch
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, matthews_corrcoef, f1_score, accuracy_score
import time

def load_and_preprocess_CarSeats(path):

    # Charger la base de données depuis un fichier CSV
    data = pd.read_csv(path)

    data = data.drop(columns="Unnamed: 0")

    # # Conversion de la colonne ShelveLoc en variables numériques
    enc = OrdinalEncoder()
    data["ShelveLoc"] = enc.fit_transform(data[["ShelveLoc"]])

    # Transformation des colonnes en 0 et 1 (Yes -> 1, No -> 0)
    data['High'] = data['High'].map({'Yes': 1, 'No': 0})
    data['Urban'] = data['Urban'].map({'Yes': 1, 'No': 0})
    data['US'] = data['US'].map({'Yes': 1, 'No': 0})

    return data


path = r"C:\Users\alexa\OneDrive - yncréa\ISEN\M2\Machine learning\Data-20241001\Carseats.csv"
data = load_and_preprocess_CarSeats(path)

X = np.array(data.drop("High", axis=1))
y = np.array(data["High"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)


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


rf_sklearn = RandomForestClassifier(random_state=42)
rf_scratch = RandomForestClassifierScratch()


parameters = {
            'n_estimators': [10, 20, 30],
            'max_depth': [10, 50, 100, 200, 500],
            'min_samples_split': [2, 5, 7]
}


k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

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