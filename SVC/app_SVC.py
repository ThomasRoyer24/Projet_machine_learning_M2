import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from SVC import SVC 

from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import roc_curve, auc

import time

def load_and_preprocess_CarSeats():

    # Charger la base de données depuis un fichier CSV
    data = pd.read_csv('Carseats.csv') 

    # Conversion de la colonne ShelveLoc en variables numériques (dummy variables)
    data = pd.get_dummies(data, columns=['ShelveLoc'],dtype=int)

    # Transformer les colonnes en 0 et 1 (Yes -> 1, No -> 0)
    data['High'] = data['High'].map({'Yes': 1, 'No': 0})
    data['Urban'] = data['Urban'].map({'Yes': 1, 'No': 0})
    data['US'] = data['US'].map({'Yes': 1, 'No': 0})

    data.drop(data.columns[data.columns.str.contains(
    'unnamed', case=False)], axis=1, inplace=True)

    return data
#prétraitement
data=load_and_preprocess_CarSeats()

labels=data['High'].copy()
data=data.drop('High',axis=1)
data=data.to_numpy()
X_train,X_test,y_train,y_test=train_test_split(data,labels,test_size=20,random_state=123)

#Optimisation paramètres
test_params = {
    'lr': [0.0001, 0.001, 0.01],
    'lambda_': [0.0001, 0.001, 0.01],
    'n_iterations': [1000]
}
best_accuracy = 0
best_params = None

for lr in test_params['lr']:
    for lambda_ in test_params['lambda_']:
        for n_iterations in test_params['n_iterations']:
            model = SVC(lr=lr, lambda_=lambda_, n_iterations=n_iterations)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = model.accuracy(y_test, y_pred)
            
            if accuracy > best_accuracy:
                best_svc=model
                best_accuracy = accuracy
                best_params = {'lr': lr, 'lambda_': lambda_, 'n_iterations': n_iterations}

print(best_params)


# Prédiction Entraînement + mesure de temps
start = time.time()
best_svc.fit(X_train,y_train)

print("Durée entrainement from scratch (s): "+str((time.time() - start)))

# Prédiction + mesure de temps
start = time.perf_counter_ns()
y_pred=best_svc.predict(X_test)
print("Durée d'éxecution from scratch (μs): "+str((time.perf_counter_ns() - start)/1000))

# Mesure accuracy
accuracy=best_svc.accuracy(y_test,y_pred)
print("Accuracy SVC from scratch: "+str(accuracy))

# Probabilités de la classe positive
y_scores = best_svc.predict_proba(X_test)[:, 1]  

# Calculer les valeurs de la courbe ROC
fpr, tpr, thresholds = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)  
print("AUC SVC from scratch: "+str(roc_auc))
# Tracer la courbe ROC
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='blue', label=f'Courbe ROC (AUC = {roc_auc:.2f})')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taux de Faux Positifs')
plt.ylabel('Taux de Vrais Positifs')
plt.title('Courbe ROC')
plt.legend(loc='lower right')
plt.grid()
plt.show()
