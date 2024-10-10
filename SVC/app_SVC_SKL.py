import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score,roc_curve,auc
import time
from sklearn.model_selection import GridSearchCV

def prepro_CarSeats():

    # Charger la base de données depuis un fichier CSV
    data = pd.read_csv('Data-20241001\Carseats.csv') 

    # Conversion de la colonne ShelveLoc en variables numériques (dummy variables)
    data = pd.get_dummies(data, columns=['ShelveLoc'],dtype=int)

    # Transformer les colonnes en 0 et 1 (Yes -> 1, No -> 0)
    data['High'] = data['High'].map({'Yes': 1, 'No': 0})
    data['Urban'] = data['Urban'].map({'Yes': 1, 'No': 0})
    data['US'] = data['US'].map({'Yes': 1, 'No': 0})

    data.drop(data.columns[data.columns.str.contains(
    'unnamed', case=False)], axis=1, inplace=True)

    labels=data['High'].copy()
    data=data.drop('High',axis=1)
    data=data.to_numpy()
    X_train,X_test,y_train,y_test=train_test_split(data,labels,test_size=20,random_state=123)

    return X_train,X_test,y_train,y_test


def prepro_Hitters():
    data_train=pd.read_csv('Data-20241001\Hitters_train.csv')
    data_test=pd.read_csv('Data-20241001\Hitters_test.csv')

    data_train.dropna()
    data_test.dropna()
    
    data_train['Division'] = data_train['Division'].map({'E': 1, 'W': 0})
    data_train['League'] = data_train['League'].map({'A': 1, 'N': 0})
    data_train['NewLeague'] = data_train['NewLeague'].map({'A': 1, 'N': 0})

    data_test['Division'] = data_test['Division'].map({'E': 1, 'W': 0})
    data_test['League'] = data_test['League'].map({'A': 1, 'N': 0})
    data_test['NewLeague'] = data_test['NewLeague'].map({'A': 1, 'N': 0})
   

    data_train.drop(data_train.columns[data_train.columns.str.contains(
    'unnamed', case=False)], axis=1, inplace=True)
    data_test.drop(data_test.columns[data_test.columns.str.contains(
    'unnamed', case=False)], axis=1, inplace=True)
    
    
    data_train['Salary'] = data_train['Salary'].apply(lambda x: 1 if x > 425 else 0)
    data_test['Salary'] = data_test['Salary'].apply(lambda x: 1 if x > 425 else 0)
    
    
    dfx_train=data_train.drop(columns='Salary')
    X_train=dfx_train.to_numpy()

    dfx_test=data_test.drop(columns='Salary')
    X_test=dfx_test.to_numpy()

    dfy_train=data_train['Salary'].copy()
    y_train=dfy_train.to_numpy()

    dfy_test=data_test['Salary'].copy()
    y_test=dfy_test.to_numpy()

    return X_train,X_test,y_train,y_test


X_train,X_test,y_train,y_test=prepro_CarSeats()


#comparaison avec sklearn
svc_sklearn=svm.SVC(kernel='linear',probability=True)

linear_params = {
    'C': [0.01, 0.1, 1, 10],
    'max_iter': [10000, 50000, 100000], 
    'tol': [1e-3, 1e-4, 1e-5]
}

svc_sklearn = GridSearchCV(svc_sklearn, linear_params, cv=5)
svc_sklearn.fit(X_train,y_train)
best_sklearn_svc = svc_sklearn.best_estimator_

print(svc_sklearn.best_estimator_)
start = time.time()
best_sklearn_svc.fit(X_train,y_train)

print("Durée entrainement sklearn (s): "+str((time.time() - start)))

start = time.perf_counter_ns()
y_pred=best_sklearn_svc.predict(X_test)
print("Durée d'éxecution sklearn (μs): "+str((time.perf_counter_ns() - start)/1000))

accuracy=accuracy_score(y_test,y_pred)
print("Accuracy SVC sklearn: "+str(accuracy))


y_scores = best_sklearn_svc.predict_proba(X_test)[:, 1]  

# Calculer les valeurs de la courbe ROC
fpr, tpr, thresholds = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)  
print("AUC SVC sklearn: "+str(roc_auc))
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