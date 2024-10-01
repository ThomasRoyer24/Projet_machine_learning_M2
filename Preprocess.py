import pandas as pd



# !!!!!!    Supprimer la premiere colonne vide -> "", au debut du fichier CSV    !!!!!!!    


def load_and_preprocess_CarSeats():

    # Charger la base de données depuis un fichier CSV
    data = pd.read_csv('Carseats.csv') 

    # Conversion de la colonne ShelveLoc en variables numériques (dummy variables)
    data = pd.get_dummies(data, columns=['ShelveLoc'])

    # Transformer les colonnes en 0 et 1 (Yes -> 1, No -> 0)
    data['High'] = data['High'].map({'Yes': 1, 'No': 0})
    data['Urban'] = data['Urban'].map({'Yes': 1, 'No': 0})
    data['US'] = data['US'].map({'Yes': 1, 'No': 0})

    return data

def load_and_preprocess_ozone():

    # Charger la base de données depuis un fichier txt
    data = pd.read_csv("ozone_complet.txt", delimiter=';', quotechar='"')
    data = data.drop(columns=['maxO3v'])
    data = data.dropna(subset=['maxO3'])

    return data


