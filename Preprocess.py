import pandas as pd
from sklearn.preprocessing import StandardScaler


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

    # Load the database from a txt file
    data = pd.read_csv('ozone_complet.txt', delimiter=';', quotechar=',')

    # Clean up the column names by removing the excess quotes
    data.columns = data.columns.str.replace('"', '', regex=False) 
    
    # Drop dispensables items
    data = data.drop(columns=['maxO3v']) #Drop the 'maxO3v' column 
    data.dropna(inplace=True) #Drop the columns where there is/are NA value(s)

    # Manual data normalisation
    features = data.drop(columns=['maxO3']).copy() # Assume ‘maxO3’ is the target
    normalized_features = (features - features.mean()) / features.std()

    # Create a DataFrame with the normalized data
    normalized_data = normalized_features.copy()
    normalized_data['maxO3'] = data['maxO3'].values # Add the target column

    return normalized_data


