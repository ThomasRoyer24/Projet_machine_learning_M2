import pandas as pd
import numpy as np

class RidgeRegression: 
    def __init__(self, penality): 
        self.penality = penality # Regulation parameter 

    def fit(self, X, y):
        samples_nb = X.shape[0] # Number of samples
        X_with_intercept = np.c_[np.ones((samples_nb, 1)), X] # Add a column for the intercept 

        # Creation of the identity matrix 
        identity_matrix = np.identity(X_with_intercept.shape[1]) 
        identity_matrix[0, 0] = 0 # Non-application of interception penalty

        # Define and solve the equations 
        A = X_with_intercept.T.dot(X_with_intercept) + self.penality * identity_matrix  # A = X^TX + λI
        B = X_with_intercept.T.dot(y)  # B = X^Ty
        self.thetas = np.linalg.solve(A, B)  # Solve A * thetas = B

        return self

    def predict(self, X): 
        samples_nb = X.shape[0] # Number of samples
        X_with_intercept = np.c_[np.ones((samples_nb, 1)), X]  # Add a column for the intercept 

        # Prediction calculation 
        predictions = X_with_intercept.dot(self.thetas) 

        return predictions  
    

# Preprocess data and run Ridge Regression
def runAndTestRidgeRegression(data):

    #Preprocess the data 
    n = 0.8*ozone_data.shape[0] # 80% of the total number of samples for training so 20% for test 

    train = ozone_data.iloc[:n]
    test = ozone_data.iloc[n:] 

    X_train = train.drop("maxO3").copy()
    y_train = train["maxO3"]
    X_test = test.drop("maxO3").copy()
    y_test = test["maxO3"]

    # Initialize and fit the model 
    model = RidgeRegression(penality=1.0)
    model.fit(X_train.values, y_train.values)

    # Make predictions
    predictions = model.predict(X_test.values)
    print(predictions, y_test.values)  # Predictions and true values 

    # Test predictions 




# Run and test the ridge regression
ozone_data = pd.read_csv("ozone_complet.txt", delimiter=";") # Loading data 
runAndTestRidgeRegression(ozone_data) 