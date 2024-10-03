import pandas as pd
import numpy as np
from Preprocess import load_and_preprocess_ozone

class RidgeRegression: 
    def __init__(self, penality): 
        self.penality = penality 

    def fit(self, X, y):
        samples_nb = X.shape[0]
        X_with_intercept = np.c_[np.ones((samples_nb, 1)), X]

        identity_matrix = np.identity(X_with_intercept.shape[1]) 
        identity_matrix[0, 0] = 0 # Do not regularise the intercept

        # Calculation of the matrices A and B for Ridge's solution
        A = X_with_intercept.T.dot(X_with_intercept) + self.penality * identity_matrix
        B = X_with_intercept.T.dot(y)
        
        # Display for debugging
        print('A matrix:\n', A)
        print('B vector:\n', B)

        self.thetas = np.linalg.solve(A, B) # Solve for the parameters
        return self

    def predict(self, X): 
        samples_nb = X.shape[0]
        X_with_intercept = np.c_[np.ones((samples_nb, 1)), X]
        predictions = X_with_intercept.dot(self.thetas) # Calculate predictions
        return predictions  

def runAndTestRidgeRegression():
    ozone_data = load_and_preprocess_ozone()

    n = int(0.8 * ozone_data.shape[0])  
    train = ozone_data.iloc[:n]
    test = ozone_data.iloc[n:]

    # Separation of characteristics and target
    X_train = train.drop('maxO3', axis=1).copy()
    y_train = train['maxO3']
    X_test = test.drop('maxO3', axis=1).copy()
    y_test = test['maxO3']

    # Creation and fitting of the Ridge regression model
    model = RidgeRegression(penality=1.0)
    model.fit(X_train.values, y_train.values)

    # Predictions on test data
    predictions = model.predict(X_test.values)
    print('Predictions:', predictions)
    print('True values:', y_test.values) 

# Execute the function to test the model
runAndTestRidgeRegression() 