import joblib
import numpy as np

def predict_turnover(model_path, input_data):
    # Load the model
    model = joblib.load(model_path)
    
    # Predict turnover
    prediction = model.predict([input_data])
    return prediction[0]

if __name__ == '__main__':
    # Example input data (based on your features)
    input_data = np.array([35, 10, 5000, 3, 1, 1, 0])  # Example employee data
    
    # Predict using logistic regression model
    result = predict_turnover('../models/logistic_regression_model.pkl', input_data)
    print("Logistic Regression prediction (0: No Turnover, 1: Turnover):", result)
    
    # Predict using decision tree model
    result = predict_turnover('../models/decision_tree_model.pkl', input_data)
    print("Decision Tree prediction (0: No Turnover, 1: Turnover):", result)
