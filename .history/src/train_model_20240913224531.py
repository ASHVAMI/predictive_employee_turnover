
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import pandas as pd

# Function to load and preprocess data
def load_and_preprocess_data():
    df = pd.read_csv('../data/employee_data.csv')
    df.drop('EmployeeID', axis=1, inplace=True)
    df = pd.get_dummies(df, drop_first=True)
    
    X = df.drop('Turnover', axis=1)
    y = df['Turnover']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test

# Function to train the Logistic Regression model
def train_logistic_regression():
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    
    # Train the model
    lr_model = LogisticRegression()
    lr_model.fit(X_train, y_train)
    
    # Save the model as logistic_regression_model.pkl
    joblib.dump(lr_model, '../models/logistic_regression_model.pkl')
    print("Logistic Regression model saved to ../models/logistic_regression_model.pkl")
    
    return lr_model

# Train the model
if __name__ == "__main__":
    train_logistic_regression()
