import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    df = pd.read_csv(file_path)
    
    # Drop irrelevant columns
    df.drop('EmployeeID', axis=1, inplace=True)
    
    # One-hot encode categorical variables
    df = pd.get_dummies(df, drop_first=True)
    
    # Split data into features and target
    X = df.drop('Turnover', axis=1)
    y = df['Turnover']
    
    return X, y

def preprocess_data(X, y):
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test
