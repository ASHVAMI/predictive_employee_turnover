from src.data_loader import load_data, preprocess_data
from src.train_model import train_logistic_regression, train_decision_tree

if __name__ == '__main__':
    # Load and preprocess data
    X, y = load_data('../data/employee_data.csv')
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    
    # Train models
    train_logistic_regression(X_train, y_train)
    train_decision_tree(X_train, y_train)
