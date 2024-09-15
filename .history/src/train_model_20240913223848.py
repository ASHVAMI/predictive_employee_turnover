from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import joblib

def train_logistic_regression(X_train, y_train):
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    
    # Save model
    joblib.dump(lr, '../models/logistic_regression_model.pkl')
    print("Logistic Regression model saved.")

def train_decision_tree(X_train, y_train):
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    
    # Save model
    joblib.dump(dt, '../models/decision_tree_model.pkl')
    print("Decision Tree model saved.")
