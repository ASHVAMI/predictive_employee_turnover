from sklearn.tree import DecisionTreeClassifier
import joblib

def train_decision_tree(X_train, y_train):
    # Initialize the Decision Tree Classifier
    dt = DecisionTreeClassifier(random_state=42)
    
    # Train the model
    dt.fit(X_train, y_train)
    
    # Save the trained model to a file
    joblib.dump(dt, '../models/decision_tree_model.pkl')
    
    print("Decision Tree model saved as decision_tree_model.pkl.")
