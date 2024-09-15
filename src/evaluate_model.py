import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def evaluate_model(model_path, X_test, y_test):
    # Load the model
    model = joblib.load(model_path)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Print evaluation metrics
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
