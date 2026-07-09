from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

def model_predictions(X_test, model):
    """
        Function to get predictions from the model
    """
    y_pred=model.predict(X_test)
    return y_pred

def evaluate_model(model, X_test, y_test):
    
    y_pred=model_predictions(X_test, model)
    
    results={
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred)
    }
    return results