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
        This function makes predictions using
        the training model from pipeline

        Parameters
        ----------
        list: X_test
            X_test from train_test_split function
        model : trained model from pipeline
        
        Returns
        -------
        list : y_pred
            Predictions from the model
        
    """
    y_pred=model.predict(X_test)
    return y_pred

def evaluate_model(model, X_test, y_test):
    """
        Computes a set of model metrics

        Parameters
        ---------
        model : trained model from pipeline
        X_test, y_test : iterable

        Returns
        -------
        dict : results
            A dictionary containing the set of computed metrics
    """
    
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

def print_results(results):
    """
    Just print results
    """
    print("\nClassification Report\n")
    print(results["classification_report"])

    print("\nConfusion Matrix\n")
    print(results["confusion_matrix"])

    print(f"\nAccuracy : {results['accuracy']:.3f}")
    print(f"Precision: {results['precision']:.3f}")
    print(f"Recall   : {results['recall']:.3f}")
    print(f"F1 Score : {results['f1_score']:.3f}")
