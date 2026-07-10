from sklearn.pipeline import Pipeline

def build_pipeline(preprocessor, model):
    """
        Build a pipeline to chain transformers
        and estimators to simplify model training

        Parameters
        ----------
        preprocessor : ColumnTransformer()
        
        model : estimator

        Returns
        -------
        self : object
    """
    pipeline=Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model)

        ]
    )
    return pipeline

def train_model(pipeline, X_train, y_train):
    """
        Fit all the transformers one after the other and sequentially transform the data. Finally, fit the transformed data using the final estimator.

        Parameters
        ----------
        pipeline : pipeline without training

        X_train, y_train : iterable training data

        Returns
        -------
        self : object
            Pipeline with fitted steps

    """
    pipeline.fit(X_train, y_train)
    return pipeline