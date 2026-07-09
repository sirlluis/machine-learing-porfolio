# configuración
from config import DATA_PATH, FIGURES_PATH

# datos
from data_loader import load_data
from data_cleaning import clean_data
from data_split import split_data

# preprocesado
from preprocessing import build_preprocessor

# modelos
from models import build_logistic_regression

# entrenamiento
from train import (
    train_model,
    build_pipeline
)

# evaluación
from evaluation import evaluate_model

# plotting
from plot_metrics import plot_confusion_matrix

# export plots
from export import save_plot

def main():
    # carga de datos
    df=load_data(DATA_PATH)
    
    # limpieza
    df=clean_data(df)
    
    # entrenamiento y split
    X_train, X_test, y_train, y_test=split_data(df)
    
    # preprocesado
    preprocessor=build_preprocessor()
    
    # construcción del modelo
    model=build_logistic_regression()
    
    # contrucción del pipeline
    pipeline=build_pipeline(
        preprocessor,
        model
    )
    
    # entrenamietno
    pipeline=train_model(
        pipeline,
        X_train,
        y_train
    )

    # evaluación
    results=evaluate_model(
        pipeline,
        X_test,
        y_test
    )
    
    # plot some metrics
    fig=plot_confusion_matrix(
        results["confusion_matrix"],
        labels=model.classes_
    )
    #saving plots
    save_plot(fig, FIGURES_PATH/"confusion_matrix.png")


    print("\nClassification Report\n")
    print(results["classification_report"])

    print("\nConfusion Matrix\n")
    print(results["confusion_matrix"])

    print(f"\nAccuracy : {results['accuracy']:.3f}")
    print(f"Precision: {results['precision']:.3f}")
    print(f"Recall   : {results['recall']:.3f}")
    print(f"F1 Score : {results['f1_score']:.3f}")

    
if __name__=="__main__":
    main()
    