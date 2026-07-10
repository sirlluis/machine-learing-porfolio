# Configuration
from config import DATA_PATH, FIGURES_PATH

# Data
from data_loader import load_data
from data_cleaning import clean_data
from data_split import split_data

# Preprocessing
from preprocessing import build_preprocessor

# Models
from models import build_logistic_regression

# Training
from train import (
    train_model,
    build_pipeline
)

# Model evaluation
from evaluation import evaluate_model, print_results

# Plotting
from plot_metrics import plot_confusion_matrix

# Export plots
from export import save_plot


def main():
    # Loading the data
    df=load_data(DATA_PATH)
    
    # cleaning data
    df=clean_data(df)
    
    # train and split
    X_train, X_test, y_train, y_test=split_data(df)
    
    # preprocessing
    preprocessor=build_preprocessor()
    
    # building the model
    model=build_logistic_regression()
    
    # build the pipeline
    pipeline=build_pipeline(
        preprocessor,
        model
    )
    
    # training
    pipeline=train_model(
        pipeline,
        X_train,
        y_train
    )

    # evaluate the model
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

    # show results
    print_results(results)

    
if __name__=="__main__":
    main()
    