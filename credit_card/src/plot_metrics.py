import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from evaluation import evaluate_model

def plot_confusion_matrix(cm, labels=None, title="Confusion Matrix"):
    disp=ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels
    )
    disp.plot(values_format="d")
    plt.title(title)
    return disp.figure_
