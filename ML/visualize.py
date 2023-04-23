import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay


def get_metrics(folder):
    # MODELS = ["KNN", "RF", "SVC", "MLP"]
    MODELS = ["SVC"]
    data = {model_name : joblib.load(os.path.join(folder, f"{model_name}.pth")) for model_name in MODELS}
    
    acc = np.array([
        [item["accuracy_score"] for item in model] for model in data.values()
    ])

    f1 = np.array([
        [item["f1_score"] for item in model] for model in data.values()
    ])

    recall = np.array([
        [item["recall_score"] for item in model] for model in data.values()
    ])
    
    metrics = list(zip(MODELS, np.around(acc.mean(axis=1) * 100, 1), np.around(f1.mean(axis=1) * 100, 1), np.around(recall.mean(axis=1) * 100, 1)))
    df = pd.DataFrame(metrics, columns=["model", "acc", "f1", "recall"])
    df.to_csv(os.path.join(folder, "metrics.csv"))
    
def get_cm(folder, plot=True):
    # MODELS = ["KNN", "RF", "SVC", "MLP"]
    MODELS = ["SVC"]
    CM = {}
    for model_name in MODELS:
        data = joblib.load(os.path.join(folder, f"{model_name}.pth"))
        cm = np.zeros(shape=(7, 7))
        for item in data:
            cm += item["confusion_matrix"]
        cm /= 5
        cm = cm.astype(np.int16)
        cmp = ConfusionMatrixDisplay(cm, display_labels=np.arange(1, 8))
        if plot:
            cmp.plot()
            plt.savefig(os.path.join(folder, f"{model_name}.jpg"))
            plt.clf()
        CM[model_name] = cm
    return CM