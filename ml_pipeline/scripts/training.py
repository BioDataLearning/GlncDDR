import pandas as pd
from pathlib import Path
from .utils import get_training_data, ml_cv, plot_graph
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import joblib
import matplotlib.pyplot as plt

def train_models(scaler, trainingfile, output_path):
    x_exp, y_train = get_training_data(trainingfile)
    X_train = scaler.transform(x_exp)

    models = [
        ('LR', LogisticRegression(C=1.5, penalty='l2', solver='newton-cg', class_weight={1: 2.873}, max_iter=10000), '#004225'),
        ('RF', RandomForestClassifier(n_estimators=75, max_features='sqrt', max_depth=3, criterion='entropy', class_weight={1: 3.5}), '#F3AA60'),
        ('SVM', SVC(C=4, kernel='rbf', gamma=0.01, class_weight={1: 2.873}, probability=True), '#EF6262'),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    trained_models = []
    metrics_list = []

    fileout = open(Path(output_path) / "training_metrics_log.txt", "w")

    for name, model, color in models:
        cv_preds, cv_probs, cv_auc, cv_acc, cv_sen, cv_spe, cv_mcc, cv_f1 = ml_cv(
            name, model, X_train, y_train, 42, output_path, fileout)

        model.fit(X_train, y_train)
        trained_models.append((name, model))
        joblib.dump(model, Path(output_path) / f"{name}_model.pkl")

        plot_graph(cv_preds, cv_probs, name, color, ax1, ax2)

        metrics_list.append({
            "Model": name,
            "Accuracy": round(cv_acc, 3),
            "Sensitivity": round(cv_sen, 3),
            "Specificity": round(cv_spe, 3),
            "MCC": round(cv_mcc, 3),
            "F1 Score": round(cv_f1, 3),
            "ROC-AUC": round(cv_auc, 3)
        })

    fileout.close()
    plt.savefig(Path(output_path) / "combined_roc_pr.png", bbox_inches='tight', dpi=300)

    # Save metrics to CSV
    df_metrics = pd.DataFrame(metrics_list)
    df_metrics.to_csv(Path(output_path) / "training_model_metrics.csv", index=False)

    return trained_models

