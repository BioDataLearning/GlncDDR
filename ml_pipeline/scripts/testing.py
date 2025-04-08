from .utils import get_test_data, metrics
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import pandas as pd

def test_models(models, scaler, testfile, output_path):
    x_exp, y_test = get_test_data(testfile)
    X_test = scaler.transform(x_exp)

    fileout = open(Path(output_path) / "test_metrics_log.txt", "w")
    metrics_list = []

    for name, model in models:
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        acc, sen, spe, _, mcc, f1, auc_val, _ = metrics(y_test, y_pred, y_proba)

        fileout.write(f"{name} Model:\n")
        fileout.write(f"Accuracy: {acc:.4f}\nSensitivity: {sen:.4f}\nSpecificity: {spe:.4f}\n")
        fileout.write(f"MCC: {mcc:.4f}\nF1: {f1:.4f}\nAUC: {auc_val:.4f}\n\n")

        metrics_list.append({
            "Model": name,
            "Accuracy": round(acc, 3),
            "Sensitivity": round(sen, 3),
            "Specificity": round(spe, 3),
            "MCC": round(mcc, 3),
            "F1 Score": round(f1, 3),
            "ROC-AUC": round(auc_val, 3)
        })

    fileout.close()

    # Save test metrics to CSV
    df_metrics = pd.DataFrame(metrics_list)
    df_metrics.to_csv(Path(output_path) / "test_model_metrics.csv", index=False)
