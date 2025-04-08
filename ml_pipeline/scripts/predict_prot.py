from .utils import get_pred_exp_data
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from pathlib import Path

def predict_protein(models, scaler, predictfile, output_path):
    x_exp, info = get_pred_exp_data(predictfile)
    X_pred = scaler.transform(x_exp)

    writer = pd.ExcelWriter(Path(output_path) / "protein_prioritization.xlsx", engine='xlsxwriter')

    for name, model in models:
        proba = model.predict_proba(X_pred)[:, 1]
        genes = [g for g, e in info]
        ensembl = [e for g, e in info]
        df = pd.DataFrame({'Ensembl': ensembl, 'Genes': genes, 'predict_proba': proba})
        df.sort_values(by='predict_proba', ascending=False, inplace=True)
        df.to_excel(writer, sheet_name=f'{name}_positive', index=False)

    writer._save()
