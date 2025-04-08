import os
import numpy as np
import random
import pandas as pd
from sklearn.utils import shuffle
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, matthews_corrcoef,
    roc_curve, auc, precision_recall_curve
)
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

def set_global_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

def get_training_data(trainingfile):
    full_train = pd.read_csv(trainingfile)
    full_train = shuffle(full_train, random_state=42)
    x = full_train.drop(columns=['Genes', 'Gene_type']).reset_index(drop=True)
    y = full_train['Gene_type'].reset_index(drop=True)
    y_train = np.array(y).astype(int).reshape((-1, 1))
    x_exp = np.asarray(x)
    return x_exp, y_train

def get_test_data(testfile):
    full_test = pd.read_csv(testfile)
    full_test = shuffle(full_test, random_state=42)
    x = full_test.drop(columns=['Genes', 'Gene_type']).reset_index(drop=True)
    y = full_test['Gene_type'].reset_index(drop=True)
    y_test = np.asarray(y)
    x_exp = np.asarray(x)
    return x_exp, y_test

def get_pred_exp_data(predictfile):
    full_pred = pd.read_csv(predictfile)
    full_pred = shuffle(full_pred, random_state=42)
    x_exp = full_pred.drop(columns=['Genes', 'Ensembl']).reset_index(drop=True)
    x_exp = np.asarray(x_exp)
    info = [(row['Genes'], row['Ensembl']) for _, row in full_pred.iterrows()]
    return x_exp, info

def metrics(Y_true, predictions, pred_probs):
    accuracy = accuracy_score(Y_true, predictions)
    confusion = confusion_matrix(Y_true, predictions)
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]
    specificity = TN / float(TN + FP)
    sensitivity = TP / float(FN + TP)
    strength = (specificity + sensitivity) / 2
    mcc = matthews_corrcoef(Y_true, predictions)
    f1 = f1_score(Y_true, predictions)
    fpr, tpr, _ = roc_curve(Y_true, pred_probs)
    aucvalue = auc(fpr, tpr)
    return accuracy, sensitivity, specificity, strength, mcc, f1, aucvalue, confusion

def plot_graph(cv_preds, cv_probs, name, color, ax1, ax2):
    fpr, tpr, _ = roc_curve(np.asarray(cv_preds), np.asarray(cv_probs))
    roc_auc = auc(fpr, tpr)
    ax1.plot(fpr, tpr, color=color, label=name + ' (AUC = %0.3f)' % roc_auc, linewidth=2)
    ax1.legend(loc='lower right')

    precision, recall, _ = precision_recall_curve(np.asarray(cv_preds), np.asarray(cv_probs))
    pr_auc = auc(recall, precision)
    ax2.plot(recall, precision, color=color, label=name + ' (AUC = %0.3f)' % pr_auc, linewidth=2)
    ax2.legend(loc='lower right')

def ml_cv(name, model, X_train, Y_train, random_state, file_path, fileout):
    seed = random_state
    cv_acc, cv_spe, cv_sen, cv_str, cv_mcc, cv_f1, cv_auc = [], [], [], [], [], [], []
    cv_preds, cv_probs = [], []

    kf = KFold(n_splits=10, shuffle=True, random_state=seed)

    for train_index, test_index in kf.split(X_train):
        X_cv_train, X_cv_test = X_train[train_index], X_train[test_index]
        Y_cv_train, Y_cv_test = Y_train[train_index], Y_train[test_index]

        model.fit(X_cv_train, Y_cv_train)

        predictions = model.predict(X_cv_test)
        pred_probs = model.predict_proba(X_cv_test)[:, 1]

        acc, sen, spe, strg, mcc, f1, auc_val, cm = metrics(Y_cv_test, predictions, pred_probs)

        cv_preds.extend(Y_cv_test.flatten())
        cv_probs.extend(pred_probs)

        cv_acc.append(acc)
        cv_sen.append(sen)
        cv_spe.append(spe)
        cv_str.append(strg)
        cv_mcc.append(mcc)
        cv_f1.append(f1)
        cv_auc.append(auc_val)

    fileout.write(f"Model: {name}\n")
    fileout.write(f"Accuracy_mean: {np.mean(cv_acc):.4f}\n")
    fileout.write(f"Sensitivity_mean: {np.mean(cv_sen):.4f}\n")
    fileout.write(f"Specificity_mean: {np.mean(cv_spe):.4f}\n")
    fileout.write(f"Strength_mean: {np.mean(cv_str):.4f}\n")
    fileout.write(f"MCC_mean: {np.mean(cv_mcc):.4f}\n")
    fileout.write(f"F1_mean: {np.mean(cv_f1):.4f}\n")
    fileout.write(f"AUC_mean: {np.mean(cv_auc):.4f}\n")
    fileout.write(f"Confusion Matrix (last fold):\n{cm}\n\n")

    return cv_preds, cv_probs, np.mean(cv_auc), np.mean(cv_acc), np.mean(cv_sen), np.mean(cv_spe), np.mean(cv_mcc), np.mean(cv_f1)