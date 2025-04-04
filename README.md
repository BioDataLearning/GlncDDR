# GlncDDR: Predicting DNA Damage Response-Associated lncRNAs using cancer data and Graph-Based Learning

## Overview

**GlncDDR** is a machine learning-based framework developed to predict long non-coding RNAs (lncRNAs) associated with the DNA Damage Response (DDR) using transcriptomic data from The Cancer Genome Atlas (TCGA). It utilizes graph-based node embeddings to enhance prediction accuracy.

This repository contains code, data preprocessing scripts, and results from the GlncDDR study.

## Key Features

- Integrates **node2vec embeddings** to reduce high-dimensional RNA-seq data while preserving gene co-expression network structure.
- Trains classical ML models (**LR**, **SVM**, **RF**, and **XGBoost**) on DDR/non-DDR genes.
- Predicts **1,437 candidate DDR-associated lncRNAs** with ROC-AUC up to **0.997** on training and **0.91–0.95** on independent test datasets.
- Identifies known lncRNAs like **TODRA (RAD51-AS1)** and **potential candidates** (e.g., **POT1-AS1, SOX9-AS1, C8orf86**) supported by literature and genomic proximity.
- 🔍 Validates predictions through genomic overlap, proximity, and literature mining.

---
## Methodology Summary

### Data:
- **Positive instances**: 491 known DDR genes (e.g., from Knijnenburg et al., Weir et al.)
- **Negative instances**: 1048 non-DDR protein-coding genes
- **Features**: TCGA RNA-seq (FPKM, log2 normalized) across 33 cancer types
- **Prediction targets**: 40,683 lncRNAs from GENCODE v36

### Feature Representation:
- Dimensionality reduction techniques tested:
  - RF-based feature selection
  - Autoencoders
  - **Node2vec** (selected for best performance)

### ML Models:
- Logistic Regression (LR)
- Support Vector Machine (SVM with RBF kernel)
- Random Forest (RF)

### Validation:
- 5-fold cross-validation
- Independent test data (312 DDR genes, 8266 non-DDR genes)
- External prediction on unseen protein-coding genes (GENCODE v36)
- Functional enrichment via DAVID and GREAT

---
