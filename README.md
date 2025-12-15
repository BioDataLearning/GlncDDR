# GlncDDR: Predicting DNA Damage Response-Associated lncRNAs Using Graph-Based Machine Learning

**GlncDDR** is a machine learning pipeline designed to predict long non-coding RNAs (lncRNAs) involved in the **DNA Damage Response (DDR)**. This pipeline leverages **graph-based node embeddings (Node2Vec)** and gene expression data from **The Cancer Genome Atlas (TCGA)** to uncover novel DDR-associated lncRNAs.

---

## Key Features

- Graph-based learning using **Node2Vec** over WGCNA co-expression networks
- Node2Vec logic implemented based on the original pseudocode by Aditya Grover & Jure Leskovec ([node2vec: Scalable Feature Learning for Networks (KDD 2016)](https://doi.org/10.1145/2939672.2939754)
)
- Classical ML models: **Logistic Regression**, **Random Forest**, **Support Vector Machine**  
- Trained on known DDR/non-DDR genes from literature  
- Predicts and ranks DDR-associated lncRNAs from **GENCODE v36**  
- Performance up to **ROC-AUC: 0.95 (train)**

---

## Folder Structure 
```
GlncDDR_Complete_Pipeline/
├── requirements.txt
├── main.py
├── embeddings/
│   ├── train_emb_len100.csv
│   ├── test_emb_len100.csv
│   ├── lncrna_emb_len100.csv
│   └── protein_emb_len100.csv
├── pipelines/
│   ├── embedding/
│   │   └── run_embedding.py
│   └── ml/
│       ├── training.py
│       ├── testing.py
│       ├── predict_lnc.py
│       ├── predict_prot.py
│       └── utils.py
└── output_dir/
```
#### Data Availability
Note: The raw expression data files used to generate graph-based embeddings (e.g., TCGA expression profiles) are not included in this repository due to size constraints (multiple terabytes).

To access the raw data or reproduce the full embedding pipeline from scratch, please contact:

Snehal Shah
snehals@clemson.edu
Liangjiang Wang
liangjw@clemson.edu

---

## How it works

### 1. Install Dependencies
###### Recommended: conda
conda create -n glncddr python=3.9 -y \\
conda activate glncddr

###### Install Python deps
```bash
pip install -r requirements.txt
```

### 2. Running the ML scripts

#### Step 1: Run Embedding (optional, one-time only)
Use this step if you’re starting from raw gene expression:

python ml_pipeline/embedding/run_embedding.py \\
  --input       data/train.csv \\
  --output_dir  embeddings \\
  --vector-size 100 \\
  --walks       5 \\
  --length      10 

*Repeat for: test.csv, lncrna.csv, protein.csv


#### Step 2: Train and Predict
python main.py \\
  --train embeddings/train_emb_len100.csv \\
  --test embeddings/test_emb_len100.csv \\
  --predict_lnc embeddings/lncrna_emb_len100.csv \\
  --predict_prot embeddings/protein_emb_len100.csv \\
  --output output_dir/

##### Output Files

| File | Description |
|------|-------------|
| `training_metrics.txt` | Training set performance metrics (Accuracy, Sensitivity, etc.) |
| `test_metrics.txt`     | Test set evaluation results |
| `combined_roc_pr.png`  | ROC and PR curves from cross-validation |
| `*.xlsx`               | Ranked predictions for lncRNAs and protein-coding genes |


#### Example Performance Table
| Model | Accuracy | Sensitivity | Specificity |   MCC   | F1 Score | ROC-AUC |
|:-----:|:--------:|:-----------:|:-----------:|:-------:|:--------:|:-------:|
|  LR   |  0.85    |    0.79     |    0.94     |  0.72   |  0.86    |  0.95   |
|  RF   |  0.86    |    0.81     |    0.94     |  0.73   |  0.87    |  0.95   |
| SVM   |  0.82    |    0.72     |    0.970    |  0.68   |  0.83    |  0.95   |

