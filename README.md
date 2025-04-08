# GlncDDR: Predicting DNA Damage Response-Associated lncRNAs Using Graph-Based Machine Learning

GlncDDR is a modular, reproducible machine learning pipeline for identifying long non-coding RNAs (lncRNAs) potentially involved in the DNA damage response (DDR), using gene expression data from TCGA and graph-based node embeddings (Node2Vec).

---

## Key Features

- ✅ Uses Node2Vec on WGCNA-derived gene co-expression graphs
- ✅ Trains classic ML models (Logistic Regression, SVM, Random Forest)
- ✅ Predicts DDR-associated lncRNAs and ranks them
- ✅ Supports reproducibility with Docker & Nextflow
- ✅ One-time graph embedding; stepwise pipeline

---

## Folder Structure
GlncDDR_Complete_Pipeline/
├── Dockerfile
├── main.nf
├── nextflow.config
├── requirements.txt
├── main.py
├── data/
│   ├── train.csv
│   ├── test.csv
│   ├── lncrna.csv
│   └── protein.csv
├── embeddings/
│   ├── train_emb_len100.csv
│   ├── test_emb_len100.csv
│   ├── lncrna_emb_len100.csv
│   └── protein_emb_len100.csv
├── ml_pipeline/
│   ├── embedding/
│   │   └── run_embedding.py
│   └── scripts/
│       ├── training.py
│       ├── testing.py
│       ├── predict_lnc.py
│       ├── predict_prot.py
│       └── utils.py



---

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt

Or build Docker image:

docker build -t glncddr .

How to Run
Step 1: Run Graph Embedding (once)

python ml_pipeline/embedding/run_embedding.py \
  --input sample_data/train.csv \
  --output embeddings/train_embed.csv \
  --walks 5 --length 10 --dim 100

Repeat for:

test.csv

lncrna.csv

protein.csv


Step 2: Train and Predict
python main.py \
  --train embeddings/train_embed.csv \
  --test embeddings/test_embed.csv \
  --predict_lnc embeddings/lncrna_embed.csv \
  --predict_prot embeddings/protein_embed.csv \
  --output output_dir/
Results will be saved in:

output_dir/training_metrics.txt

output_dir/test_metrics.txt

output_dir/combined_roc_pr.png

Excel files for prediction ranking

Run with Docker + Nextflow (Optional)

1. Build Docker Image
docker build -t glncddr .


2. Run via Nextflow
2.1. If you want to run embedding first:
nextflow run main.nf \
  --run_embedding true \
  --raw_train data/train.csv \
  --raw_test data/test.csv \
  --raw_lnc data/lncrna.csv \
  --raw_prot data/protein.csv \
  --output output_dir/
  
2.2.  If you want to use precomputed embeddings:
nextflow run main.nf \
  --run_embedding false \
  --train embeddings/train_emb_len100.csv \
  --test embeddings/test_emb_len100.csv \
  --predict_lnc embeddings/lncrna_emb_len100.csv \
  --predict_prot embeddings/protein_emb_len100.csv \
  --output output_dir/

  
Output
Performance metrics: Accuracy, Sensitivity, Specificity, MCC, F1, ROC-AUC

ROC and PR plots

Top-ranked DDR lncRNA predictions in Excel files

