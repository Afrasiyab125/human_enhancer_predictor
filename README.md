# 🧬 Enhancer Prediction Web App

A Streamlit web application that predicts whether a DNA sequence is an
**enhancer** or **non-enhancer** using a Fully-Connected Neural Network (FCNN)
trained on 84-dimensional k-mer frequency features.

---

## How the Pipeline Works

### Step 1 — Build the CSV dataset (`final_enhancer_pipeline_clean.ipynb`)

| Sub-step | What happens |
|----------|-------------|
| 1 | Parse VISTA FASTA → extract chr1 enhancer coordinates (BED) |
| 2 | `bedtools getfasta` → retrieve 200 bp enhancer windows (FASTA) |
| 3 | Compute GC-content statistics of positive sequences |
| 4 | Generate negative sequences from unannotated genomic background |
| 5 | GC-balance negatives (keep only sequences within mean ± 1 SD of positives) |
| 6 | Split everything 70/20/10 → train / val / test |
| 7 | `seq2kmer()` → convert every FASTA sequence to an 84-dim k-mer vector |
| 8 | Assign labels (1 = enhancer, 0 = non-enhancer), stack arrays |
| 9 | Save **`enhancer_kmer_features.csv`** (84 kmer columns + `label` + `split`) |

The 84 k-mer columns are named `kmer_A`, `kmer_C`, … `kmer_TTT` covering all
4 mononucleotides, 16 dinucleotides, and 64 trinucleotides.

### Step 2 — Train the FCNN (`final_kmer_classification_pipeline_fixed.ipynb`)

```
Input(84) → Dense(512, relu) → BatchNorm → Dropout(0.3)
          → Dense(256, relu) → BatchNorm → Dropout(0.3)
          → Dense(128, relu) → BatchNorm → Dropout(0.2)
          → Dense(64,  relu)
          → Dense(1,   float32) → Sigmoid
```

Trained with:
- Adam (lr=1e-3), binary cross-entropy
- Class-weighted loss for imbalance
- Early stopping on val AUC (patience=8)
- ReduceLROnPlateau

Outputs saved to disk:
- `fcnn_best.keras` — best model weights (by val AUC)
- `scaler.joblib`   — fitted `StandardScaler`

---

## Running the App

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Place model files in the same folder as `app.py`
```
app.py
fcnn_best.keras        ← from ModelCheckpoint in the training notebook
scaler.joblib          ← from joblib.dump(scaler, ...) in the training notebook
requirements.txt
```

### 3. Launch
```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## Using the App Without Pre-trained Weights

If you haven't trained the model yet, the app launches in **Demo mode**
(random predictions). Upload `fcnn_best.keras` and `scaler.joblib` via the
sidebar at any time — no restart needed.

---

## Features

| Feature | Description |
|---------|-------------|
| **Single sequence** | Paste any DNA sequence; get enhancer probability + confidence |
| **FASTA batch** | Upload a `.fa` file; all sequences are classified and results downloadable as CSV |
| **Examples** | Three built-in example sequences to explore the tool |
| **K-mer profile** | Bar chart of the 20 most informative k-mer features for your sequence |
| **Sequence validation** | Checks for invalid bases, excessive N content, and minimum length |

---

## Exporting Model Files from the Training Notebook

In `final_kmer_classification_pipeline_fixed.ipynb`, the model and scaler are
saved automatically:

```python
# Scaler
joblib.dump(scaler, f"{SAVE_DIR}/scaler.joblib")   # already in the notebook

# FCNN best weights — saved by ModelCheckpoint callback
# Output: {SAVE_DIR}/fcnn_best.keras
```


---

## Project Info

**Project** — Bioinformatics  
Islamia College University Peshawar  
Supervisor: Dr. Faheem
