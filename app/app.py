"""
Enhancer Prediction Web App
============================
Predicts whether an input DNA sequence is an ENHANCER or NON-ENHANCER
using a Fully-Connected Neural Network (FCNN) trained on 84-dimensional
k-mer frequency features (k=1,2,3).

Pipeline mirrors:
  • final_enhancer_pipeline_clean.ipynb  → feature engineering (k-mer CSV)
  • final_kmer_classification_pipeline_fixed.ipynb → FCNN architecture & training

Usage
-----
1.  Place your trained model file (fcnn_best.keras) and scaler
    (scaler.joblib) in the same directory as this script, OR upload them
    through the sidebar when the app is running.
2.  Run:  streamlit run app.py
"""

import itertools
import re
import warnings
import io

import numpy as np
import streamlit as st

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# K-MER FEATURE ENGINEERING  (matches final_enhancer_pipeline_clean.ipynb)
# ─────────────────────────────────────────────────────────────────────────────

# All 84 k-mer keys in the exact order used during training
NUC = (
    ["".join(n) for n in itertools.product("ACGT", repeat=1)]  # 4  mononucleotides
    + ["".join(n) for n in itertools.product("ACGT", repeat=2)]  # 16 dinucleotides
    + ["".join(n) for n in itertools.product("ACGT", repeat=3)]  # 64 trinucleotides
)  # total: 84

# Column names expected by the scaler / model
FEATURE_COLS = [f"kmer_{k}" for k in NUC]


def seq2kmer_single(seq: str) -> np.ndarray:
    """
    Convert one DNA sequence into an 84-dimensional normalised k-mer
    frequency vector.  Returns shape (84,) float32.

    Logic is identical to seq2kmer() in the pipeline notebook:
      - count each kmer
      - normalise by the number of possible positions for that k
    """
    seq = seq.strip().upper()
    cnt = {k: 0.0 for k in NUC}

    for i in range(len(seq)):
        if seq[i] in cnt:
            cnt[seq[i]] += 1

    for i in range(len(seq) - 1):
        kmer = seq[i : i + 2]
        if kmer in cnt:
            cnt[kmer] += 1

    for i in range(len(seq) - 2):
        kmer = seq[i : i + 3]
        if kmer in cnt:
            cnt[kmer] += 1

    vec = np.zeros(len(NUC), dtype=np.float32)
    for j, k in enumerate(NUC):
        d = len(seq) - (len(k) - 1)
        vec[j] = cnt[k] / (d if d > 0 else 1)

    return vec


MIN_BP = 150
MAX_BP = 250


def validate_sequence(seq: str) -> tuple[bool, str]:
    """
    Quality gates on user-supplied sequence.
    Returns (is_valid, error_message).

    Length requirement: 150–250 bp.
    The model was trained exclusively on 200 bp windows, so sequences
    outside 150–250 bp are outside the training distribution and will
    produce unreliable predictions.
    """
    seq = seq.strip().upper()
    if len(seq) == 0:
        return False, "Please enter a DNA sequence."
    invalid = set(seq) - set("ACGTN")
    if invalid:
        return (
            False,
            f"Sequence contains invalid characters: {', '.join(sorted(invalid))}. "
            "Only A, C, G, T, N are allowed.",
        )
    n_frac = seq.count("N") / len(seq)
    if n_frac > 0.10:
        return (
            False,
            f"{n_frac*100:.1f}% of bases are ambiguous (N). "
            "The model requires sequences with ≤10% N bases.",
        )
    if len(seq) < MIN_BP:
        return (
            False,
            f"Sequence is too short ({len(seq)} bp). "
            f"Minimum required length is {MIN_BP} bp — the model was trained on 200 bp windows.",
        )
    if len(seq) > MAX_BP:
        return (
            False,
            f"Sequence is too long ({len(seq)} bp). "
            f"Maximum allowed length is {MAX_BP} bp — the model was trained on 200 bp windows. "
            "Consider trimming to a 200 bp window centred on your region of interest.",
        )
    return True, ""


# ─────────────────────────────────────────────────────────────────────────────
# FCNN ARCHITECTURE  (matches final_kmer_classification_pipeline_fixed.ipynb)
# ─────────────────────────────────────────────────────────────────────────────

def build_fcnn(num_features: int = 84):
    """
    Reconstruct the exact FCNN used during training so that saved weights
    can be loaded back.

    Architecture:
        Input(84) → Dense(512,relu) → BN → Dropout(0.3)
                  → Dense(256,relu) → BN → Dropout(0.3)
                  → Dense(128,relu) → BN → Dropout(0.2)
                  → Dense(64,relu)
                  → Dense(1, float32) → Sigmoid(float32)
    """
    import tensorflow as tf
    from tensorflow.keras import layers

    inputs = layers.Input(shape=(num_features,), name="features")

    x = layers.Dense(512, activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(64, activation="relu")(x)

    output = layers.Dense(1, dtype="float32")(x)
    output = layers.Activation("sigmoid", dtype="float32", name="prob")(output)

    model = tf.keras.Model(inputs=inputs, outputs=output, name="FCNN")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# MODEL / SCALER LOADING  (cached so weights load only once per session)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model_and_scaler(model_bytes: bytes | None, scaler_bytes: bytes | None):
    """
    Load (or rebuild) the FCNN and its StandardScaler.

    Priority:
      1. User-uploaded files (model_bytes / scaler_bytes)
      2. Files named fcnn_best.keras / scaler.joblib in the working directory
      3. Demo mode: untrained model + identity scaler (for UI preview only)
    """
    import os, joblib, tempfile

    # ── Scaler ────────────────────────────────────────────────────────────────
    scaler = None
    if scaler_bytes:
        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
            tmp.write(scaler_bytes)
            tmp_path = tmp.name
        scaler = joblib.load(tmp_path)
        os.unlink(tmp_path)
    elif os.path.exists("scaler.joblib"):
        scaler = joblib.load("scaler.joblib")

    # ── Model ─────────────────────────────────────────────────────────────────
    import tensorflow as tf

    model = None
    if model_bytes:
        with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp:
            tmp.write(model_bytes)
            tmp_path = tmp.name
        try:
            model = tf.keras.models.load_model(tmp_path)
        except Exception:
            model = build_fcnn()
            model.load_weights(tmp_path)
        os.unlink(tmp_path)
    elif os.path.exists("fcnn_best.keras"):
        try:
            model = tf.keras.models.load_model("fcnn_best.keras")
        except Exception:
            model = build_fcnn()
            model.load_weights("fcnn_best.keras")

    demo_mode = model is None
    if demo_mode:
        model = build_fcnn()  # untrained — predictions are random

    return model, scaler, demo_mode


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────────────────────────────────────

def predict(seq: str, model, scaler) -> tuple[str, float, np.ndarray]:
    """
    1. Extract 84-dim k-mer vector from seq
    2. Scale with the training scaler (if available)
    3. Run FCNN inference
    Returns (label, confidence, kmer_vec)
    """
    kmer_vec = seq2kmer_single(seq)
    X = kmer_vec.reshape(1, -1).astype(np.float32)

    if scaler is not None:
        X = scaler.transform(X).astype(np.float32)

    prob = float(model.predict(X, verbose=0).ravel()[0])
    label = "ENHANCER" if prob >= 0.5 else "NON-ENHANCER"
    return label, prob, kmer_vec


# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG & STYLE
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Enhancer Predictor | FCNN",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject minimal custom CSS
st.markdown(
    """
    <style>
    /* Header banner */
    .main-banner {
        background: linear-gradient(135deg, #1a237e 0%, #0d47a1 50%, #1565c0 100%);
        color: white;
        padding: 2rem 2.5rem 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
    }
    .main-banner h1 { color: white; margin-bottom: 0.3rem; font-size: 2rem; }
    .main-banner p  { opacity: 0.85; font-size: 1rem; margin: 0; }

    /* Result cards */
    .result-enhancer {
        background: linear-gradient(135deg, #e8f5e9, #c8e6c9);
        border-left: 6px solid #2e7d32;
        padding: 1.2rem 1.5rem;
        border-radius: 8px;
        margin-top: 1rem;
    }
    .result-non-enhancer {
        background: linear-gradient(135deg, #fce4ec, #f8bbd0);
        border-left: 6px solid #c62828;
        padding: 1.2rem 1.5rem;
        border-radius: 8px;
        margin-top: 1rem;
    }
    .result-title { font-size: 1.6rem; font-weight: 700; margin-bottom: 0.3rem; }
    .confidence-text { font-size: 1rem; opacity: 0.85; }

    /* Sequence textarea */
    .stTextArea textarea { font-family: 'Courier New', monospace; font-size: 0.9rem; }

    /* Feature bar mini-chart label */
    .feat-label { font-size: 0.75rem; font-family: monospace; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — model upload & info
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Model Files")
    st.markdown(
        "Upload your **trained FCNN weights** and **scaler** exported from "
        "`final_kmer_classification_pipeline_fixed.ipynb`."
    )
    model_file = st.file_uploader(
        "FCNN weights  (.keras or .h5)",
        type=["keras", "h5"],
        help="fcnn_best.keras saved by ModelCheckpoint callback",
    )
    scaler_file = st.file_uploader(
        "StandardScaler  (.joblib)",
        type=["joblib"],
        help="scaler.joblib saved by joblib.dump(scaler, ...)",
    )

    st.divider()
    st.markdown("## 📚 Pipeline Summary")
    st.markdown(
        """
**Data source**
VISTA Enhancer DB (positive) vs GC-balanced genomic background (negative)

**Feature engineering**
84-dimensional normalised k-mer frequency vector
- 4  mono-nucleotides (k=1)
- 16 di-nucleotides   (k=2)
- 64 tri-nucleotides  (k=3)

**Model — FCNN**
```
Input(84)
→ Dense(512) → BN → Dropout(0.3)
→ Dense(256) → BN → Dropout(0.3)
→ Dense(128) → BN → Dropout(0.2)
→ Dense(64)
→ Dense(1)   → Sigmoid
```
Trained with Adam, binary cross-entropy, early stopping on val AUC.
        """
    )

    st.divider()
    st.markdown("## 🔬 About")
    st.info(
        "Bioinformatics & Healthcare AI\n\n"
        
    )

# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODEL (after sidebar so file uploaders are already mounted)
# ─────────────────────────────────────────────────────────────────────────────

model_bytes  = model_file.read()  if model_file  else None
scaler_bytes = scaler_file.read() if scaler_file else None

with st.spinner("Loading model…"):
    model, scaler, demo_mode = load_model_and_scaler(model_bytes, scaler_bytes)

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <div class="main-banner">
      <h1>🧬 Enhancer Prediction Tool</h1>
      <p>
        Predict whether a DNA sequence is an <strong>enhancer</strong> or
        <strong>non-enhancer</strong> using a Fully-Connected Neural Network (FCNN)
        trained on k-mer frequency features.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

if demo_mode:
    st.warning(
        "⚠️ **Demo mode** — no trained weights found. "
        "Upload `fcnn_best.keras` and `scaler.joblib` via the sidebar for real predictions.",
        icon="⚠️",
    )
elif scaler is None:
    st.info(
        "ℹ️ Model loaded but **no scaler** found. "
        "Predictions will still run but may be less accurate without feature scaling. "
        "Upload `scaler.joblib` to fix this.",
        icon="ℹ️",
    )

# ─────────────────────────────────────────────────────────────────────────────
# INPUT TABS  — single sequence  |  FASTA batch  |  example sequences
# ─────────────────────────────────────────────────────────────────────────────

EXAMPLES = {
    "VISTA-like Enhancer (GC-balanced, 200 bp)": (
        "ATGCTAGCATGCTAGCATGCATGCGCGCATGCATGCATGCATGCATGCATGCATGCATGC"
        "ATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGC"
        "ATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGC"
        "ATGCATGCATGCATGCATGCATGCAT"
    ),  # 200 bp
    "Random genomic background (AT-rich, 200 bp)": (
        "TTAAGCCTTAGGCTTAAGCCTTAGGCTTAAGCCTTAGGCTTAAGCCTTAGGCTTAAGCCTT"
        "AGGCTTAAGCCTTAGGCTTAAGCCTTAGGCTTAAGCCTTAGGCTTAAGCCTTAGGCTTAAG"
        "CCTTAGGCTTAAGCCTTAGGCTTAAGCCTTAGGCTTAAGCCTTAGGCTTAAGCCTTAGGCT"
        "TAAGCCTTAGGCTTAAGC"
    ),  # 200 bp
    "GC-rich sequence (200 bp)": (
        "GCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGC"
        "GCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGC"
        "GCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCG"
        "CGCGCGCGCGCG"
    ),  # 200 bp
}

tab_single, tab_fasta, tab_examples = st.tabs(
    ["🔬 Single Sequence", "📂 FASTA Batch", "💡 Examples"]
)

# ── Tab 1: single sequence ────────────────────────────────────────────────────
with tab_single:
    col_input, col_result = st.columns([1, 1], gap="large")

    with col_input:
        st.markdown("### Enter DNA Sequence")
        st.caption(
            f"⚠️ **Required length: {MIN_BP}–{MAX_BP} bp** "
            "(model trained on 200 bp windows — sequences outside this range are rejected)"
        )
        user_seq = st.text_area(
            "Paste your sequence (A, C, G, T, N only):",
            height=200,
            placeholder="Paste a 150–250 bp DNA sequence here (A, C, G, T, N only)…",
            label_visibility="collapsed",
        )

        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            run_btn = st.button("🔍 Predict", type="primary", use_container_width=True)
        with c2:
            clear_btn = st.button("🗑 Clear", use_container_width=True)

        # Sequence stats
        if user_seq.strip():
            clean = user_seq.strip().upper()
            total = len(clean)
            gc = (clean.count("G") + clean.count("C")) / max(total, 1) * 100
            n_pct = clean.count("N") / max(total, 1) * 100
            st.markdown(
                f"**Length:** {total} bp &nbsp;|&nbsp; "
                f"**GC:** {gc:.1f}% &nbsp;|&nbsp; "
                f"**N:** {n_pct:.1f}%"
            )

    with col_result:
        st.markdown("### Prediction Result")
        if run_btn and user_seq.strip():
            valid, err = validate_sequence(user_seq)
            if not valid:
                st.error(f"❌ {err}")
            else:
                with st.spinner("Running FCNN…"):
                    label, prob, kmer_vec = predict(user_seq, model, scaler)

                conf = prob if label == "ENHANCER" else 1 - prob
                emoji = "✅" if label == "ENHANCER" else "❌"
                css_cls = "result-enhancer" if label == "ENHANCER" else "result-non-enhancer"
                color = "#2e7d32" if label == "ENHANCER" else "#c62828"

                st.markdown(
                    f"""
                    <div class="{css_cls}">
                      <div class="result-title" style="color:{color}">
                        {emoji} {label}
                      </div>
                      <div class="confidence-text" style="color:black">
                        Enhancer probability: <strong>{prob*100:.2f}%</strong><br>
                        Confidence: <strong>{conf*100:.1f}%</strong>
                        {"(⚠️ demo mode)" if demo_mode else ""}
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.progress(float(prob), text=f"P(enhancer) = {prob:.4f}")

                # K-mer feature visualisation (top contributing bases)
                with st.expander("📊 K-mer Feature Profile"):
                    import pandas as pd

                    feat_df = pd.DataFrame(
                        {"k-mer": NUC, "frequency": kmer_vec}
                    )
                    # Show mono & di only (more readable)
                    feat_df_show = feat_df.iloc[:20].set_index("k-mer")
                    st.bar_chart(feat_df_show, height=250)
                    st.caption(
                        "Normalised frequencies for the 4 mono-nucleotides "
                        "and 16 di-nucleotides."
                    )

        elif run_btn:
            st.warning("Please enter a sequence first.")
        else:
            st.markdown(
                "<div style='color:#888; padding-top:3rem; text-align:center;'>"
                "Enter a sequence and click <strong>Predict</strong></div>",
                unsafe_allow_html=True,
            )

# ── Tab 2: FASTA batch ────────────────────────────────────────────────────────
with tab_fasta:
    st.markdown("### Batch Prediction from FASTA File")
    st.markdown(
        "Upload a FASTA file; each sequence is independently classified. "
        "Results are shown in a table and can be downloaded as CSV."
    )

    fasta_file = st.file_uploader("Upload FASTA (.fa / .fasta / .txt)", type=["fa", "fasta", "txt"])

    if fasta_file:
        content = fasta_file.read().decode("utf-8", errors="replace")
        # Parse FASTA
        records = []
        current_id, current_seq = None, []
        for line in content.splitlines():
            line = line.strip()
            if line.startswith(">"):
                if current_id is not None:
                    records.append((current_id, "".join(current_seq)))
                current_id = line[1:].split()[0]
                current_seq = []
            elif line:
                current_seq.append(line)
        if current_id is not None:
            records.append((current_id, "".join(current_seq)))

        st.info(f"Found **{len(records)}** sequences in the uploaded file.")

        if st.button("🔍 Run Batch Prediction", type="primary"):
            import pandas as pd

            rows = []
            progress = st.progress(0, text="Predicting…")
            for i, (seq_id, seq) in enumerate(records):
                valid, err = validate_sequence(seq)
                if not valid:
                    rows.append(
                        {
                            "ID": seq_id,
                            "Length": len(seq),
                            "GC%": None,
                            "P(enhancer)": None,
                            "Prediction": f"INVALID: {err}",
                        }
                    )
                else:
                    label, prob, _ = predict(seq, model, scaler)
                    gc = (seq.upper().count("G") + seq.upper().count("C")) / len(seq) * 100
                    rows.append(
                        {
                            "ID": seq_id,
                            "Length": len(seq),
                            "GC%": round(gc, 2),
                            "P(enhancer)": round(prob, 4),
                            "Prediction": label,
                        }
                    )
                progress.progress((i + 1) / len(records), text=f"Predicting… ({i+1}/{len(records)})")

            results_df = pd.DataFrame(rows)
            st.dataframe(results_df, use_container_width=True)

            csv_bytes = results_df.to_csv(index=False).encode()
            st.download_button(
                "⬇️ Download Results CSV",
                data=csv_bytes,
                file_name="enhancer_predictions.csv",
                mime="text/csv",
            )

            # Summary
            valid_rows = results_df.dropna(subset=["P(enhancer)"])
            n_enh = (valid_rows["Prediction"] == "ENHANCER").sum()
            n_non = (valid_rows["Prediction"] == "NON-ENHANCER").sum()
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Total sequences", len(records))
            col_b.metric("Predicted ENHANCER", n_enh)
            col_c.metric("Predicted NON-ENHANCER", n_non)

# ── Tab 3: example sequences ──────────────────────────────────────────────────
with tab_examples:
    st.markdown("### Try Example Sequences")
    st.markdown("Click a button to load an example sequence into the predictor.")

    for name, seq in EXAMPLES.items():
        col_a, col_b = st.columns([3, 1])
        with col_a:
            st.code(seq[:80] + "…", language=None)
            st.caption(f"**{name}** — {len(seq)} bp")
        with col_b:
            if st.button(f"Predict →", key=f"ex_{name}"):
                valid, err = validate_sequence(seq)
                if not valid:
                    st.error(err)
                else:
                    label, prob, kmer_vec = predict(seq, model, scaler)
                    emoji = "✅" if label == "ENHANCER" else "❌"
                    conf = prob if label == "ENHANCER" else 1 - prob
                    st.success(
                        f"{emoji} **{label}** "
                        f"(P={prob*100:.1f}%, conf={conf*100:.1f}%)"
                    )
        st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#888; font-size:0.85rem;'>"
    "Enhancer Predictor &nbsp;|&nbsp; FCNN + K-mer features &nbsp;|&nbsp; "
    "Bioinformatics & Healthcare AI &nbsp;|&nbsp; Supervisor: Dr. Fahim "
    "</div>",
    unsafe_allow_html=True,
)
