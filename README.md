# ME20 — Acne vs Eczema Classifier

## Overview

This project trains a binary image classifier to distinguish **Acne** from **Eczema** skin
conditions using **transfer learning** with MobileNetV2 pre-trained on ImageNet. The model is
built with TensorFlow / Keras and follows a two-phase training strategy: a feature-extraction
phase (frozen base) followed by a fine-tuning phase (top 30 layers unfrozen). The dataset
already ships with `train/` and `test/` splits; the notebook carves out a validation set
(15%) from the training data at runtime.

---

## Running the App

```bash
cd ME20
streamlit run app.py
```

The app will open automatically at **http://localhost:8501**

**Streamlit Cloud URL:** **

---

## Dataset

A brief description of the dataset source is provided in [`dataset/README.md`](dataset/README.md).

**Dataset:** Skin Disease Dataset — Kaggle  
**Classes:** `Acne`, `Eczema`  
**Split strategy:** Pre-made train/test from dataset · 15% val carved from train at runtime

---

## Environment Setup

### Requirements

Python **3.12** is recommended. All dependencies are pinned in [`requirements.txt`](requirements.txt).

### Install locally

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

### Key packages

| Package | Version | Purpose |
|---|---|---|
| `tensorflow` | 2.19.0 | Model training |
| `keras` | 3.15.0 | High-level API |
| `scikit-learn` | 1.8.0 | Evaluation metrics |
| `matplotlib` | 3.10.0 | Plotting |
| `seaborn` | 0.13.2 | Confusion matrix heatmap |
| `streamlit` | 1.60.0 | Web UI |

> **Note:** For faster training, use Google Colab with a T4 GPU runtime
> (`Runtime → Change runtime type → T4 GPU`).

---

## Project Structure

```
ME20/
├── dataset/
│   └── README.md          # Dataset source description
├── model/                 # Saved .keras model files
├── notebooks/
│   └── ME20.ipynb         # Full training pipeline
├── results/               # Plots, confusion matrices, learning curves
├── ood.py                 # OOD detection utility
├── requirements.txt
├── CONTRIBUTORS.md
└── README.md              # This file
```

---

## Challenges & Solutions

| Challenge | Solution / Notes |
|---|---|
| 22 classes in a single dataset | Filtered only `Acne` and `Eczema` folders by exact parent name matching (`parent == 'train'` / `'test'`) |
| No validation split provided | Carved 15% from training data per class using `random.shuffle` + slice |
| Visually similar inflammatory skin conditions | Fine-tuning top 30 MobileNetV2 layers improves texture and colour discrimination |

---

## Possible Improvements

- Extend to multi-class skin disease classification using all 22 available classes
- Experiment with **EfficientNetB0** as an alternative base model
- Apply class-weighted training if class imbalance is detected
- Export to **TensorFlow Lite** for mobile dermatology screening tools

---

## Results

> Full learning curves and confusion matrices are saved in `results/`.

---

## Contributors

See [CONTRIBUTORS.md](CONTRIBUTORS.md) for the full list of names, GitHub usernames, and registration numbers.
