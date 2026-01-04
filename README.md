# ChurnGuard — Explainable Customer Churn Prediction

Predict customer churn using machine learning and explain *why* customers are likely to leave.

---

## Project Overview

ChurnGuard is an end-to-end machine learning system for predicting customer churn using real-world tabular data.  
The project covers the full ML lifecycle — from data preprocessing and model evaluation to deployment as a FastAPI service for live inference.

The goal is to demonstrate practical machine learning, responsible evaluation, and production-style software engineering.

---

## Tech Stack

- Python
- Pandas, NumPy, scikit-learn
- FastAPI
- Matplotlib
- Git

---

## Results

| Model | ROC-AUC | F1 |
|---|---:|---:|
| Logistic Regression (baseline) | 0.835 | 0.573 |
| Random Forest | 0.834 | 0.586 |
| Gradient Boosting | **0.854** | 0.584 |

### Baseline Confusion Matrix
![Baseline confusion matrix](reports/figures/baseline_confusion_matrix.png)

### Feature Importance (Permutation Importance)
![Permutation importance](reports/figures/permutation_importance_top15.png)

**Notes**
- Leakage-prone columns (e.g. *Churn Reason*, *Churn Score*) were removed to ensure fair evaluation.
- ROC-AUC and F1 were selected due to class imbalance.
- Gradient Boosting achieved the strongest overall discrimination performance.

---

## 🚀 API Usage (FastAPI)

This project includes a FastAPI service for running live churn predictions using the trained model.

### 1. Set up environment
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .\.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
