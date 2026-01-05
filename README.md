ChurnGuard — Explainable Customer Churn Prediction

End-to-end customer churn prediction system with a leakage-safe machine learning pipeline and production-ready API.

ChurnGuard demonstrates how to take a tabular ML problem from raw data → trained model → evaluated results → deployed inference service, with a strong focus on engineering quality, reproducibility, and explainability.

🔍 What This Project Does

Predicts customer churn probability from structured customer data

Trains and evaluates multiple ML models using a clean, leakage-safe pipeline

Selects models using ROC-AUC and F1 score, not accuracy alone

Provides model explainability through feature importance

Serves predictions via a FastAPI REST API with Swagger documentation

🧠 Models Implemented

The following models were trained and compared using the same preprocessing pipeline:

Logistic Regression — interpretable baseline

Random Forest — ensemble model for non-linear relationships

Gradient Boosting — best overall performance

Best result:

ROC-AUC ≈ 0.85 (Gradient Boosting)

All evaluation artefacts (metrics, confusion matrix, feature importance) are saved to the reports/ directory.

📊 Evaluation & Explainability

Metrics used: ROC-AUC, F1 score

Confusion matrix generated for error analysis

Feature importance plots used to interpret model behaviour

Evaluation performed on a held-out test set to ensure fair comparison

This ensures model selection is data-driven and defensible, rather than arbitrary.

⚙️ Project Structure
churnguard/
│
├── src/churnguard/
│   ├── data/          # Data loading & validation
│   ├── features/      # Feature engineering & preprocessing
│   ├── models/        # Training & evaluation logic
│   ├── api.py         # FastAPI application
│   └── config.py      # Configuration
│
├── models/            # Saved trained model (joblib)
├── reports/           # Metrics, confusion matrix, feature importance
├── pyproject.toml     # Packaged Python project
└── README.md

🚀 Quick Start
1️⃣ Install dependencies
pip install -e .

2️⃣ Train the model
python -m churnguard.models.train


This will:

Train all models

Evaluate performance

Save the best model to models/

Export reports to reports/

🌐 API Usage (FastAPI)
Start the API
uvicorn churnguard.api:app --reload

Swagger UI
http://127.0.0.1:8000/docs

Available Endpoints
Method	Endpoint	Description
GET	/health	Service & model status
GET	/schema	Expected feature schema
GET	/example	Example request payload
POST	/predict	Returns churn probability
📥 Example Prediction Request
{
  "tenure": 12,
  "monthly_charges": 75.3,
  "total_charges": 900.5,
  "contract_type": "Month-to-month",
  "payment_method": "Electronic check"
}

Example Response
{
  "churn_probability": 0.78,
  "prediction": "Churn"
}

🧩 Engineering Decisions

Leakage prevention: preprocessing fitted on training data only

Reproducibility: deterministic pipelines and saved artefacts

Model versioning: trained model persisted using joblib

Separation of concerns: training logic isolated from API layer

Deployment-ready: FastAPI with schema validation and documentation

📌 Why This Project Matters

ChurnGuard demonstrates:

Practical machine learning engineering, not just modelling

Understanding of evaluation trade-offs

Ability to deploy ML systems, not just train them

Clean project organisation suitable for production environments

This project was built to reflect real-world ML workflows, not academic shortcuts.

🔮 Possible Extensions

Probability calibration

SHAP-based explainability

Model monitoring & drift detection

Containerisation (Docker)

CI pipeline for retraining and evaluation

👤 Author

Ali Hamza
BSc Computer Science with Artificial Intelligence
University of Huddersfield
