# Model Card — ChurnGuard (Customer Churn Prediction)

## Summary
ChurnGuard is a supervised machine learning project that predicts whether a customer is likely to churn. The project demonstrates an end-to-end ML workflow suitable for industrial placement-level work.

## Intended Use
- Identify customers at high churn risk for retention analysis.
- Support data-driven decision-making (not automated decisions).

## Data
Customer demographic and service usage data.

### Target
- `Churn Value` (0 = retained, 1 = churned)

### Leakage Prevention
The following columns were removed to prevent data leakage:
- Churn Reason
- Churn Score
- Churn Label
- CustomerID

## Models Evaluated
- Logistic Regression (baseline)
- Random Forest
- Gradient Boosting (selected)

## Preprocessing
- One-hot encoding for categorical variables
- Feature scaling for numerical variables
- Pipeline-based preprocessing to ensure reproducibility

## Evaluation Metrics
- ROC-AUC (primary)
- F1 Score (secondary)

### Results
| Model | ROC-AUC | F1 |
|---|---:|---:|
| Logistic Regression | 0.835 | 0.573 |
| Random Forest | 0.834 | 0.586 |
| Gradient Boosting | **0.854** | 0.584 |

## Interpretability
Permutation feature importance was used to understand which features most influenced predictions.

## Limitations
- Single dataset
- Threshold fixed at 0.5
- No fairness audit conducted

## Future Improvements
- Threshold optimisation
- Hyperparameter tuning
- API deployment
- Fairness and bias analysis

## Reproducibility
- Train models: `python -m churnguard.models`
- Train + save model: `python -m churnguard.predict --train`
