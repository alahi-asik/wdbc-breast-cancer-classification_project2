# WDBC Breast Cancer Classification

## Project Overview
Binary classification model to predict breast cancer diagnosis (Malignant/Benign) 
using the Wisconsin Diagnostic Breast Cancer (WDBC) dataset from the UCI 
Machine Learning Repository.

**Dataset:** 569 samples, 30 numeric features, binary target  
**Environment:** Python 3.9+, Google Colab

---

## Project Structure
- `wdbc_classification_Asik.ipynb` — main notebook
- `requirements.txt` — package dependencies
- `wdbc.data` — dataset (download from UCI if not present)

---

## How to Run
1. Open `wdbc_classification_Asik.ipynb` in Google Colab
2. Upload `wdbc.data` to your Google Drive under a `datasets/` folder
3. Run `pip install -r requirements.txt`
4. Run all cells top to bottom

---

## Methodology
1. **Data Quality Check** — missing values, class distribution
2. **Exploratory Data Analysis** — 4 visualizations with rationale
3. **Baseline Model** — Logistic Regression (97% accuracy)
4. **Feature Selection** — Random Forest importance, reduced to top features
5. **Hyperparameter Tuning** — GridSearchCV with 5-fold CV
6. **Alternative Model** — Tuned Random Forest
7. **Model Comparison** — ROC curves, confusion matrices
8. **Threshold Tuning** — clinical priority (Precision ≥ 0.85, max Recall)
9. **Model Calibration** — reliability diagram for both models
10. **Model Persistence** — saved as single sklearn Pipeline

---

## Results
| Model | Accuracy | F1 Score | ROC-AUC |
|---|---|---|---|
| Baseline LR | ~97% | ~0.96 | ~0.99 |
| Tuned LR | ~97% | ~0.97 | ~0.999 |
| Tuned RF | ~96% | ~0.96 | ~0.999 |

**Recommended model:** Tuned Logistic Regression with decision threshold 0.35  
**Clinical rationale:** Prioritises recall for malignant cases to minimise missed cancers

---

## Key Findings
- The WDBC dataset is nearly linearly separable — LR performs excellently
- Most important features: `concave_points_worst`, `area_worst`, `perimeter_worst`
- Random Forest is less well-calibrated than LR for probability outputs
- Lowering the decision threshold increases malignant recall at a small precision cost

---

## Requirements
See `requirements.txt` or install directly:
pip install numpy pandas matplotlib seaborn scikit-learn optuna shap statsmodels joblib
