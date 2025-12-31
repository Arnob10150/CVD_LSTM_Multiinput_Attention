# An Advanced Multi Input LSTM Framework with Attention for Predicting the Risk Level of Cardiovascular Disease

This project contains a Jupyter notebook that trains and evaluates a **multi input LSTM** model with **attention based fusion** to classify **CVD Risk Level** from a structured clinical dataset.

## What is in this repo

- `CVD_LSTM_multiinput_attention_final.ipynb`  
  End to end workflow: data loading, preprocessing, multi input model training, evaluation, uncertainty (MC Dropout), explainability (SHAP or LIME), and a simple digital twin style scenario simulator.

## Dataset
- `CAIR-CVD-2025: Cardiovascular Risk from Bangladesh`  
  https://www.kaggle.com/datasets/jocelyndumlao/cair-cvd-2025-cardiovascular-risk-from-bangladesh

## Key ideas implemented in the notebook

- **Multi input feature grouping** (each group is a separate model input)
- **LSTM encoder per group** (time dimension is kept as 1 so the same architecture can be extended to longitudinal data later)
- **Attention**
  - temporal attention inside each stream
  - inter group attention for fusion (learns which feature groups matter more)
- **Evaluation**: classification report, confusion matrix, ROC AUC (multi class), and optional K Fold runs
- **Uncertainty estimation**: Monte Carlo Dropout with predictive entropy and class probability intervals
- **Explainability**: SHAP (KernelExplainer) or LIME (tabular) with saved plots
- **Scenario simulation**: small what if changes (LDL drop, SBP drop, quitting smoking) to see how predicted risk shifts over horizons

## End-to-end-workflow

- Raw CVD Dataset → Data Cleaning & Validation → Feature Engineering → Encoding & Scaling → Multi-Input Feature Grouping → Train/Test Split → Multi-Input LSTM + Attention Model → Training & Optimization → Evaluation (Confusion Matrix, ROC-AUC) → Uncertainty Estimation (MC Dropout) → Explainability (SHAP/LIME) → Scenario Simulation → Final Risk Level Prediction


### Feature groups (inputs)

**Demographics**
- Sex
- Age
- Family History of CVD

**Anthropometrics**
- Weight
- Height (cm)
- BMI
- Abdominal Circumference (cm)
- Waist-to-Height Ratio

**Vitals**
- Systolic BP
- Diastolic BP
- Blood Pressure Category

**Lipids**
- Total Cholesterol
- HDL
- Estimated LDL

**Glucose / Metabolic**
- Fasting Blood Sugar
- Diabetes Status

**Lifestyle**
- Smoking Status
- Physical Activity Level

### Columns dropped in the notebook (if present)

- Blood Pressure (mmHg)  (string like "120/80")
- Height (m)  (Height (cm) is used)
- CVD Risk Score  (continuous, not used for this classification task)

If your dataset uses slightly different names (for example units or spacing), update the column lists in the notebook accordingly.

## Setup

### Option A: Google Colab (recommended)

1. Upload `CVD Dataset.csv` to the Colab session.
2. Upload or open the notebook in Colab.
3. Run all cells from top to bottom.

### Option B: Local environment

**Python:** 3.9+ recommended

Install dependencies:

```bash
pip install numpy pandas scikit-learn tensorflow matplotlib seaborn tqdm scipy shap lime
