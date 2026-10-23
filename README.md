# RetailGenius: Customer Churn Analysis

A learning project exploring a churn-prediction workflow and model explainability with Python, scikit-learn, MLflow, and SHAP.

> **Repository status:** This public snapshot is incomplete and is not currently runnable end to end. It contains preprocessing and SHAP explanation scripts, but their referenced configuration and training components, dependency manifest, and sample data are not included. No model-performance results are published.

## Current implementation

- Preprocessing expects a `Target_Churn` column, maps common yes/no and true/false labels to 1/0, removes duplicate rows and fully empty columns, and writes a Parquet dataset.
- The explainability script loads an MLflow model from `MODEL_URI`, applies the model's preprocessing pipeline, and writes SHAP summary and beeswarm plots.
- Input data and trained model artifacts are not committed.

## Repository contents

```text
.
├── src/
│   ├── data_preprocessing.py
│   └── shap_explain.py
├── .gitignore
└── README.md
```

## What is needed to reproduce the workflow

The current scripts depend on modules and artifacts that are not present in this repository snapshot, including `src.config`, a trained MLflow model, and the data paths configured by the project. A dependency file and an authorized sample dataset are also needed before a clean setup and end-to-end run can be documented.

## Responsible reporting

This repository does not currently publish model metrics, validation results, or business-impact claims. Add a versioned train/test split, baseline comparison, and evaluation report before presenting predictive performance.

## Technologies explored

Python · pandas · scikit-learn · MLflow · SHAP · Matplotlib
