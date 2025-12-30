RetailGenius – Customer Churn Prediction

## 📌 Project Overview

RetailGenius is a machine learning project designed to **predict customer churn** using historical customer data.
The goal of this project is not only to build an accurate predictive model, but also to ensure **model transparency and explainability** using Explainable AI (XAI) techniques.

The project follows a **production-style ML pipeline**, including data preprocessing, model training, experiment tracking with MLflow, and model explainability using SHAP.

---

## 🎯 Objectives

* Predict whether a customer is likely to churn
* Track experiments and models using MLflow
* Explain model predictions using SHAP
* Provide business-relevant insights from model explanations

---

## 🗂️ Project Structure

```
retailgenius-churn-prediction/
│
├── data/
│   ├── raw/                # Raw input dataset (CSV)
│   └── processed/          # Processed dataset (Parquet)
│
├── src/
│   ├── config.py           # Paths and configuration
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py            # Model training + MLflow logging
│   ├── predict.py          # Inference script
│   └── shap_explain.py     # SHAP explainability
│
├── reports/
│   ├── shap_summary.png
│   └── shap_beeswarm.png
│
├── notebooks/              # (Optional) Exploratory notebooks
├── models/                 # Saved models (if any)
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## 📊 Dataset Description

The dataset contains customer-level information such as:

* Demographics (Age, Gender)
* Financial metrics (Annual Income, Total Spend)
* Behavioral metrics (Purchases, Returns, Support Contacts)
* Engagement metrics (Satisfaction Score, Last Purchase)
* Marketing features (Email Opt-In, Promotion Response)
* Target variable: **Target_Churn**

The target column is renamed to `churn` during preprocessing.

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the repository

```bash
git clone https://github.com/Dileep45817/retailgenius-churn-prediction.git
cd retailgenius-churn-prediction
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Add dataset

Place the dataset in:

```bash
data/raw/churn.csv
```

---

## 🧪 Running the Pipeline

### Step 1: Data Preprocessing

```bash
python -m src.data_preprocessing
```

### Step 2: Train the Model

```bash
python -m src.train
```

This step:

* Trains a Random Forest classifier
* Logs metrics and model artifacts to MLflow

### Step 3: Launch MLflow UI

```bash
mlflow ui
```

Access at:
👉 `http://127.0.0.1:5000`

---

## 🔍 Model Explainability (SHAP)

### Step 4: Set Model URI

Copy the **Run ID** from MLflow UI and set:

```bash
export MODEL_URI="runs:/<RUN_ID>/model"
```

### Step 5: Generate SHAP Plots

```bash
python -m src.shap_explain
```

### Generated Outputs

* `shap_summary.png` – Global feature importance
* `shap_beeswarm.png` – Feature impact distribution

These plots provide **global interpretability** of the churn model.

---

## 🧠 Explainability Notes

* SHAP summary and beeswarm plots were used as they are **stable and reliable**
* Local explanation plots (waterfall, force, dependence) were excluded due to instability with one-hot encoded features in tree-based models
* This approach follows SHAP best practices

---

## 🛠️ Technologies Used

* Python
* Pandas, NumPy
* Scikit-learn
* MLflow
* SHAP
* Matplotlib

---

## 📈 Business Insights

The explainability analysis shows that churn is primarily influenced by:

* Customer satisfaction score
* Number of support interactions
* Inactivity period
* Customer tenure


