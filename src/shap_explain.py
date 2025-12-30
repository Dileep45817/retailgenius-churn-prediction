import os
import mlflow
import shap
import pandas as pd
import matplotlib.pyplot as plt

from src.config import Paths


def main():
    p = Paths()

    # Load processed data
    df = pd.read_parquet(p.data_processed)
    X = df.drop(columns=["churn"])

    # Load model
    model_uri = os.environ.get("MODEL_URI")
    if not model_uri:
        raise ValueError(
            "MODEL_URI not set.\n"
            "Example:\n"
            "export MODEL_URI='runs:/<RUN_ID>/model'"
        )

    pipeline = mlflow.sklearn.load_model(model_uri)
    preprocess = pipeline.named_steps["pre"]
    model = pipeline.named_steps["model"]

    # Transform features
    X_trans = preprocess.transform(X)

    try:
        feature_names = preprocess.get_feature_names_out()
    except Exception:
        feature_names = [f"feature_{i}" for i in range(X_trans.shape[1])]

    X_trans_df = pd.DataFrame(
        X_trans.toarray() if hasattr(X_trans, "toarray") else X_trans,
        columns=feature_names
    )

    # SHAP TreeExplainer (SAFE MODE)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(
        X_trans_df,
        check_additivity=False
    )

    # Binary classification → churn class
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values

    p.reports_dir.mkdir(parents=True, exist_ok=True)

    # =========================
    # 1️⃣ SUMMARY PLOT
    # =========================
    shap.summary_plot(sv, X_trans_df, show=False)
    plt.tight_layout()
    plt.savefig(p.reports_dir / "shap_summary.png", dpi=200)
    plt.close()

    # =========================
    # 2️⃣ BEESWARM PLOT
    # =========================
    shap.summary_plot(
        sv,
        X_trans_df,
        plot_type="bee",
        show=False
    )
    plt.tight_layout()
    plt.savefig(p.reports_dir / "shap_beeswarm.png", dpi=200)
    plt.close()

    print("\n✅ SHAP explainability completed successfully")
    print("📁 Outputs saved in reports/ directory")


if __name__ == "__main__":
    main()

