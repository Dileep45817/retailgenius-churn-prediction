import pandas as pd
from src.config import Paths

def main():
    p = Paths()

    df = pd.read_csv(p.data_raw)

    print("\nDetected columns:")
    for c in df.columns:
        print("-", c)

    # ✅ explicit rename (dataset-specific, clean, exam-safe)
    if "Target_Churn" not in df.columns:
        raise ValueError("Target_Churn column not found in dataset")

    df = df.rename(columns={"Target_Churn": "churn"})

    # normalize churn values
    if df["churn"].dtype == "object":
        df["churn"] = (
            df["churn"]
            .str.strip()
            .str.lower()
            .map({"yes": 1, "no": 0, "true": 1, "false": 0})
        )

    df["churn"] = df["churn"].astype(int)

    df = df.drop_duplicates()
    df = df.dropna(axis=1, how="all")

    p.data_processed.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p.data_processed, index=False)

    print("\n✅ Preprocessing completed successfully")
    print("Final columns:", df.columns.tolist())

if __name__ == "__main__":
    main()

