import os
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris


def main():
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(out_dir, exist_ok=True)

    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    df = df.drop(columns=["target"]).copy()
    df.columns = [
        "sepal_length_cm",
        "sepal_width_cm",
        "petal_length_cm",
        "petal_width_cm",
    ]

    # keep clean class labels to avoid noisy labels and maintain learning quality
    df["species"] = iris.target.astype(int).map({0: "setosa", 1: "versicolor", 2: "virginica"})

    rng = np.random.default_rng(42)

    # light cleaning noise: enough for students to preprocess, but not too much to destroy signal
    for col in ["sepal_length_cm", "sepal_width_cm", "petal_length_cm", "petal_width_cm"]:
        idx = rng.choice(len(df), size=max(2, len(df) // 40), replace=False)
        df.loc[idx, col] = np.nan

    # mild duplicated samples to teach deduplication without overwhelming the task
    dup_idx = rng.choice(len(df), size=8, replace=False)
    df = pd.concat([df, df.iloc[dup_idx].copy()], ignore_index=True)

    # small realistic outliers, not extreme values that dominate the data
    outlier_values = {
        "sepal_length_cm": [8.5, 3.5],
        "sepal_width_cm": [3.5, 2.4],
        "petal_length_cm": [7.0, 1.2],
        "petal_width_cm": [2.5, 0.1],
    }
    for col, values in outlier_values.items():
        idx = rng.choice(len(df), size=len(values), replace=False)
        for i, value in zip(idx, values):
            df.at[i, col] = value

    # engineered features, but only the most meaningful ones for iris classification
    df["petal_area_cm2"] = df["petal_length_cm"] * df["petal_width_cm"]
    df["petal_to_sepal_length_ratio"] = np.where(
        df["sepal_length_cm"].notna() & (df["sepal_length_cm"] != 0),
        df["petal_length_cm"] / df["sepal_length_cm"],
        np.nan,
    )
    df["petal_length_minus_sepal_length"] = df["petal_length_cm"] - df["sepal_length_cm"]

    # fill obvious type issues and keep clean labels
    for col in ["sepal_length_cm", "sepal_width_cm", "petal_length_cm", "petal_width_cm", "petal_area_cm2", "petal_to_sepal_length_ratio", "petal_length_minus_sepal_length"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # median fill for missing values
    for col in ["sepal_length_cm", "sepal_width_cm", "petal_length_cm", "petal_width_cm", "petal_area_cm2", "petal_to_sepal_length_ratio", "petal_length_minus_sepal_length"]:
        df[col] = df[col].fillna(df[col].median())

    # remove duplicates after filling values
    df = df.drop_duplicates().reset_index(drop=True)

    output_path = os.path.join(out_dir, "iris_flowers_improved.csv")
    df.to_csv(output_path, index=False)

    print(f"Dataset generated: {output_path}")
    print(f"Shape: {df.shape}")
    print(df.isna().sum().to_dict())
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
