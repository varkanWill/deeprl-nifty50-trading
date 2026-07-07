"""
Feature Engineering Pipeline

Reads raw stock data, computes technical indicators using
FinRL's FeatureEngineer, and saves the processed dataset.
"""

import pandas as pd

from finrl.meta.preprocessor.preprocessors import FeatureEngineer

from settings.config import (
    RAW_DATA_PATH,
    PROCESSED_DATA_PATH,
    TECHNICAL_INDICATORS,
)


def main():

    print("=" * 80)
    print("Loading Raw Dataset")
    print("=" * 80)

    df = pd.read_csv(RAW_DATA_PATH)

    df = (
        df.sort_values(["date", "tic"])
        .reset_index(drop=True)
    )

    print(f"Rows Loaded    : {len(df)}")
    print(f"Columns Loaded : {len(df.columns)}")
    print()

    print("=" * 80)
    print("Generating Technical Indicators")
    print("=" * 80)

    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=TECHNICAL_INDICATORS,
        use_vix=False,
        use_turbulence=False,
        user_defined_feature=False,
    )

    processed_df = fe.preprocess_data(df)

    processed_df = (
        processed_df
        .sort_values(["date", "tic"])
        .reset_index(drop=True)
    )

    PROCESSED_DATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    processed_df.to_csv(
        PROCESSED_DATA_PATH,
        index=False,
    )

    print()
    print("=" * 80)
    print("Feature Engineering Completed Successfully")
    print("=" * 80)

    print(f"Saved File : {PROCESSED_DATA_PATH}")
    print(f"Rows       : {len(processed_df)}")
    print(f"Columns    : {len(processed_df.columns)}")
    print()

    print("=" * 80)
    print("Generated Columns")
    print("=" * 80)
    print(processed_df.columns.tolist())

    print()
    print("=" * 80)
    print("Missing Values")
    print("=" * 80)
    print(processed_df.isnull().sum())

    print()
    print("=" * 80)
    print("First Five Rows")
    print("=" * 80)
    print(processed_df.head())


if __name__ == "__main__":
    main()