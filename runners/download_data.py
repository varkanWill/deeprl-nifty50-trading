from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import pandas as pd

from finrl.meta.preprocessor.yahoodownloader import YahooDownloader

from settings.config import (
    START_DATE,
    END_DATE,
    TICKERS,
    RAW_DATA_DIR,
    RAW_DATA_FILE,
)

def download_data() -> pd.DataFrame:
    """
    Download historical stock data from Yahoo Finance.

    Returns
    -------
    pd.DataFrame
        Raw historical stock dataframe.
    """

    print("=" * 60)
    print("Downloading stock data from Yahoo Finance...")
    print("=" * 60)

    downloader = YahooDownloader(
        start_date=START_DATE,
        end_date=END_DATE,
        ticker_list=TICKERS,
    )

    df = downloader.fetch_data()

    return df


def validate_data(df: pd.DataFrame) -> None:
    """
    Validate downloaded dataset.
    """

    print("\nValidating downloaded dataset...\n")

    if df.empty:
        raise ValueError("Downloaded dataframe is empty.")

    required_columns = [
        "date",
        "tic",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    print("Dataset Information")
    print("-" * 60)
    print(df.info())

    print("\nSummary Statistics")
    print("-" * 60)
    print(df.describe())

    print("\nMissing Values")
    print("-" * 60)
    print(df.isnull().sum())

    print("\nStocks Downloaded")
    print("-" * 60)
    print(df["tic"].unique())

    print(f"\nNumber of Stocks : {df['tic'].nunique()}")

    print(f"\nDate Range : {df['date'].min()} --> {df['date'].max()}")

    print(f"\nTotal Rows : {len(df):,}")


def save_data(df: pd.DataFrame) -> None:
    """
    Save dataframe to CSV.
    """

    RAW_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = df.sort_values(
        by=["date", "tic"]
    ).reset_index(drop=True)

    df.to_csv(
        RAW_DATA_FILE,
        index=False,
    )

    print("\n" + "=" * 60)
    print(f"Dataset successfully saved to:\n{RAW_DATA_FILE}")
    print("=" * 60)


def main() -> None:

    df = download_data()

    validate_data(df)

    save_data(df)

    print("\nFirst 5 Rows")
    print("-" * 60)
    print(df.head())


if __name__ == "__main__":
    main()