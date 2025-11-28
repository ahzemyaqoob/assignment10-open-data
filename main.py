"""
Open Data Project – Settlement Population Analysis (Lithuania)

Dataset:
    "Lietuvos gyventojų skaičius gyvenvietėse"
    https://data.gov.lt/datasets/2938/

Before running:
1. Download the CSV (format = CSV) from the dataset page.
2. Save it as 'gyventojai_gyvenvietese.csv' in the same folder as this script.
3. Install dependencies:
       pip install pandas matplotlib
4. Run the program:
       python main.py
"""

from __future__ import annotations

import os
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt

# Name of the CSV file you downloaded.
DATA_FILE = "gyventojai_gyvenvietese.csv"


def load_data(path: str) -> pd.DataFrame:
    """
    Load CSV file into a pandas DataFrame.

    Raises:
        FileNotFoundError: if the file does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Data file '{path}' not found.\n"
            "Make sure you downloaded the CSV from\n"
            "https://data.gov.lt/datasets/2938/ and saved it in this folder."
        )

    df = pd.read_csv(path)
    return df


def choose_column(df: pd.DataFrame, purpose: str) -> str:
    """
    Interactively ask the user to choose a column name for a given purpose.

    Args:
        df: DataFrame with the data.
        purpose: Short description, e.g. 'settlement name' or 'population'.

    Returns:
        The chosen column name.
    """
    print("\nAvailable columns:")
    for name in df.columns:
        print(f" - {name}")

    while True:
        col = input(
            f"\nType the exact column name to use as {purpose}: "
        ).strip()
        if col in df.columns:
            return col
        print("Column not found, please try again (check spelling and spaces).")


def clean_numeric_column(df: pd.DataFrame, column: str) -> pd.Series:
    """
    Convert a column to numeric, handling commas, non-breaking spaces, etc.

    Values that cannot be converted become NaN.
    """
    series = (
        df[column]
        .astype(str)
        .str.replace("\u00a0", " ", regex=False)   # non-breaking space
        .str.replace(",", "", regex=False)        # remove thousand separators
        .str.strip()
    )
    numbers = pd.to_numeric(series, errors="coerce")
    return numbers


def compute_basic_stats(
    df: pd.DataFrame, name_col: str, pop_col: str
) -> None:
    """
    Print simple statistics about settlements and population.
    """
    total_settlements = df[name_col].nunique()
    total_population = df[pop_col].sum()
    mean_population = df[pop_col].mean()
    median_population = df[pop_col].median()

    print("\n=== Basic statistics ===")
    print(f"Number of unique settlements: {total_settlements}")
    print(f"Total population (sum over all settlements): {int(total_population):,}")
    print(f"Average settlement population: {mean_population:,.1f}")
    print(f"Median settlement population:  {median_population:,.1f}")


def top_settlements(
    df: pd.DataFrame, name_col: str, pop_col: str, top_n: int = 10
) -> pd.DataFrame:
    """
    Return the top N settlements by population.
    """
    return (
        df[[name_col, pop_col]]
        .dropna(subset=[pop_col])
        .sort_values(pop_col, ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def print_top_settlements(
    df: pd.DataFrame, name_col: str, pop_col: str, top_n: int = 10
) -> None:
    """
    Print the top N settlements by population.
    """
    top_df = top_settlements(df, name_col, pop_col, top_n)
    print(f"\n=== Top {top_n} settlements by population ===")
    for idx, row in top_df.iterrows():
        rank = idx + 1
        name = str(row[name_col])
        population = int(row[pop_col])
        print(f"{rank:>2}. {name:<30} {population:,}")


def group_by_column(
    df: pd.DataFrame,
    group_col: str,
    pop_col: str,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Aggregate population by a grouping column (e.g. municipality, county).

    For each group, compute:
      - total_population
      - settlements_count
      - average_population
    """
    grouped = (
        df.dropna(subset=[pop_col])
        .groupby(group_col)
        .agg(
            total_population=(pop_col, "sum"),
            settlements_count=(pop_col, "count"),
            average_population=(pop_col, "mean"),
        )
        .sort_values("total_population", ascending=False)
        .head(top_n)
        .reset_index()
    )
    return grouped


def print_group_summary(
    df: pd.DataFrame, group_col: str, pop_col: str, top_n: int = 10
) -> None:
    """
    Print a summary of aggregated population by a chosen group column.
    """
    grouped = group_by_column(df, group_col, pop_col, top_n)
    print(
        f"\n=== Top {top_n} '{group_col}' values "
        f"by total population (sum of their settlements) ==="
    )
    for _, row in grouped.iterrows():
        name = str(row[group_col])
        total = int(row["total_population"])
        count = int(row["settlements_count"])
        avg = row["average_population"]
        print(
            f"{name:<30} total={total:,}  "
            f"settlements={count}  avg={avg:,.1f}"
        )


def plot_population_histogram(
    df: pd.DataFrame, pop_col: str, bins: int = 50
) -> None:
    """
    Plot a histogram of settlement populations.
    """
    plt.figure(figsize=(10, 6))
    df[pop_col].dropna().plot(kind="hist", bins=bins)
    plt.xlabel("Settlement population")
    plt.ylabel("Number of settlements")
    plt.title("Distribution of settlement population sizes")
    plt.tight_layout()
    plt.show()


def main() -> None:
    """
    Main entry point for the program.
    """
    print("Open Data Project – Settlement Population Analysis (Lithuania)")
    print(f"Trying to load data file: {DATA_FILE!r}")

    df_raw = load_data(DATA_FILE)
    print(f"Loaded {len(df_raw)} rows.")

    # Let the user choose which columns mean what.
    name_col = choose_column(df_raw, "settlement name")
    pop_col = choose_column(df_raw, "population")

    # Clean population column and drop rows without population.
    df = df_raw.copy()
    df[pop_col] = clean_numeric_column(df, pop_col)
    df = df.dropna(subset=[pop_col])

    # Basic statistics and top settlements.
    compute_basic_stats(df, name_col, pop_col)
    print_top_settlements(df, name_col, pop_col, top_n=10)

    # Optional aggregation by another column (e.g. municipality).
    use_grouping = input(
        "\nDo you want to aggregate by another column "
        "(e.g. municipality)? [y/N]: "
    ).strip().lower()

    if use_grouping == "y":
        group_col = choose_column(df, "grouping (e.g. municipality)")
        print_group_summary(df, group_col, pop_col, top_n=10)

    # Optional histogram plot.
    use_plot = input(
        "\nDo you want to show a histogram of settlement populations? "
        "[y/N]: "
    ).strip().lower()

    if use_plot == "y":
        plot_population_histogram(df, pop_col)

    print("\nDone. You can now use these results in your report.")


if __name__ == "__main__":
    main()
