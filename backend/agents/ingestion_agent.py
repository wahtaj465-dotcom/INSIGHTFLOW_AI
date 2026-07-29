import pandas as pd
from pathlib import Path


def load_dataset(file_path):
    """
    Load a CSV or Excel dataset and return it as a Pandas DataFrame.
    """

    file_path = Path(file_path)

    # Check whether the file exists
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    extension = file_path.suffix.lower()

    # Load CSV
    if extension == ".csv":
        df = pd.read_csv(file_path)

    # Load Excel
    elif extension in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path)

    else:
        raise ValueError(
            f"Unsupported file type: {extension}. "
            "Currently supported: CSV and Excel."
        )

    # Prevent empty datasets from entering the pipeline
    if df.empty:
        raise ValueError("The uploaded dataset is empty.")

    return df


if __name__ == "__main__":

    dataset = load_dataset("data/sales.csv")

    print("\nDataset loaded successfully!")

    print("\nFirst 5 rows:")
    print(dataset.head())

    print("\nShape:")
    print(dataset.shape)

    print("\nColumns:")
    print(dataset.columns.tolist())

    print("\nData Types:")
    print(dataset.dtypes)