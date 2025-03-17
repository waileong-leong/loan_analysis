import polars as pl
import re


# 1. Data Loading and Preprocessing
def load_and_preprocess_data(
    file_path, handle_outliers: bool = True, filter_outlier: bool = True
):
    """
    Load the loan data and perform initial preprocessing
    """
    print("Loading and preprocessing data...")
    # Load the data
    df = pl.read_csv(file_path)

    # Display basic information
    print(f"Dataset shape: {df.shape}")
    print("\nColumn names and data types:")
    print(df.schema)

    # Convert issue_d to datetime
    df = df.with_columns(
        [
            pl.col("issue_d")
            .str.strptime(pl.Datetime, format="%b-%Y")
            .alias("issue_date")
        ]
    ).drop("issue_d")

    # Convert term to numeric (months)
    df = df.with_columns(
        [pl.col("term").str.extract(r"(\d+)").cast(pl.Int32).alias("term_months")]
    ).drop("term")

    # Convert emp_length to ordinal values
    def map_emp_length(emp_length):
        if emp_length == "< 1 year":
            return 0
        elif emp_length == "10+ years":
            return 10
        else:
            match = re.search(r"(\d+)", emp_length)
            if match:
                return int(match.group(1))
            return None

    df = df.with_columns(
        [
            pl.col("emp_length")
            .map_elements(map_emp_length, return_dtype=pl.Int8)
            .alias("emp_length_years")
        ]
    )

    # Create binary default indicator
    df = df.with_columns(
        [
            (
                pl.col("loan_status").is_in(
                    [
                        "Default",
                        "Charged Off",
                    ]
                )
            )
            .cast(pl.Int32)
            .alias("is_default")
        ]
    )

    # Filtering loan_amnt > 0 and annual_inc >= 0
    df = df.filter((pl.col("loan_amnt") > 0) & (pl.col("annual_inc") >= 0))
    df = df.filter(
        pl.col("loan_status").is_in(
            [
                "Fully Paid",
                "Default",
                "Charged Off",
            ]
        )
    )

    # Handle outliers in numerical columns
    numerical_columns = ["dti"]

    if handle_outliers:

        for col in numerical_columns:
            if col in df.columns:
                # Calculate quantiles for outlier detection
                stats = df.select(
                    [
                        pl.col(col).quantile(0.25).alias("q1"),
                        pl.col(col).quantile(0.75).alias("q3"),
                        pl.col(col).median().alias("median"),
                    ]
                )

                q1 = stats[0, "q1"]
                q3 = stats[0, "q3"]
                median = stats[0, "median"]
                iqr = q3 - q1
                lower_bound = q1 - 3 * iqr
                upper_bound = q3 + 3 * iqr

                print(f"Outlier boundaries for {col}:")
                print(f"  Lower bound: {lower_bound}")
                print(f"  Upper bound: {upper_bound}")

                # Get count of outliers
                outlier_count = df.filter(
                    (pl.col(col) < lower_bound) | (pl.col(col) > upper_bound)
                ).height

                print(
                    f"  Number of outliers: {outlier_count} ({outlier_count/df.height*100:.2f}%)"
                )

                # Cap outliers
                df = df.with_columns(
                    [
                        pl.when(pl.col(col) < lower_bound)
                        .then(lower_bound)
                        .when(pl.col(col) > upper_bound)
                        .then(upper_bound)
                        .otherwise(pl.col(col))
                        .alias(f"{col}_capped"),
                        pl.when(pl.col(col) < lower_bound)
                        .then(1)
                        .when(pl.col(col) > upper_bound)
                        .then(1)
                        .otherwise(0)
                        .alias(f"{col}_outlier"),
                    ]
                )

    # Filter outliers
    if filter_outlier:
        for col in numerical_columns:
            if f"{col}_capped" in df.columns:
                df = df.filter(pl.col(f"{col}_outlier") == 0).drop(f"{col}_outlier")

    print(f"Dataset shape after filtering: {df.shape}")
    return df
