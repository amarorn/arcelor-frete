import logging
import polars as pd

logging.basicConfig(level=logging.INFO)


def run(df_bronze: pd.DataFrame) -> pd.DataFrame:
    logging.info("🧹 Cleaning %d rows", len(df_bronze))

    df = df_bronze.copy()

    df = df.drop_duplicates()
    df = df.dropna(how="all")

    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    logging.info("Silver cleansing finished. Rows after cleaning: %d", len(df))
    return df
