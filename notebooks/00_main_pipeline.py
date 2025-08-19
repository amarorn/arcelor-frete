from pathlib import Path
import argparse
import logging

from notebooks.bronze import api_ingestion
from notebooks.silver import data_cleansing
from notebooks.gold import data_aggregation

logging.basicConfig(level=logging.INFO)


def run_pipeline(config_path: str) -> None:
    cfg_path = Path(config_path)
    logging.info("⛏️  Starting Medallion pipeline using config %s", cfg_path)

    bronze_df = api_ingestion.run(cfg_path)
    silver_df = data_cleansing.run(bronze_df)
    data_aggregation.run(silver_df)

    logging.info("✅ Pipeline finished successfully. 🏁")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Execute full Medallion pipeline")
    parser.add_argument("--config", default="config/config_api_real.json", help="JSON config path")
    args = parser.parse_args()

    run_pipeline(args.config)