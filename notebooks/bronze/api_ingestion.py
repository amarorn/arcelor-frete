import json
import logging
from pathlib import Path
import polars as pd
from typing import Union

try:
    from utils.api_client import APIClient
except ImportError:
    class APIClient:
        def __init__(self, base_url: str):
            self.base_url = base_url

        def fetch(self, endpoint: str) -> list:
            logging.warning("Mock APIClient in use – returning empty dataset from %s", endpoint)
            return []

logging.basicConfig(level=logging.INFO)


def run(config_path: Union[str, Path]) -> pd.DataFrame:
    cfg_path = Path(config_path)
    with open(cfg_path, "r", encoding="utf-8") as fp:
        cfg = json.load(fp)

    api_cfg = cfg["api"]
    client = APIClient(api_cfg["base_url"])
    logging.info("🔄 Ingesting data from %s", api_cfg["endpoint"])

    raw_records = client.fetch(api_cfg["endpoint"])
    df_raw = pd.DataFrame.from_records(raw_records)

    logging.info("Bronze ingestion completed. Rows: %d", len(df_raw))
    return df_raw
