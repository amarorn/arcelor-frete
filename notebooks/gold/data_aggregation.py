import logging
from pathlib import Path
import polars as pd
from utils.domain.processor import FretePorRotaService

logging.basicConfig(level=logging.INFO)


def run(df_silver: pd.DataFrame, output_dir: str = "gold_outputs") -> None:
    logging.info("📊 Starting Gold aggregation over %d rows", len(df_silver))

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    service = FretePorRotaService(df_silver)
    windows = service.filtrar_por_datas()

    resultado_3m = service.calcular_agregados_por_rota(windows["df_3m"])
    resultado_6m = service.calcular_agregados_por_rota(windows["df_6m"])
    resultado_12m = service.calcular_agregados_por_rota(windows["df_12m"])

    for name, df in [("3m", resultado_3m), ("6m", resultado_6m), ("12m", resultado_12m)]:
        file_path = output_path / f"resultado_frete_{name}.xlsx"
        df.to_excel(file_path, index=False)
        logging.info("Saved %s", file_path)

    logging.info("🏆 Gold aggregation completed.")
