# adapters/excel_exporter.py
import pandas as pd
import logging
from ports import IDataWriter

class ExcelExporter(IDataWriter):
    def write(self, df: pd.DataFrame, path: str) -> None:
        logging.info("Exportando DataFrame para %s", path)
        df.to_excel(path, index=False)
        logging.info("Exportação finalizada.")
