# adapters/excel_adapter.py
import pandas as pd
import logging
import os
from utils.ports import IDataReader

logging.basicConfig(level=logging.INFO)

class ExcelAdapter(IDataReader):
    def __init__(self, path: str, date_col: str = '00.dt_doc_faturamento'):
        self.path = path
        self.date_col = date_col

    def _clean_numeric(self, df: pd.DataFrame, cols):
        for col in cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace(',', '.').astype(float)
        return df

    def _normalize_dates(self, df: pd.DataFrame):
        if self.date_col in df.columns:
            df[self.date_col] = pd.to_datetime(df[self.date_col], errors='coerce', dayfirst=True)
            df = df[df[self.date_col].notna()].copy()
        return df

    def read(self) -> pd.DataFrame:
        logging.info("Lendo arquivo: %s", self.path)

        ext = os.path.splitext(self.path)[1].lower()
        if ext == ".csv":
            df = pd.read_csv(self.path)
        elif ext in [".xls", ".xlsx"]:
            df = pd.read_excel(self.path, engine="openpyxl")
        else:
            raise ValueError(f"Formato de arquivo não suportado: {ext}")

        colunas_para_converter = [
            '02.01.00 - Volume (ton)',
            '02.01.01 - Frete Geral (BRL)',
            '02.01.02 - DISTANCIA (KM)'
        ]
        df = self._normalize_dates(df)
        df = self._clean_numeric(df, colunas_para_converter)

        logging.info("Leitura e limpeza inicial concluídas. Registros: %d", len(df))
        return df