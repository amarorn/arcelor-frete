# ports.py
from typing import Protocol
import pandas as pd

class IDataReader(Protocol):
    def read(self) -> pd.DataFrame:
        ...

class IDataWriter(Protocol):
    def write(self, df: pd.DataFrame, path: str) -> None:
        ...

class IOpportunityAnalyzer(Protocol):
    def analyze_reduction_opportunities(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analisa oportunidades de redução de preços baseado no preço médio por TKU por micro-região
        """
        ...

class IMicroRegionPriceCalculator(Protocol):
    def calculate_average_price_by_microregion(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula o preço médio por TKU por micro-região
        """
        ...