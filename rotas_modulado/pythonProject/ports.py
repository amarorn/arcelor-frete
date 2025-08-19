# ports.py
from typing import Protocol
import pandas as pd

class IDataReader(Protocol):
    def read(self) -> pd.DataFrame:
        ...

class IDataWriter(Protocol):
    def write(self, df: pd.DataFrame, path: str) -> None:
        ...
