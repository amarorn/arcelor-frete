import pytest
import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os
import sys

sys.path.append(str(Path(__file__).parent.parent))


@pytest.fixture
def sample_data():
    np.random.seed(42)
    
    data = {
        '00.dt_doc_faturamento': pd.date_range('2024-01-01', periods=100, freq='D'),
        '01.Rota_MuniOrigem_MuniDestino': ['ROTA_A_B'] * 50 + ['ROTA_C_D'] * 50,
        'nm_transportadora_aux': ['TRANSP_1'] * 30 + ['TRANSP_2'] * 30 + ['TRANSP_3'] * 40,
        '02.01.00 - Volume (ton)': np.random.uniform(10, 100, 100),
        '02.01.01 - Frete Geral (BRL)': np.random.uniform(1000, 10000, 100),
        '02.01.02 - DISTANCIA (KM)': np.random.uniform(100, 1000, 100),
        '02.03.00 - Preço_Frete Geral (BRL) / TON': np.random.uniform(50, 200, 100),
        '02.03.02 - Preço_Frete Geral (BRL / TON / KM)': np.random.uniform(0.1, 2.0, 100),
        '00.nm_modal': ['RODOVIARIO'] * 100,
        'nm_tipo_rodovia': ['PEDAGIO'] * 50 + ['LIVRE'] * 50,
        'nm_veiculo': ['TRUCK'] * 60 + ['CARRETA'] * 40
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_polars_df(sample_data):
    return pl.from_pandas(sample_data)


@pytest.fixture
def temp_excel_file(sample_data):
    with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
        sample_data.to_excel(tmp.name, index=False)
        yield tmp.name
        os.unlink(tmp.name)


@pytest.fixture
def temp_model_dir():
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir


@pytest.fixture
def mock_logger(mocker):
    return mocker.patch('logging.getLogger')
