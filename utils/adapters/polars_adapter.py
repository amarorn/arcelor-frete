"""
Adaptador para Polars - Processamento de dados de alta performance
Substitui pandas para operações de análise de fretes
"""

import polars as pl
from pathlib import Path
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class PolarsAdapter:
    """
    Adaptador para processamento de dados usando Polars
    Oferece 10-100x melhor performance que pandas para datasets grandes
    """
    
    def __init__(self):
        self.df: Optional[pl.DataFrame] = None
        self.column_mapping = {
            '00.dt_doc_faturamento': 'data_faturamento',
            '01.Rota_MuniOrigem_MuniDestino': 'rota',
            'nm_transportadora_aux': 'transportadora',
            '02.01.00 - Volume (ton)': 'volume_ton',
            '02.01.01 - Frete Geral (BRL)': 'frete_brl',
            '02.01.02 - DISTANCIA (KM)': 'distancia_km',
            '02.03.00 - Preço_Frete Geral (BRL) / TON': 'preco_ton',
            '02.03.02 - Preço_Frete Geral (BRL / TON / KM)': 'preco_ton_km',
            '00.nm_modal': 'modal',
            'nm_tipo_rodovia': 'tipo_rodovia',
            'nm_veiculo': 'tipo_veiculo',
            # Novas colunas da base real
            '01.Rota MuniOrigem MuniDestino': 'rota_municipio',
            '01.Rota MicroOrigem MicroDestino': 'rota_microregiao',
            '01.Rota MesoOrigem MesoDestino': 'rota_mesoregiao',
            '01.Rota UFOrigem UFDestino': 'rota_uf',
            'de centro unidade descricao': 'centro_origem_descricao',
            '00.nm.centro origem unidade': 'centro_origem_unidade'
        }
    
    def read_excel(self, file_path: str, **kwargs) -> pl.DataFrame:
        """
        Lê arquivo Excel usando Polars (muito mais rápido que pandas)
        """
        try:
            logger.info(f"Lendo arquivo Excel: {file_path}")
            
            # Polars lê Excel de forma mais eficiente
            df = pl.read_excel(file_path)
            
            logger.info(f"Arquivo lido com sucesso. Shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Erro ao ler arquivo Excel: {e}")
            raise
    
    def prepare_data(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Prepara e limpa os dados para análise
        """
        logger.info("Preparando dados para análise...")
        
        # Renomear colunas para nomes mais limpos (apenas as que existem)
        existing_mapping = {k: v for k, v in self.column_mapping.items() if k in df.columns}
        df = df.rename(existing_mapping)
        
        # Converter tipos de dados (somente colunas presentes)
        columns_to_cast = ['volume_ton', 'frete_brl', 'distancia_km', 'preco_ton', 'preco_ton_km']
        present_columns = [c for c in columns_to_cast if c in df.columns]
        if present_columns:
            df = df.with_columns([pl.col(c).cast(pl.Float64) for c in present_columns])
        
        # Converter data se existir a coluna
        if 'data_faturamento' in df.columns:
            df = df.with_columns([
                pl.col('data_faturamento').dt.month().alias('mes'),
                pl.col('data_faturamento').dt.quarter().alias('trimestre'),
                pl.col('data_faturamento').dt.year().alias('ano')
            ])
        
        # Filtrar dados válidos
        df = df.filter(
            (pl.col('volume_ton') > 0) &
            (pl.col('distancia_km') > 0) &
            (pl.col('frete_brl') > 0)
        )
        
        # Adicionar features calculadas
        df = df.with_columns([
            (pl.col('frete_brl') / (pl.col('volume_ton') * pl.col('distancia_km')))
            .alias('preco_ton_km_calculado'),
            (pl.col('frete_brl') / pl.col('volume_ton'))
            .alias('preco_ton_calculado')
        ])
        
        logger.info(f"Dados preparados. Shape final: {df.shape}")
        return df
    
    def filter_by_date_range(self, df: pl.DataFrame, days: int) -> pl.DataFrame:
        """
        Filtra dados por período específico
        """
        max_date = df.select(pl.col('data_faturamento').max()).item()
        min_date = max_date - pl.duration(days=days)
        
        return df.filter(
            (pl.col('data_faturamento') >= min_date) &
            (pl.col('data_faturamento') <= max_date)
        )
    
    def get_period_dataframes(self, df: pl.DataFrame) -> Dict[str, pl.DataFrame]:
        """
        Gera DataFrames para diferentes períodos (3m, 6m, 12m)
        """
        periods = {
            'df_3m': 90,
            'df_6m': 180,
            'df_12m': 365
        }
        
        result = {}
        for name, days in periods.items():
            result[name] = self.filter_by_date_range(df, days)
            logger.info(f"{name}: {result[name].shape[0]} registros")
        
        return result
    
    def calculate_route_metrics(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Calcula métricas agregadas por rota
        """
        return df.group_by('rota').agg([
            pl.col('volume_ton').sum().alias('volume_total'),
            pl.col('frete_brl').sum().alias('frete_total'),
            pl.col('distancia_km').mean().alias('distancia_media'),
            pl.col('preco_ton_km_calculado').mean().alias('preco_medio_ton_km'),
            pl.col('preco_ton_calculado').mean().alias('preco_medio_ton'),
            pl.len().alias('num_entregas')
        ]).sort('volume_total', descending=True)
    
    def calculate_carrier_metrics(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Calcula métricas agregadas por transportadora
        """
        return df.group_by('transportadora').agg([
            pl.col('volume_ton').sum().alias('volume_total'),
            pl.col('frete_brl').sum().alias('frete_total'),
            pl.col('preco_ton_km_calculado').mean().alias('preco_medio_ton_km'),
            pl.col('rota').n_unique().alias('num_rotas'),
            pl.len().alias('num_entregas')
        ]).sort('volume_total', descending=True)
    
    def export_to_excel(self, df: pl.DataFrame, file_path: str) -> None:
        """
        Exporta DataFrame para Excel usando Polars
        """
        try:
            df.write_excel(file_path)
            logger.info(f"Dados exportados para: {file_path}")
        except Exception as e:
            logger.error(f"Erro ao exportar para Excel: {e}")
            raise
