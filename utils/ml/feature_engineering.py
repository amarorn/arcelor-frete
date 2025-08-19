
# Motor de criação de features para ML
# Pega dados básicos (volume, distância, data) e cria +60 features inteligentes
# Cada grupo tem sua estratégia específica pra capturar padrões dos dados

import polars as pl
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class FreteFeatureEngineer:
    
    def __init__(self):
        self.feature_groups = {
            'temporal': [],
            'geographic': [],
            'operational': [],
            'economic': [],
            'interaction': [],
            'derived': []
        }
        
        self.feature_stats = {}
        self.feature_importance = {}
    
    def create_temporal_features(self, df: pl.DataFrame) -> pl.DataFrame:
        # Features temporais: descobre padrões no tempo
        # Ideia: frete pode ser mais caro em fim de ano, segunda-feira, feriados, etc.
        # Cria indicadores como: dia da semana, estação, alta temporada, feriados
        logger.info("Criando features temporais...")
        
        df_temporal = df.with_columns([
            pl.col('data_faturamento').dt.weekday().alias('dia_semana'),       # 1=segunda, 7=domingo
            pl.col('data_faturamento').dt.ordinal_day().alias('dia_ano'),      # Dia 1-365 do ano
            pl.col('data_faturamento').dt.week().alias('semana_ano'),          # Semana 1-52 do ano
            pl.col('data_faturamento').dt.quarter().alias('trimestre'),        # Trimestre 1-4
            pl.col('data_faturamento').dt.year().alias('ano'),                 # Ano
            
            # Estações do ano (1=verão, 2=outono, 3=inverno, 4=primavera)
            pl.when(pl.col('data_faturamento').dt.month().is_in([12, 1, 2])).then(1)
            .when(pl.col('data_faturamento').dt.month().is_in([3, 4, 5])).then(2)
            .when(pl.col('data_faturamento').dt.month().is_in([6, 7, 8])).then(3)
            .otherwise(4).alias('estacao'),
            
            # Alta temporada: janeiro (férias), julho (férias) e dezembro (natal)
            pl.when(pl.col('data_faturamento').dt.month().is_in([1, 7, 12])).then(1)
            .otherwise(0).alias('mes_alta_temporada'),
            
            # Fim de semana pode ter preço diferente
            pl.when(pl.col('data_faturamento').dt.weekday() >= 5).then(1)
            .otherwise(0).alias('fim_semana'),
            
            pl.col('data_faturamento').dt.month().alias('mes'),
            pl.col('data_faturamento').dt.day().alias('dia_mes'),
            
            # Qual semana do mês (1-5)
            ((pl.col('data_faturamento').dt.day() - 1) // 7 + 1).alias('semana_mes')
        ])
        
        df_temporal = df_temporal.with_columns([
            pl.when(
                (pl.col('mes') == 12) & (pl.col('dia_mes') >= 20)
            ).then(1)
            .when(
                (pl.col('mes') == 1) & (pl.col('dia_mes') <= 10)
            ).then(1)
            .when(
                (pl.col('mes') == 7) & (pl.col('dia_mes') >= 15) & (pl.col('dia_mes') <= 25)
            ).then(1)
            .otherwise(0).alias('periodo_feriado')
        ])
        
        self.feature_groups['temporal'] = [
            'dia_semana', 'dia_ano', 'semana_ano', 'trimestre', 'ano',
            'estacao', 'mes_alta_temporada', 'fim_semana', 'mes', 'dia_mes',
            'semana_mes', 'periodo_feriado'
        ]
        
        return df_temporal
    
    def create_geographic_features(self, df: pl.DataFrame) -> pl.DataFrame:
        # Features geográficas: entende onde tá rolando o frete
        # Ideia: SP→RJ é diferente de AM→RR, regiões têm preços diferentes
        # Cria indicadores de: estados, regiões, complexidade da rota
        logger.info("Criando features geográficas...")
        
        df_geo = df.with_columns([
            # Pega o primeiro estado da rota (origem) e o último (destino)
            pl.col('rota').str.extract(r'([A-Z]{2})').alias('estado_origem'),
            pl.col('rota').str.extract_all(r'([A-Z]{2})').list.last().alias('estado_destino'),
            
            # Classifica por região do Brasil (cada região tem características próprias)
            pl.when(pl.col('rota').str.contains('SP|RJ|MG|ES')).then(pl.lit('Sudeste'))        # Mais industrializado
            .when(pl.col('rota').str.contains('RS|SC|PR')).then(pl.lit('Sul'))               # Agronegócio forte
            .when(pl.col('rota').str.contains('GO|MT|MS|DF')).then(pl.lit('Centro-Oeste'))   # Cerrado/agro
            .when(pl.col('rota').str.contains('BA|SE|AL|PE|PB|RN|CE|PI|MA')).then(pl.lit('Nordeste'))  # Distâncias grandes
            .when(pl.col('rota').str.contains('AM|PA|AP|TO|RO|AC|RR')).then(pl.lit('Norte')) # Amazônia/complexo
            .otherwise(pl.lit('Outras')).alias('regiao'),
            
            # Indicadores de estados específicos
            pl.col('rota').str.contains('SP').cast(pl.Int8).alias('origem_sp'),
            pl.col('rota').str.contains('RJ').cast(pl.Int8).alias('destino_rj'),
            pl.col('rota').str.contains('MG').cast(pl.Int8).alias('passa_mg'),
            pl.col('rota').str.contains('RS').cast(pl.Int8).alias('destino_rs'),
            pl.col('rota').str.contains('PR').cast(pl.Int8).alias('destino_pr'),
            
            # Distância por região
            pl.when(pl.col('rota').str.contains('SP|RJ|MG')).then(pl.lit('curta'))
            .when(pl.col('rota').str.contains('RS|SC|PR|GO|MT')).then(pl.lit('media'))
            .otherwise(pl.lit('longa')).alias('distancia_regional')
        ])
        
        # Features de complexidade da rota
        df_geo = df_geo.with_columns([
            pl.when(pl.col('distancia_km') <= 100).then(pl.lit('local'))
            .when(pl.col('distancia_km') <= 500).then(pl.lit('regional'))
            .when(pl.col('distancia_km') <= 1000).then(pl.lit('interestadual'))
            .otherwise(pl.lit('longa_distancia')).alias('categoria_distancia'),
            
            # Indicador de rota complexa (múltiplos estados)
            pl.when(
                pl.col('estado_origem') != pl.col('estado_destino')
            ).then(1).otherwise(0).alias('rota_interestadual')
        ])
        
        self.feature_groups['geographic'] = [
            'estado_origem', 'estado_destino', 'regiao', 'origem_sp', 'destino_rj',
            'passa_mg', 'destino_rs', 'destino_pr', 'distancia_regional',
            'categoria_distancia', 'rota_interestadual'
        ]
        
        return df_geo
    
    def create_operational_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Cria features operacionais
        """
        logger.info("Criando features operacionais...")
        
        df_op = df.with_columns([
            # Features de modal
            pl.when(pl.col('modal') == 'RODOVIARIO').then(1)
            .when(pl.col('modal') == 'FERROVIARIO').then(2)
            .when(pl.col('modal') == 'AQUAVIARIO').then(3)
            .otherwise(0).alias('modal_encoded'),
            
            # Features de tipo de rodovia
            pl.when(pl.col('tipo_rodovia').str.contains('BR')).then(1)
            .when(pl.col('tipo_rodovia').str.contains('SP')).then(2)
            .when(pl.col('tipo_rodovia').str.contains('RJ')).then(3)
            .otherwise(0).alias('tipo_rodovia_encoded'),
            
            # Features de tipo de veículo
            pl.when(pl.col('tipo_veiculo').str.contains('TRUCK')).then(1)
            .when(pl.col('tipo_veiculo').str.contains('CARRO')).then(2)
            .when(pl.col('tipo_veiculo').str.contains('VAN')).then(3)
            .otherwise(0).alias('tipo_veiculo_encoded'),
            
            # Eficiência operacional
            (pl.col('volume_ton') / pl.col('distancia_km')).alias('eficiencia_volume'),
            (pl.col('frete_brl') / pl.col('distancia_km')).alias('custo_por_km'),
            (pl.col('frete_brl') / pl.col('volume_ton')).alias('custo_por_ton')
        ])
        
        # Features de capacidade
        df_op = df_op.with_columns([
            pl.when(pl.col('volume_ton') <= 10).then(pl.lit('pequena'))
            .when(pl.col('volume_ton') <= 50).then(pl.lit('media'))
            .when(pl.col('volume_ton') <= 100).then(pl.lit('grande'))
            .otherwise(pl.lit('muito_grande')).alias('categoria_volume'),
            
            pl.when(pl.col('distancia_km') <= 100).then(pl.lit('curta'))
            .when(pl.col('distancia_km') <= 500).then(pl.lit('media'))
            .when(pl.col('distancia_km') <= 1000).then(pl.lit('longa'))
            .otherwise(pl.lit('muito_longa')).alias('categoria_distancia_km')
        ])
        
        self.feature_groups['operational'] = [
            'modal_encoded', 'tipo_rodovia_encoded', 'tipo_veiculo_encoded',
            'eficiencia_volume', 'custo_por_km', 'custo_por_ton',
            'categoria_volume', 'categoria_distancia_km'
        ]
        
        return df_op
    
    def create_economic_features(self, df: pl.DataFrame) -> pl.DataFrame:
        # Features econômicas: como o dinheiro se comporta
        # Ideia: calcular vários tipos de preço, ratios, densidades, categorias
        # Cada cálculo captura um aspecto diferente do custo/eficiência
        logger.info("Criando features econômicas...")
        
        df_econ = df.with_columns([
            # Preços básicos calculados de várias formas
            (pl.col('frete_brl') / (pl.col('volume_ton') * pl.col('distancia_km'))).alias('preco_ton_km_calculado'),  # Preço por ton*km
            (pl.col('frete_brl') / pl.col('volume_ton')).alias('preco_por_ton'),                                      # Preço por tonelada
            (pl.col('frete_brl') / pl.col('distancia_km')).alias('preco_por_km'),                                     # Preço por km
            
            # Métricas de eficiência e escala
            (pl.col('volume_ton') * pl.col('distancia_km')).alias('tonelada_km'),        # Total ton*km (escala do frete)
            (pl.col('volume_ton') / pl.col('distancia_km')).alias('densidade_volume'),   # Volume por km (densidade)
            pl.col('volume_ton').pow(2).alias('volume_quadrado'),                        # Volume² (efeitos não-lineares)
            pl.col('distancia_km').pow(2).alias('distancia_quadrada'),                   # Distância² (crescimento quadrático)
            
            # Custos alternativos
            (pl.col('frete_brl') / (pl.col('volume_ton') * pl.col('distancia_km'))).alias('custo_efetivo'),
            (pl.col('frete_brl') / pl.col('volume_ton')).alias('custo_por_tonelada'),
            
            # Categorias de preço (baixo/médio/alto) baseado no preço por ton*km
            pl.when(pl.col('preco_ton_km_calculado') <= 0.5).then(pl.lit('baixo'))
            .when(pl.col('preco_ton_km_calculado') <= 1.0).then(pl.lit('medio'))
            .when(pl.col('preco_ton_km_calculado') <= 2.0).then(pl.lit('alto'))
            .otherwise(pl.lit('muito_alto')).alias('categoria_preco')
        ])
        
        # Features de variação de preço
        df_econ = df_econ.with_columns([
            pl.when(pl.col('preco_ton_km_calculado') <= 0.3).then(1)
            .when(pl.col('preco_ton_km_calculado') <= 0.6).then(2)
            .when(pl.col('preco_ton_km_calculado') <= 1.0).then(3)
            .when(pl.col('preco_ton_km_calculado') <= 1.5).then(4)
            .otherwise(5).alias('faixa_preco')
        ])
        
        self.feature_groups['economic'] = [
            'preco_ton_km_calculado', 'preco_por_ton', 'preco_por_km',
            'tonelada_km', 'densidade_volume', 'volume_quadrado', 'distancia_quadrada',
            'custo_efetivo', 'custo_por_tonelada', 'categoria_preco', 'faixa_preco'
        ]
        
        return df_econ
    
    def create_interaction_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Cria features de interação entre variáveis
        """
        logger.info("Criando features de interação...")
        
        df_inter = df.with_columns([
            # Interações temporais
            (pl.col('mes') * pl.col('volume_ton')).alias('mes_volume'),
            (pl.col('trimestre') * pl.col('distancia_km')).alias('trimestre_distancia'),
            (pl.col('ano') * pl.col('preco_por_ton')).alias('ano_preco'),
            (pl.col('dia_semana') * pl.col('volume_ton')).alias('dia_semana_volume'),
            
            # Interações geográficas
            (pl.col('origem_sp').cast(pl.Float64) * pl.col('distancia_km')).alias('sp_distancia'),
            (pl.col('destino_rj').cast(pl.Float64) * pl.col('volume_ton')).alias('rj_volume'),
            (pl.col('passa_mg').cast(pl.Float64) * pl.col('preco_por_km')).alias('mg_preco'),
            
            # Interações operacionais
            (pl.col('modal_encoded') * pl.col('distancia_km')).alias('modal_distancia'),
            (pl.col('tipo_rodovia_encoded') * pl.col('volume_ton')).alias('rodovia_volume'),
            (pl.col('tipo_veiculo_encoded') * pl.col('preco_por_ton')).alias('veiculo_preco'),
            
            # Interações econômicas
            (pl.col('volume_ton') * pl.col('distancia_km') * pl.col('mes')).alias('volume_distancia_mes'),
            (pl.col('frete_brl') / (pl.col('volume_ton') * pl.col('distancia_km') * pl.col('trimestre'))).alias('custo_trimestral'),
            (pl.col('preco_ton_km_calculado') * pl.col('estacao')).alias('preco_estacao')
        ])
        
        self.feature_groups['interaction'] = [
            'mes_volume', 'trimestre_distancia', 'ano_preco', 'dia_semana_volume',
            'sp_distancia', 'rj_volume', 'mg_preco', 'modal_distancia',
            'rodovia_volume', 'veiculo_preco', 'volume_distancia_mes',
            'custo_trimestral', 'preco_estacao'
        ]
        
        return df_inter
    
    def create_derived_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Cria features derivadas e transformadas
        """
        logger.info("Criando features derivadas...")
        
        df_derived = df.with_columns([
            # Transformações logarítmicas
            pl.col('volume_ton').log().alias('log_volume'),
            pl.col('distancia_km').log().alias('log_distancia'),
            pl.col('frete_brl').log().alias('log_frete'),
            
            # Transformações de raiz quadrada
            pl.col('volume_ton').sqrt().alias('sqrt_volume'),
            pl.col('distancia_km').sqrt().alias('sqrt_distancia'),
            
            # Transformações recíprocas
            (1 / pl.col('volume_ton')).alias('inv_volume'),
            (1 / pl.col('distancia_km')).alias('inv_distancia'),
            
            # Features de razão
            (pl.col('frete_brl') / pl.col('volume_ton')).alias('frete_volume_ratio'),
            (pl.col('distancia_km') / pl.col('volume_ton')).alias('distancia_volume_ratio'),
            
            # Features de diferença
            (pl.col('distancia_km') - pl.col('volume_ton')).alias('distancia_volume_diff'),
            (pl.col('frete_brl') - pl.col('volume_ton')).alias('frete_volume_diff')
        ])
        
        # Features de percentil
        df_derived = df_derived.with_columns([
            pl.when(pl.col('volume_ton') <= pl.col('volume_ton').quantile(0.25)).then(pl.lit('baixo'))
            .when(pl.col('volume_ton') <= pl.col('volume_ton').quantile(0.75)).then(pl.lit('medio'))
            .otherwise(pl.lit('alto')).alias('percentil_volume'),
            
            pl.when(pl.col('distancia_km') <= pl.col('distancia_km').quantile(0.25)).then(pl.lit('baixo'))
            .when(pl.col('distancia_km') <= pl.col('distancia_km').quantile(0.75)).then(pl.lit('medio'))
            .otherwise(pl.lit('alto')).alias('percentil_distancia')
        ])
        
        self.feature_groups['derived'] = [
            'log_volume', 'log_distancia', 'log_frete', 'sqrt_volume', 'sqrt_distancia',
            'inv_volume', 'inv_distancia', 'frete_volume_ratio', 'distancia_volume_ratio',
            'distancia_volume_diff', 'frete_volume_diff', 'percentil_volume', 'percentil_distancia'
        ]
        
        return df_derived
    
    def create_all_features(self, df: pl.DataFrame) -> pl.DataFrame:
        logger.info("Criando todas as features...")
        
        df_features = df.pipe(self.create_temporal_features)\
                       .pipe(self.create_geographic_features)\
                       .pipe(self.create_operational_features)\
                       .pipe(self.create_economic_features)\
                       .pipe(self.create_interaction_features)\
                       .pipe(self.create_derived_features)
        
        logger.info(f"Features criadas. Shape final: {df_features.shape}")
        logger.info(f"Total de features por grupo:")
        for group, features in self.feature_groups.items():
            logger.info(f"  {group}: {len(features)}")
        
        return df_features
    
    def analyze_feature_importance(self, df: pl.DataFrame, target_col: str = 'preco_ton_km_calculado') -> Dict[str, float]:
        # Análise de importância: descobre quais features são mais úteis
        # Usa correlação pra medir: quanto maior a correlação, mais a feature "explica" o preço
        # Correlação: -1 a +1, sendo 0=sem relação, +1=correlação perfeita positiva
        logger.info("Analisando importância das features...")
        
        # Junta todas as features criadas dos 6 grupos
        all_features = []
        for features in self.feature_groups.values():
            all_features.extend(features)
        
        # Só pega features numéricas (correlação só funciona com números)
        numeric_features = []
        for col in all_features:
            if col in df.columns and df[col].dtype in [pl.Float64, pl.Int64, pl.Int8]:
                numeric_features.append(col)
        
        # Calcula correlação de cada feature com o preço alvo
        correlations = {}
        for feature in numeric_features:
            try:
                corr = df.select([
                    pl.corr(pl.col(feature), pl.col(target_col))  # Correlação entre feature e preço
                ])
                
                if corr.shape[0] > 0 and corr[0, 0] is not None:
                    correlations[feature] = abs(corr[0, 0])  # Valor absoluto (só importa a força, não direção)
            except:
                continue  # Se der erro, ignora essa feature
        
        # Ordena por correlação (maior primeiro)
        sorted_correlations = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
        
        # Salva estatísticas pra relatório
        self.feature_importance = dict(sorted_correlations)
        self.feature_stats = {
            'total_features': len(numeric_features),
            'features_analyzed': len(correlations),
            'top_correlations': sorted_correlations[:10]  # Top 10 features mais importantes
        }
        
        return self.feature_importance
    
    def get_feature_summary(self) -> Dict[str, Any]:
        """
        Retorna resumo das features criadas
        """
        return {
            'feature_groups': self.feature_groups,
            'feature_stats': self.feature_stats,
            'feature_importance': self.feature_importance
        }
