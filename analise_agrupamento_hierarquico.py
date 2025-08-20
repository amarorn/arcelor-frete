#!/usr/bin/env python3
"""
Script para análise de agrupamento hierárquico dos MESES DE MAIO, JUNHO E JULHO DE 2025
Implementa estrutura: Centro Origem > Sub-categoria > Rota Específica
Filtra dados por período específico para análise trimestral Q2 2025
INCLUI FILTRO DE VOLUME MÍNIMO para cálculo do benchmark (remove rotas com baixo volume)
"""

import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
from datetime import datetime, timedelta
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, lit, expr, udf, struct, row_number, desc, to_date, date_sub, current_date
from pyspark.sql.types import StringType, DoubleType, BooleanType, DateType
from pyspark.sql.window import Window

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HierarchicalGroupingAnalyzer:
    """
    Analisador de agrupamento hierárquico para dashboard dos meses de maio, junho e julho de 2025
    INCLUI FILTRO DE VOLUME MÍNIMO para cálculo do benchmark, removendo rotas com baixo volume
    """
    
    def __init__(self, spark_session):
        self.spark = spark_session
        
        # Configurações de análise
        self.volume_minimo_benchmark = 1000  # 1000 toneladas como volume mínimo para benchmark
        self.volume_minimo_analise = 100     # 100 toneladas como volume mínimo para análise geral
        
        # Configurações para cálculo inteligente do benchmark
        self.percentil_benchmark = 75        # Percentil para definir volume mínimo do benchmark
        self.min_rotas_benchmark = 10       # Mínimo de rotas para considerar benchmark válido
        self.max_rotas_benchmark = 100      # Máximo de rotas para benchmark (evitar muito ruído)
        self.adaptativo_benchmark = True    # Se deve calcular volume mínimo automaticamente
        
        self.microregion_mapping = [
            ('JOÃO MONLEVADE', 'JOÃO MONLEVADE'),
            ('USINA MONLEVADE', 'JOÃO MONLEVADE'),
            ('USINA', 'ITABIRA'),   
            ('ITABIRA', 'ITABIRA'),
            ('BELO HORIZONTE', 'BELO HORIZONTE'),
            ('CONTAGEM', 'CONTAGEM'),
            ('SABARÁ', 'SABARÁ'),
            ('SANTA LUZIA', 'SANTA LUZIA'),
            ('NOVA LIMA', 'NOVA LIMA'),
            ('BRUMADINHO', 'BRUMADINHO'),
            ('IBIRITÉ', 'IBIRITÉ'),
            ('BETIM', 'BETIM'),
            ('LAGOA SANTA', 'LAGOA SANTA'),
            ('VESPASIANO', 'VESPASIANO'),
            ('RIBEIRÃO DAS NEVES', 'RIBEIRÃO DAS NEVES'),
            ('CAETÉ', 'CAETÉ'),
            ('SÃO JOSÉ DA LAPA', 'SÃO JOSÉ DA LAPA'),
            ('FLORESTAL', 'FLORESTAL'),
            ('JABOTICATUBAS', 'JABOTICATUBAS'),
            ('MATEUS LEME', 'MATEUS LEME'),
            ('IGARAPÉ', 'IGARAPÉ'),
            ('SÃO JOAQUIM DE BICAS', 'SÃO JOAQUIM DE BICAS'),
            ('SÃO JOSÉ DO GOIABAL', 'SÃO JOSÉ DO GOIABAL'),
            ('MARAVILHAS', 'MARAVILHAS'),
            ('ONÇA DE PITANGUI', 'ONÇA DE PITANGUI'),
            ('PARÁ DE MINAS', 'PARÁ DE MINAS'),
            ('PITANGUI', 'PITANGUI'),
            ('CONCEIÇÃO DO MATO DENTRO', 'CONCEIÇÃO DO MATO DENTRO'),
            ('SANTANA DO PARAÍSO', 'SANTANA DO PARAÍSO'),
            ('CORONEL FABRICIANO', 'CORONEL FABRICIANO'),
            ('IPATINGA', 'IPATINGA'),
            ('TIMÓTEO', 'TIMÓTEO'),
            ('CARATINGA', 'CARATINGA'),
            ('INHAPIM', 'INHAPIM'),
            ('GOVERNADOR VALADARES', 'GOVERNADOR VALADARES'),
            ('TEÓFILO OTONI', 'TEÓFILO OTONI'),
            ('NANUC', 'NANUC'),
            ('SÃO JOÃO DEL REI', 'SÃO JOÃO DEL REI'),
            ('BARBACENA', 'BARBACENA'),
            ('CONSELHEIRO LAFAIETE', 'CONSELHEIRO LAFAIETE'),
            ('OURO PRETO', 'OURO PRETO'),
            ('MARIANA', 'MARIANA')
        ]
    
    def configurar_volumes_minimos(self, volume_minimo_analise=100, volume_minimo_benchmark=1000):
        """
        Configura os volumes mínimos para análise e benchmark
        
        Args:
            volume_minimo_analise (int): Volume mínimo em toneladas para incluir na análise geral
            volume_minimo_benchmark (int): Volume mínimo em toneladas para incluir no cálculo do benchmark
        """
        self.volume_minimo_analise = volume_minimo_analise
        self.volume_minimo_benchmark = volume_minimo_benchmark
        
        logger.info(f"Volumes mínimos configurados:")
        logger.info(f"  - Análise geral: {self.volume_minimo_analise:,} ton")
        logger.info(f"  - Benchmark: {self.volume_minimo_benchmark:,} ton")
    
    def configurar_benchmark_inteligente(self, 
                                       percentil_benchmark=75,
                                       min_rotas_benchmark=10,
                                       max_rotas_benchmark=100,
                                       adaptativo_benchmark=True):
        """
        Configura parâmetros para cálculo inteligente do volume mínimo do benchmark
        
        Args:
            percentil_benchmark (int): Percentil para definir volume mínimo (25, 50, 75, 90)
            min_rotas_benchmark (int): Mínimo de rotas para considerar benchmark válido
            max_rotas_benchmark (int): Máximo de rotas para benchmark (evitar muito ruído)
            adaptativo_benchmark (bool): Se deve calcular volume mínimo automaticamente
        """
        self.percentil_benchmark = percentil_benchmark
        self.min_rotas_benchmark = min_rotas_benchmark
        self.max_rotas_benchmark = max_rotas_benchmark
        self.adaptativo_benchmark = adaptativo_benchmark
        
        logger.info(f"Benchmark inteligente configurado:")
        logger.info(f"  - Percentil: {self.percentil_benchmark}%")
        logger.info(f"  - Mínimo de rotas: {self.min_rotas_benchmark}")
        logger.info(f"  - Máximo de rotas: {self.max_rotas_benchmark}")
        logger.info(f"  - Modo adaptativo: {'Ativado' if self.adaptativo_benchmark else 'Desativado'}")
    
    def calcular_volume_minimo_inteligente(self, df):
        """
        Calcula o volume mínimo do benchmark de forma inteligente baseado nos dados
        
        Args:
            df: DataFrame com dados de volume por rota
            
        Returns:
            float: Volume mínimo calculado inteligentemente
        """
        logger.info("Calculando volume mínimo do benchmark de forma inteligente...")
        
        try:
            # Calcular volume total por rota
            df_volume_por_rota = df.groupBy('centro_origem', 'rota_municipio').agg(
                {'volume_ton': 'sum'}
            ).withColumnRenamed('sum(volume_ton)', 'volume_total_rota')
            
            # Converter para pandas para análise estatística
            df_pandas = df_volume_por_rota.toPandas()
            
            if df_pandas.empty:
                logger.warning("Dados vazios para cálculo do volume mínimo")
                return self.volume_minimo_benchmark
            
            # Estatísticas descritivas dos volumes
            volumes = df_pandas['volume_total_rota'].dropna()
            
            if len(volumes) == 0:
                logger.warning("Nenhum volume válido encontrado")
                return self.volume_minimo_benchmark
            
            # Calcular percentis
            percentil_25 = volumes.quantile(0.25)
            percentil_50 = volumes.quantile(0.50)
            percentil_75 = volumes.quantile(0.75)
            percentil_90 = volumes.quantile(0.90)
            
            # Calcular média e desvio padrão
            media = volumes.mean()
            desvio_padrao = volumes.std()
            
            # Calcular volume mínimo baseado no percentil configurado
            if self.percentil_benchmark == 25:
                volume_candidato = percentil_25
            elif self.percentil_benchmark == 50:
                volume_candidato = percentil_50
            elif self.percentil_benchmark == 75:
                volume_candidato = percentil_75
            elif self.percentil_benchmark == 90:
                volume_candidato = percentil_90
            else:
                # Interpolação linear entre percentis
                if self.percentil_benchmark <= 50:
                    peso = (self.percentil_benchmark - 25) / 25
                    volume_candidato = percentil_25 + peso * (percentil_50 - percentil_25)
                else:
                    peso = (self.percentil_benchmark - 50) / 25
                    volume_candidato = percentil_50 + peso * (percentil_75 - percentil_50)
            
            # Verificar quantas rotas seriam incluídas com esse volume
            rotas_incluidas = len(volumes[volumes >= volume_candidato])
            
            logger.info(f"Análise estatística dos volumes:")
            logger.info(f"  - Total de rotas: {len(volumes)}")
            logger.info(f"  - Média: {media:,.1f} ton")
            logger.info(f"  - Desvio padrão: {desvio_padrao:,.1f} ton")
            logger.info(f"  - Percentil 25: {percentil_25:,.1f} ton")
            logger.info(f"  - Percentil 50: {percentil_50:,.1f} ton")
            logger.info(f"  - Percentil 75: {percentil_75:,.1f} ton")
            logger.info(f"  - Percentil 90: {percentil_90:,.1f} ton")
            
            # Ajustar volume baseado no número de rotas desejado
            volume_final = self._ajustar_volume_por_rotas(volumes, volume_candidato, rotas_incluidas)
            
            logger.info(f"Volume mínimo calculado inteligentemente:")
            logger.info(f"  - Volume candidato: {volume_candidato:,.1f} ton")
            logger.info(f"  - Rotas incluídas: {rotas_incluidas}")
            logger.info(f"  - Volume final ajustado: {volume_final:,.1f} ton")
            
            return volume_final
            
        except Exception as e:
            logger.error(f"Erro ao calcular volume mínimo inteligente: {str(e)}")
            logger.info("Usando volume mínimo padrão")
            return self.volume_minimo_benchmark
    
    def _ajustar_volume_por_rotas(self, volumes, volume_candidato, rotas_incluidas):
        """
        Ajusta o volume mínimo para garantir número adequado de rotas para benchmark
        
        Args:
            volumes: Series com volumes das rotas
            volume_candidato: Volume candidato inicial
            rotas_incluidas: Número de rotas incluídas com volume candidato
            
        Returns:
            float: Volume ajustado
        """
        # Se já está no range desejado, usar o candidato
        if self.min_rotas_benchmark <= rotas_incluidas <= self.max_rotas_benchmark:
            return volume_candidato
        
        # Se muito poucas rotas, diminuir o volume
        if rotas_incluidas < self.min_rotas_benchmark:
            # Buscar volume que inclua pelo menos min_rotas_benchmark
            volumes_ordenados = sorted(volumes, reverse=True)
            if len(volumes_ordenados) >= self.min_rotas_benchmark:
                volume_ajustado = volumes_ordenados[self.min_rotas_benchmark - 1]
                logger.info(f"Ajustando volume para incluir pelo menos {self.min_rotas_benchmark} rotas")
                return volume_ajustado
            else:
                # Se não há rotas suficientes, usar o menor volume
                return volumes.min()
        
        # Se muitas rotas, aumentar o volume
        if rotas_incluidas > self.max_rotas_benchmark:
            # Buscar volume que inclua no máximo max_rotas_benchmark
            volumes_ordenados = sorted(volumes, reverse=True)
            if len(volumes_ordenados) >= self.max_rotas_benchmark:
                volume_ajustado = volumes_ordenados[self.max_rotas_benchmark - 1]
                logger.info(f"Ajustando volume para incluir no máximo {self.max_rotas_benchmark} rotas")
                return volume_ajustado
            else:
                return volume_candidato
        
        return volume_candidato
    
    def analisar_distribuicao_volumes(self, df):
        """
        Analisa a distribuição dos volumes para otimização do benchmark
        
        Args:
            df: DataFrame com dados de volume
            
        Returns:
            dict: Análise da distribuição dos volumes
        """
        logger.info("Analisando distribuição dos volumes para otimização...")
        
        try:
            # Calcular volume total por rota
            df_volume_por_rota = df.groupBy('centro_origem', 'rota_municipio').agg(
                {'volume_ton': 'sum'}
            ).withColumnRenamed('sum(volume_ton)', 'volume_total_rota')
            
            # Converter para pandas
            df_pandas = df_volume_por_rota.toPandas()
            volumes = df_pandas['volume_total_rota'].dropna()
            
            if len(volumes) == 0:
                return {"erro": "Nenhum volume válido encontrado"}
            
            # Análise estatística completa
            analise = {
                "total_rotas": len(volumes),
                "media": float(volumes.mean()),
                "mediana": float(volumes.median()),
                "desvio_padrao": float(volumes.std()),
                "minimo": float(volumes.min()),
                "maximo": float(volumes.max()),
                "percentis": {
                    "25": float(volumes.quantile(0.25)),
                    "50": float(volumes.quantile(0.50)),
                    "75": float(volumes.quantile(0.75)),
                    "90": float(volumes.quantile(0.90)),
                    "95": float(volumes.quantile(0.95))
                },
                "faixas_volume": {
                    "0-100": len(volumes[(volumes >= 0) & (volumes < 100)]),
                    "100-500": len(volumes[(volumes >= 100) & (volumes < 500)]),
                    "500-1000": len(volumes[(volumes >= 500) & (volumes < 1000)]),
                    "1000-5000": len(volumes[(volumes >= 1000) & (volumes < 5000)]),
                    "5000+": len(volumes[volumes >= 5000])
                }
            }
            
            # Recomendações baseadas na análise
            analise["recomendacoes"] = self._gerar_recomendacoes_volume(analise)
            
            logger.info(f"Análise de distribuição concluída:")
            logger.info(f"  - Total de rotas: {analise['total_rotas']}")
            logger.info(f"  - Média: {analise['media']:,.1f} ton")
            logger.info(f"  - Mediana: {analise['mediana']:,.1f} ton")
            
            return analise
            
        except Exception as e:
            logger.error(f"Erro na análise de distribuição: {str(e)}")
            return {"erro": str(e)}
    
    def _gerar_recomendacoes_volume(self, analise):
        """
        Gera recomendações baseadas na análise da distribuição dos volumes
        
        Args:
            analise: Dicionário com análise estatística
            
        Returns:
            list: Lista de recomendações
        """
        recomendacoes = []
        
        total_rotas = analise["total_rotas"]
        
        # Analisar distribuição por faixas
        faixas = analise["faixas_volume"]
        
        # Se muitas rotas com volume baixo, sugerir ajuste
        if faixas["0-100"] > total_rotas * 0.3:
            recomendacoes.append("Muitas rotas com volume baixo (< 100 ton) - considerar filtro mais rigoroso")
        
        # Se poucas rotas com volume alto, sugerir percentil menor
        if faixas["1000+"] < total_rotas * 0.1:
            recomendacoes.append("Poucas rotas com volume alto (> 1000 ton) - usar percentil 50 ou 25")
        
        # Se distribuição muito desigual, sugerir percentil intermediário
        if faixas["500-1000"] < total_rotas * 0.05:
            recomendacoes.append("Distribuição muito desigual - usar percentil 75 para equilibrar")
        
        # Se muitas rotas qualificadas, sugerir percentil maior
        if faixas["1000+"] > total_rotas * 0.4:
            recomendacoes.append("Muitas rotas qualificadas - usar percentil 90 para maior seletividade")
        
        # Recomendação de percentil ideal
        if faixas["1000+"] >= total_rotas * 0.2:
            recomendacoes.append("Usar percentil 75 - boa relação entre qualidade e quantidade")
        elif faixas["1000+"] >= total_rotas * 0.1:
            recomendacoes.append("Usar percentil 50 - equilibrar quantidade e qualidade")
        else:
            recomendacoes.append("Usar percentil 25 - priorizar quantidade de rotas para benchmark")
        
        return recomendacoes
    
    def load_data_from_excel(self, excel_path: str):
        """
        Carrega dados do Excel usando pandas e converte para Spark DataFrame
        """
        logger.info(f"Carregando dados de: {excel_path}")
        
        try:
            # Tentar ler com Spark Excel primeiro
            df = self.spark.read.format("com.crealytics.spark.excel") \
                .option("header", "true") \
                .option("inferSchema", "true") \
                .load(excel_path)
            logger.info(f"Dados carregados com Spark Excel: {df.count()} linhas")
            return df
        except Exception as e:
            logger.info(f"Spark Excel não disponível, usando pandas: {str(e)}")
            
            # Fallback para pandas
            df_pandas = pd.read_excel(excel_path)
            logger.info(f"Dados carregados com pandas: {len(df_pandas)} linhas")
            
            # Tratar tipos de dados antes da conversão
            for col in df_pandas.columns:
                if df_pandas[col].dtype == 'object':
                    df_pandas[col] = df_pandas[col].astype(str)
                elif df_pandas[col].dtype == 'datetime64[ns]':
                    df_pandas[col] = df_pandas[col].dt.strftime('%Y-%m-%d')
            
            # Renomear colunas para evitar problemas com pontos no Spark
            column_mapping = {
                '00.dt_doc_faturamento': 'data_faturamento',
                '00.nm_centro_origem_unidade': 'centro_origem',
                'dc_centro_unidade_descricao': 'descricao_centro',
                '00.nm_modal': 'modal',
                'nm_tipo_rodovia': 'tipo_rodovia',
                'nm_veiculo': 'tipo_veiculo',
                '01.Rota_UFOrigem_UFDestino': 'rota_uf',
                '01.Rota_MesoOrigem_MesoDestino': 'rota_mesoregiao',
                '01.Rota_MicroOrigem_MicroDestino': 'rota_microregiao',
                '01.Rota_MuniOrigem_MuniDestino': 'rota_municipio',
                'id_transportadora': 'transportadora',
                'Date': 'data',
                'id_fatura': 'fatura',
                'id_transporte': 'transporte',
                '02.01.00 - Volume (ton)': 'volume_ton',
                '02.01.01 - Frete Geral (BRL)': 'frete_brl',
                '02.01.02 - DISTANCIA (KM)': 'distancia_km',
                '02.03.00 - Preço_Frete Geral (BRL) / TON': 'preco_ton',
                '02.03.02 - Preço_Frete Geral (BRL / TON / KM)': 'preco_ton_km'
            }
            
            df_pandas = df_pandas.rename(columns=column_mapping)
            
            # Converter para Spark DataFrame
            df = self.spark.createDataFrame(df_pandas)
            logger.info(f"Convertido para Spark DataFrame: {df.count()} linhas")
            return df
    
    def filter_may_june_july_2025(self, df):
        """
        Filtra dados dos meses de maio, junho e julho de 2025
        """
        logger.info("Filtrando dados dos meses de maio, junho e julho de 2025...")
        
        # Converter coluna de data para DateType com formato específico
        # O formato das datas inclui horário (00:00:00), então precisamos tratar isso
        df_with_date = df.withColumn(
            'data_faturamento_date',
            to_date(col('data_faturamento'), 'yyyy-MM-dd HH:mm:ss')
        )
        
        # Se falhar, tentar formato sem horário
        df_with_date = df_with_date.withColumn(
            'data_faturamento_date',
            when(col('data_faturamento_date').isNull(),
                 to_date(col('data_faturamento'), 'yyyy-MM-dd'))
            .otherwise(col('data_faturamento_date'))
        )
        
        # Filtrar dados dos meses específicos: maio, junho e julho de 2025
        df_filtered = df_with_date.filter(
            (col('data_faturamento_date') >= '2025-05-01') &
            (col('data_faturamento_date') <= '2025-07-31')
        )
        
        # Remover coluna temporária
        df_filtered = df_filtered.drop('data_faturamento_date')
        
        total_records = df.count()
        filtered_records = df_filtered.count()
        
        logger.info(f"Total de registros: {total_records:,}")
        logger.info(f"Registros de maio, junho e julho de 2025: {filtered_records:,}")
        logger.info(f"Redução: {((total_records - filtered_records) / total_records * 100):.1f}%")
        
        # Verificar período dos dados filtrados
        if filtered_records > 0:
            min_date = df_filtered.agg({"data_faturamento": "min"}).collect()[0][0]
            max_date = df_filtered.agg({"data_faturamento": "max"}).collect()[0][0]
            logger.info(f"Período dos dados filtrados: {min_date} a {max_date}")
            
            # Contar registros por mês
            df_monthly = df_filtered.withColumn(
                'mes',
                expr("substring(data_faturamento, 6, 2)")
            )
            
            may_count = df_monthly.filter(col('mes') == '05').count()
            june_count = df_monthly.filter(col('mes') == '06').count()
            july_count = df_monthly.filter(col('mes') == '07').count()
            
            logger.info(f"Distribuição por mês:")
            logger.info(f"  - Maio 2025: {may_count:,} registros")
            logger.info(f"  - Junho 2025: {june_count:,} registros")
            logger.info(f"  - Julho 2025: {july_count:,} registros")
        
        return df_filtered
    
    def prepare_data_hierarchical(self, df):
        """
        Prepara dados para agrupamento hierárquico
        """
        logger.info("Preparando dados para agrupamento hierárquico...")
        
        # Contar rotas únicas ANTES dos filtros
        total_rotas_antes = df.select('centro_origem', 'rota_municipio').dropDuplicates().count()
        logger.info(f"Total de rotas únicas ANTES dos filtros: {total_rotas_antes}")
        
        # Mapear colunas e filtrar dados válidos
        df_processed = df.select(
            df['centro_origem'].alias('centro_origem'),
            df['volume_ton'].cast('double').alias('volume_ton'),
            df['distancia_km'].cast('double').alias('distancia_km'),
            df['frete_brl'].cast('double').alias('custo_sup_tku'),
            df['data_faturamento'].alias('data_faturamento'),
            df['rota_mesoregiao'].alias('rota_mesoregiao'),
            df['rota_microregiao'].alias('rota_microregiao'),
            df['rota_municipio'].alias('rota_municipio')
        )
        
        # Contar rotas únicas APÓS seleção de colunas
        total_rotas_apos_selecao = df_processed.select('centro_origem', 'rota_municipio').dropDuplicates().count()
        logger.info(f"Total de rotas únicas APÓS seleção de colunas: {total_rotas_apos_selecao}")
        
        # Aplicar filtros básicos - INCLUINDO ROTAS COM VOLUME/DISTÂNCIA ZERO
        df_processed = df_processed.filter(
            # Incluir todas as rotas, mesmo com volume ou distância zero
            # Apenas remover registros completamente inválidos
            (df_processed['centro_origem'].isNotNull()) & 
            (df_processed['rota_municipio'].isNotNull())
        )
        
        # Log do filtro básico
        total_apos_filtro_basico = df_processed.count()
        total_rotas_apos_filtro_basico = df_processed.select('centro_origem', 'rota_municipio').dropDuplicates().count()
        
        logger.info(f"Filtro básico aplicado (INCLUINDO ROTAS COM VOLUME/DISTÂNCIA ZERO):")
        logger.info(f"  - Apenas centro_origem e rota_municipio não nulos")
        logger.info(f"  - Total de registros após filtro básico: {total_apos_filtro_basico:,}")
        logger.info(f"  - Total de rotas únicas após filtro básico: {total_rotas_apos_filtro_basico}")
        
        # Verificar se conseguimos recuperar as rotas perdidas
        if total_rotas_apos_filtro_basico >= 89:
            logger.info(f"✅ Sucesso! Todas as 89 rotas foram incluídas!")
        else:
            logger.warning(f"⚠️ Ainda faltam {89 - total_rotas_apos_filtro_basico} rotas")
        
        # Calcular preço por TKU - TRATANDO CASOS DE VOLUME/DISTÂNCIA ZERO
        df_processed = df_processed.withColumn(
            'preco_ton_km',
            when(
                (df_processed['volume_ton'] > 0) & (df_processed['distancia_km'] > 0),
                df_processed['custo_sup_tku'] / (df_processed['volume_ton'] * df_processed['distancia_km'])
            ).otherwise(0.0)  # Valor padrão para rotas com volume ou distância zero
        )
        
        # Adicionar coluna de status para rotas com problemas de dados
        df_processed = df_processed.withColumn(
            'status_dados',
            when(
                (df_processed['volume_ton'] > 0) & (df_processed['distancia_km'] > 0),
                'Dados_Válidos'
            ).when(
                df_processed['volume_ton'] <= 0,
                'Volume_Zero_ou_Negativo'
            ).when(
                df_processed['distancia_km'] <= 0,
                'Distancia_Zero_ou_Negativa'
            ).otherwise('Dados_Inválidos')
        )
        
        # Log do status dos dados
        logger.info(f"Status dos dados por rota:")
        df_status = df_processed.select('centro_origem', 'rota_municipio', 'status_dados').dropDuplicates()
        for status in ['Dados_Válidos', 'Volume_Zero_ou_Negativo', 'Distancia_Zero_ou_Negativa', 'Dados_Inválidos']:
            count = df_status.filter(col('status_dados') == status).count()
            if count > 0:
                logger.info(f"  - {status}: {count} rotas")
        
        # Extrair micro-região usando UDF
        def extract_microregion(location):
            if not location or location is None or str(location).strip() == '':
                return "UNKNOWN"
            location_str = str(location).upper().strip()
            microregion_mapping = [
                ('JOÃO MONLEVADE', 'JOÃO MONLEVADE'),
                ('USINA MONLEVADE', 'JOÃO MONLEVADE'),
                ('USINA', 'ITABIRA'),
                ('ITABIRA', 'ITABIRA'),
                ('BELO HORIZONTE', 'BELO HORIZONTE'),
                ('CONTAGEM', 'CONTAGEM'),
                ('SABARÁ', 'SABARÁ'),
                ('SANTA LUZIA', 'SANTA LUZIA'),
                ('NOVA LIMA', 'NOVA LIMA'),
                ('BRUMADINHO', 'BRUMADINHO'),
                ('IBIRITÉ', 'IBIRITÉ'),
                ('BETIM', 'BETIM'),
                ('LAGOA SANTA', 'LAGOA SANTA'),
                ('VESPASIANO', 'VESPASIANO'),
                ('RIBEIRÃO DAS NEVES', 'RIBEIRÃO DAS NEVES'),
                ('CAETÉ', 'CAETÉ'),
                ('SÃO JOSÉ DA LAPA', 'SÃO JOSÉ DA LAPA'),
                ('FLORESTAL', 'FLORESTAL'),
                ('JABOTICATUBAS', 'JABOTICATUBAS'),
                ('MATEUS LEME', 'MATEUS LEME'),
                ('IGARAPÉ', 'IGARAPÉ'),
                ('SÃO JOAQUIM DE BICAS', 'SÃO JOAQUIM DE BICAS'),
                ('SÃO JOSÉ DO GOIABAL', 'SÃO JOSÉ DO GOIABAL'),
                ('MARAVILHAS', 'MARAVILHAS'),
                ('ONÇA DE PITANGUI', 'ONÇA DE PITANGUI'),
                ('PARÁ DE MINAS', 'PARÁ DE MINAS'),
                ('PITANGUI', 'PITANGUI'),
                ('CONCEIÇÃO DO MATO DENTRO', 'CONCEIÇÃO DO MATO DENTRO'),
                ('SANTANA DO PARAÍSO', 'SANTANA DO PARAÍSO'),
                ('CORONEL FABRICIANO', 'CORONEL FABRICIANO'),
                ('IPATINGA', 'IPATINGA'),
                ('TIMÓTEO', 'TIMÓTEO'),
                ('CARATINGA', 'CARATINGA'),
                ('INHAPIM', 'INHAPIM'),
                ('GOVERNADOR VALADARES', 'GOVERNADOR VALADARES'),
                ('TEÓFILO OTONI', 'TEÓFILO OTONI'),
                ('NANUC', 'NANUC'),
                ('SÃO JOÃO DEL REI', 'SÃO JOÃO DEL REI'),
                ('BARBACENA', 'BARBACENA'),
                ('CONSELHEIRO LAFAIETE', 'CONSELHEIRO LAFAIETE'),
                ('OURO PRETO', 'OURO PRETO'),
                ('MARIANA', 'MARIANA')
            ]
            for key, microregion in microregion_mapping:
                if key in location_str:
                    return microregion
            return location_str
        
        extract_udf = udf(extract_microregion, StringType())
        
        df_processed = df_processed.withColumn(
            'microregiao_origem', 
            extract_udf(col('centro_origem'))
        )
        
        # Criar estrutura hierárquica
        df_processed = df_processed.withColumn(
            'nivel_1_centro_origem',
            when(col('centro_origem').like('%USINA%'), 'Usina')
            .when(col('centro_origem').like('%JOÃO MONLEVADE%'), 'João Monlevade')
            .when(col('centro_origem').like('%ITABIRA%'), 'Itabira')
            .otherwise(col('centro_origem'))
        )
        
        # Criar sub-categoria (nível 2)
        df_processed = df_processed.withColumn(
            'nivel_2_subcategoria',
            when(col('rota_microregiao').isNotNull(), col('rota_microregiao'))
            .otherwise(col('rota_municipio'))
        )
        
        # Criar rota específica (nível 3)
        df_processed = df_processed.withColumn(
            'nivel_3_rota_especifica',
            when(col('rota_municipio').isNotNull(), col('rota_municipio'))
            .otherwise(col('rota_microregiao'))
        )
        
        # Contar rotas únicas APÓS criação da hierarquia
        total_rotas_apos_hierarquia = df_processed.select('nivel_1_centro_origem', 'nivel_2_subcategoria', 'nivel_3_rota_especifica').dropDuplicates().count()
        logger.info(f"Total de rotas únicas APÓS criação da hierarquia: {total_rotas_apos_hierarquia}")
        
        logger.info(f"Dados preparados para hierarquia: {df_processed.count()} linhas válidas")
        return df_processed
    
    def create_hierarchical_structure(self, df):
        """
        Cria estrutura hierárquica similar ao dashboard da imagem
        """
        logger.info("Criando estrutura hierárquica...")
        
        # Contar rotas únicas ANTES dos agrupamentos
        total_rotas_inicio = df.select('nivel_1_centro_origem', 'nivel_2_subcategoria', 'nivel_3_rota_especifica').dropDuplicates().count()
        logger.info(f"Total de rotas únicas ANTES dos agrupamentos: {total_rotas_inicio}")
        
        # 1. Agrupar por nível 1 (Centro Origem)
        df_nivel1 = df.groupBy('nivel_1_centro_origem').agg(
            {'volume_ton': 'sum', 'custo_sup_tku': 'sum', 'distancia_km': 'avg'}
        ).withColumnRenamed('sum(volume_ton)', 'volume_total_nivel1') \
         .withColumnRenamed('sum(custo_sup_tku)', 'custo_total_nivel1') \
         .withColumnRenamed('avg(distancia_km)', 'distancia_media_nivel1')
        
        # 2. Agrupar por nível 2 (Sub-categoria)
        df_nivel2 = df.groupBy('nivel_1_centro_origem', 'nivel_2_subcategoria').agg(
            {'volume_ton': 'sum', 'custo_sup_tku': 'sum', 'distancia_km': 'avg'}
        ).withColumnRenamed('sum(volume_ton)', 'volume_total_nivel2') \
         .withColumnRenamed('sum(custo_sup_tku)', 'custo_total_nivel2') \
         .withColumnRenamed('avg(distancia_km)', 'distancia_media_nivel2')
        
        # 3. Agrupar por nível 3 (Rota Específica)
        df_nivel3 = df.groupBy('nivel_1_centro_origem', 'nivel_2_subcategoria', 'nivel_3_rota_especifica').agg(
            {'volume_ton': 'sum', 'custo_sup_tku': 'sum', 'distancia_km': 'avg', 'preco_ton_km': 'avg'}
        ).withColumnRenamed('sum(volume_ton)', 'volume_total_nivel3') \
         .withColumnRenamed('sum(custo_sup_tku)', 'custo_total_nivel3') \
         .withColumnRenamed('avg(distancia_km)', 'distancia_media_nivel3') \
         .withColumnRenamed('avg(preco_ton_km)', 'preco_medio_ton_km')
        
        # Contar rotas no nível 3 APÓS agrupamento
        total_rotas_nivel3 = df_nivel3.count()
        logger.info(f"Total de rotas no nível 3 APÓS agrupamento: {total_rotas_nivel3}")
        
        # 4. Calcular preços médios por micro-região para comparação
        # Filtrar apenas rotas com volume significativo para o benchmark
        # Primeiro, calcular o volume total por rota para aplicar o filtro
        df_volume_por_rota = df.groupBy('centro_origem', 'rota_municipio').agg(
            {'volume_ton': 'sum'}
        ).withColumnRenamed('sum(volume_ton)', 'volume_total_rota')
        
        # Contar rotas ANTES do filtro de benchmark
        total_rotas_antes_benchmark = df_volume_por_rota.count()
        logger.info(f"Total de rotas ANTES do filtro de benchmark: {total_rotas_antes_benchmark}")
        
        # CALCULAR VOLUME MÍNIMO INTELIGENTEMENTE se ativado
        volume_minimo_benchmark_efetivo = self.volume_minimo_benchmark
        if self.adaptativo_benchmark:
            volume_minimo_benchmark_efetivo = self.calcular_volume_minimo_inteligente(df)
            logger.info(f"Volume mínimo adaptativo calculado: {volume_minimo_benchmark_efetivo:,.1f} ton")
        
        # Filtrar rotas com volume total >= volume mínimo para benchmark
        df_rotas_volume_alto = df_volume_por_rota.filter(
            col('volume_total_rota') >= volume_minimo_benchmark_efetivo
        )
        
        # Contar rotas APÓS filtro de benchmark
        total_rotas_apos_benchmark = df_rotas_volume_alto.count()
        logger.info(f"Total de rotas APÓS filtro de benchmark (>= {volume_minimo_benchmark_efetivo:,.1f} ton): {total_rotas_apos_benchmark}")
        
        # Aplicar o filtro de volume no cálculo do benchmark
        df_microregiao_prices = df.join(
            df_rotas_volume_alto,
            ['centro_origem', 'rota_municipio'],
            'inner'
        ).groupBy('microregiao_origem').agg(
            {'preco_ton_km': 'avg'}
        ).withColumnRenamed('avg(preco_ton_km)', 'preco_medio_microregiao')
        
        # Log das rotas filtradas para benchmark
        total_rotas_benchmark = df_rotas_volume_alto.count()
        
        logger.info(f"Benchmark - Filtro de volume aplicado:")
        logger.info(f"  - Volume mínimo configurado: {self.volume_minimo_benchmark:,} ton")
        logger.info(f"  - Volume mínimo efetivo: {volume_minimo_benchmark_efetivo:,.1f} ton")
        logger.info(f"  - Rotas para benchmark: {total_rotas_benchmark}")
        logger.info(f"  - Filtro aplicado: apenas rotas com volume >= {volume_minimo_benchmark_efetivo:,.1f} ton")
        
        # 5. Mesclar todos os níveis usando a micro-região correta
        # Primeiro, vamos adicionar a micro-região de origem ao df_nivel3
        df_nivel3_with_micro = df_nivel3.join(
            df.select('nivel_1_centro_origem', 'nivel_2_subcategoria', 'nivel_3_rota_especifica', 'microregiao_origem').dropDuplicates(),
            ['nivel_1_centro_origem', 'nivel_2_subcategoria', 'nivel_3_rota_especifica'],
            'left'
        )
        
        # Contar rotas APÓS primeiro JOIN
        total_rotas_apos_join1 = df_nivel3_with_micro.count()
        logger.info(f"Total de rotas APÓS JOIN com micro-região: {total_rotas_apos_join1}")
        
        # Agora fazer o merge com os preços médios
        df_hierarchical = df_nivel3_with_micro.join(
            df_nivel2, 
            ['nivel_1_centro_origem', 'nivel_2_subcategoria'], 
            'left'
        ).join(
            df_nivel1,
            'nivel_1_centro_origem',
            'left'
        ).join(
            df_microregiao_prices,
            'microregiao_origem',
            'left'
        )
        
        # Contar rotas APÓS todos os JOINs
        total_rotas_apos_joins = df_hierarchical.count()
        logger.info(f"Total de rotas APÓS todos os JOINs: {total_rotas_apos_joins}")
        
        # Verificar se houve perda de rotas durante os JOINs
        if total_rotas_nivel3 > total_rotas_apos_joins:
            rotas_perdidas_joins = total_rotas_nivel3 - total_rotas_apos_joins
            logger.warning(f"⚠️ {rotas_perdidas_joins} rotas perdidas durante os JOINs!")
            
            # Identificar quais rotas foram perdidas
            df_rotas_perdidas_joins = df_nivel3.select('nivel_1_centro_origem', 'nivel_2_subcategoria', 'nivel_3_rota_especifica').subtract(
                df_hierarchical.select('nivel_1_centro_origem', 'nivel_2_subcategoria', 'nivel_3_rota_especifica')
            )
            logger.warning(f"Rotas perdidas durante os JOINs:")
            for rota in df_rotas_perdidas_joins.collect():
                logger.warning(f"  - {rota['nivel_1_centro_origem']} → {rota['nivel_2_subcategoria']} → {rota['nivel_3_rota_especifica']}")
        
        # 6. Calcular métricas finais - TRATANDO CASOS DE DISTÂNCIA ZERO
        df_hierarchical = df_hierarchical.withColumn(
            'frete_geral_brl_ton',
            when(col('volume_total_nivel3') > 0, 
                 col('custo_total_nivel3') / col('volume_total_nivel3'))
            .otherwise(0.0)
        )
        
        df_hierarchical = df_hierarchical.withColumn(
            'frete_geral_brl_ton_km',
            when(
                (col('volume_total_nivel3') > 0) & (col('distancia_media_nivel3') > 0),
                col('custo_total_nivel3') / (col('volume_total_nivel3') * col('distancia_media_nivel3'))
            ).otherwise(0.0)  # Valor padrão para rotas com volume ou distância zero
        )
        
        # 7. Calcular oportunidade - TRATANDO CASOS ESPECIAIS
        df_hierarchical = df_hierarchical.withColumn(
            'oportunidade_brl_ton_km',
            when(
                (col('preco_medio_microregiao').isNotNull()) & 
                (col('frete_geral_brl_ton_km') > 0),
                col('frete_geral_brl_ton_km') - col('preco_medio_microregiao')
            ).otherwise(0.0)
        )
        
        # 8. Determinar ação - CONSIDERANDO STATUS DOS DADOS
        df_hierarchical = df_hierarchical.withColumn(
            'acao',
            when(
                (col('oportunidade_brl_ton_km') > 0.01) & (col('frete_geral_brl_ton_km') > 0),
                'Redução'
            ).when(
                (col('oportunidade_brl_ton_km') < -0.01) & (col('frete_geral_brl_ton_km') > 0),
                'Aumento'
            ).when(
                col('frete_geral_brl_ton_km') == 0,
                'Dados_Insuficientes'
            ).otherwise('Manter')
        )
        
        # 9. Adicionar observações para rotas com problemas
        df_hierarchical = df_hierarchical.withColumn(
            'observacao',
            when(
                col('frete_geral_brl_ton_km') == 0,
                'Rota com volume ou distância zero - análise manual necessária'
            ).when(
                col('oportunidade_brl_ton_km') == 0,
                'Sem dados suficientes para calcular oportunidade'
            ).otherwise('Análise automática realizada')
        )
        
        # 9. Ordenar por hierarquia
        df_hierarchical = df_hierarchical.orderBy(
            'nivel_1_centro_origem',
            'nivel_2_subcategoria',
            'nivel_3_rota_especifica'
        )
        
        # Contar rotas FINAIS
        total_rotas_final = df_hierarchical.count()
        logger.info(f"Estrutura hierárquica criada: {total_rotas_final} linhas")
        
        # Resumo final de perda de rotas
        if total_rotas_inicio > total_rotas_final:
            total_perdidas = total_rotas_inicio - total_rotas_final
            logger.warning(f"⚠️ RESUMO: {total_perdidas} rotas perdidas no total!")
            logger.warning(f"   - Início: {total_rotas_inicio} rotas")
            logger.warning(f"   - Final: {total_rotas_final} rotas")
        else:
            logger.info(f"✅ Todas as {total_rotas_inicio} rotas foram preservadas!")
        
        return df_hierarchical
    
    def generate_hierarchical_report(self, df_hierarchical):
        """
        Gera relatório hierárquico dos meses de maio, junho e julho de 2025
        DADOS LIMPOS - apenas as 8 colunas principais
        """
        logger.info("Gerando relatório hierárquico dos meses de maio, junho e julho de 2025...")
        
        # Selecionar colunas para o relatório final
        df_report = df_hierarchical.select(
            'nivel_1_centro_origem',
            'nivel_2_subcategoria', 
            'nivel_3_rota_especifica',
            'volume_total_nivel3',
            'frete_geral_brl_ton',
            'frete_geral_brl_ton_km',
            'oportunidade_brl_ton_km',
            'acao',
            'observacao'
        )
        
        # Renomear colunas para o formato final
        df_report = df_report.withColumnRenamed('nivel_1_centro_origem', 'Centro_Origem') \
                            .withColumnRenamed('nivel_2_subcategoria', 'Sub_Categoria') \
                            .withColumnRenamed('nivel_3_rota_especifica', 'Rota_Especifica') \
                            .withColumnRenamed('volume_total_nivel3', 'Volume_TON') \
                            .withColumnRenamed('frete_geral_brl_ton', 'Fr_Geral_BRL_TON') \
                            .withColumnRenamed('frete_geral_brl_ton_km', 'Fr_Geral_BRL_TON_KM') \
                            .withColumnRenamed('oportunidade_brl_ton_km', 'Oport_BRL_TON_KM') \
                            .withColumnRenamed('observacao', 'Observacao')
        
        # Criar diretório de saída se não existir
        os.makedirs("outputs", exist_ok=True)
        
        # Salvar resultados
        output_path = "outputs/analise_hierarquica_maio_junho_julho_2025"
        df_report.write.mode("overwrite").parquet(output_path)
        
        # Salvar relatório em formato legível - INCLUINDO STATUS E OBSERVAÇÕES
        df_report.toPandas().to_excel(
            "outputs/dashboard_hierarquico_maio_junho_julho_2025_SEM_FILTRO_DISTANCIA.xlsx", 
            index=False,
            sheet_name='Analise_Hierarquica'
        )
        
        # Retornar resumo
        total_rotas = df_report.count()
        rotas_reducao = df_report.filter(col('acao') == 'Redução').count()
        rotas_aumento = df_report.filter(col('acao') == 'Aumento').count()
        rotas_manter = df_report.filter(col('acao') == 'Manter').count()
        rotas_dados_insuficientes = df_report.filter(col('acao') == 'Dados_Insuficientes').count()
        
        # Top oportunidades de redução
        top_reducoes = df_report.filter(col('acao') == 'Redução') \
            .orderBy(desc('Oport_BRL_TON_KM')) \
            .limit(5)
        
        # Top rotas por volume
        top_volume = df_report.orderBy(desc('Volume_TON')).limit(5)
        
        # Rotas com dados insuficientes
        rotas_problema = df_report.filter(col('acao') == 'Dados_Insuficientes') \
            .select('Centro_Origem', 'Sub_Categoria', 'Rota_Especifica', 'Volume_TON', 'Observacao')
        
        return {
            'total_rotas': total_rotas,
            'rotas_reducao': rotas_reducao,
            'rotas_aumento': rotas_aumento,
            'rotas_manter': rotas_manter,
            'rotas_dados_insuficientes': rotas_dados_insuficientes,
            'top_reducoes': top_reducoes.toPandas().to_dict('records'),
            'top_volume': top_volume.toPandas().to_dict('records'),
            'rotas_problema': rotas_problema.toPandas().to_dict('records')
        }

def main():
    """
    Função principal para executar análise hierárquica dos meses de maio, junho e julho de 2025
    """
    try:
        # Inicializar Spark
        try:
            from pyspark.sql import SparkSession
            spark = SparkSession.builder \
                .appName("AnaliseHierarquicaMaioJunhoJulho2025") \
                .config("spark.sql.adaptive.enabled", "true") \
                .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
                .config("spark.driver.extraJavaOptions", "-Djava.security.manager=allow -Djava.security.policy=unlimited") \
                .config("spark.executor.extraJavaOptions", "-Djava.security.manager=allow -Djava.security.policy=unlimited") \
                .config("spark.driver.host", "localhost") \
                .config("spark.driver.bindAddress", "localhost") \
                .config("spark.sql.warehouse.dir", "/tmp/spark-warehouse") \
                .config("spark.local.dir", "/tmp/spark-temp") \
                .master("local[*]") \
                .getOrCreate()
            
            logger.info("Spark inicializado localmente")
        except ImportError:
            logger.info("Usando sessão Spark do Databricks")
            spark = None
        
        if spark is None:
            logger.error("Spark não está disponível. Execute este script no Databricks ou instale PySpark localmente.")
            return
        
        # Inicializar analisador
        analyzer = HierarchicalGroupingAnalyzer(spark)
        
        # Configurar volumes mínimos
        analyzer.configurar_volumes_minimos(volume_minimo_analise=100, volume_minimo_benchmark=1000)
        
        # Configurar benchmark inteligente
        analyzer.configurar_benchmark_inteligente(percentil_benchmark=75, min_rotas_benchmark=10, max_rotas_benchmark=100, adaptativo_benchmark=True)
        
        # Caminho para o arquivo de dados
        excel_path = "sample_data.xlsx"
        
        # Tentar diferentes caminhos
        possible_paths = [
            excel_path,
            os.path.join(os.getcwd(), excel_path),
        ]
        
        try:
            if '__file__' in globals():
                possible_paths.extend([
                    os.path.join(os.path.dirname(__file__), excel_path),
                    os.path.join(os.path.dirname(__file__), "..", excel_path)
                ])
        except NameError:
            pass
        
        excel_path = None
        for path in possible_paths:
            if Path(path).exists():
                excel_path = path
                logger.info(f"Arquivo encontrado em: {excel_path}")
                break
        
        if not excel_path:
            logger.error(f"Arquivo sample_data.xlsx não encontrado")
            return
        
        # Executar análise
        start_time = time.time()
        
        # Carregar dados
        df_raw = analyzer.load_data_from_excel(excel_path)
        
        # Filtrar meses de maio, junho e julho de 2025
        df_filtered = analyzer.filter_may_june_july_2025(df_raw)
        
        # Preparar dados para hierarquia
        df_processed = analyzer.prepare_data_hierarchical(df_filtered)
        
        # ANALISAR DISTRIBUIÇÃO DOS VOLUMES para otimização do benchmark
        logger.info("=" * 60)
        logger.info("ANÁLISE DA DISTRIBUIÇÃO DOS VOLUMES")
        logger.info("=" * 60)
        
        analise_distribuicao = analyzer.analisar_distribuicao_volumes(df_processed)
        if "erro" not in analise_distribuicao:
            logger.info(f"Distribuição por faixas de volume:")
            for faixa, quantidade in analise_distribuicao["faixas_volume"].items():
                logger.info(f"  - {faixa}: {quantidade} rotas")
            
            logger.info(f"\nRecomendações para otimização:")
            for recomendacao in analise_distribuicao["recomendacoes"]:
                logger.info(f"  - {recomendacao}")
        else:
            logger.warning(f"Erro na análise de distribuição: {analise_distribuicao['erro']}")
        
        logger.info("=" * 60)
        
        # Criar estrutura hierárquica
        df_hierarchical = analyzer.create_hierarchical_structure(df_processed)
        
        # Gerar relatório
        report = analyzer.generate_hierarchical_report(df_hierarchical)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Mostrar resultados
        logger.info("=" * 80)
        logger.info("RELATÓRIO DA ANÁLISE HIERÁRQUICA - MAIO, JUNHO E JULHO 2025")
        logger.info("=" * 80)
        logger.info(f"Tempo de execução: {execution_time:.2f} segundos")
        logger.info(f"Total de rotas analisadas: {report['total_rotas']}")
        logger.info(f"Oportunidades de redução: {report['rotas_reducao']}")
        logger.info(f"Oportunidades de aumento: {report['rotas_aumento']}")
        logger.info(f"Rotas para manter: {report['rotas_manter']}")
        logger.info(f"Rotas com dados insuficientes: {report['rotas_dados_insuficientes']}")
        
        # Mostrar rotas com dados insuficientes
        if report['rotas_dados_insuficientes'] > 0:
            logger.info(f"\nROTAS COM DADOS INSUFICIENTES (requerem análise manual):")
            for rota in report['rotas_problema']:
                logger.info(f"  - {rota['Rota_Especifica']}: {rota['Observacao']}")
        
        logger.info("\nTOP 5 OPORTUNIDADES DE REDUÇÃO (Maio-Junho-Julho 2025):")
        for i, op in enumerate(report['top_reducoes'][:5], 1):
            logger.info(f"{i}. {op['Rota_Especifica']}: {op['Oport_BRL_TON_KM']:.4f} BRL/TON/KM")
        
        logger.info("\nTOP 5 ROTAS POR VOLUME (Maio-Junho-Julho 2025):")
        for i, rota in enumerate(report['top_volume'][:5], 1):
            logger.info(f"{i}. {rota['Rota_Especifica']}: {rota['Volume_TON']:,.1f} ton")
        
        logger.info(f"\nRelatório salvo em: outputs/dashboard_hierarquico_maio_junho_julho_2025_SEM_FILTRO_DISTANCIA.xlsx")
        logger.info("Análise hierárquica dos meses de maio, junho e julho de 2025 concluída com sucesso!")
        
    except Exception as e:
        logger.error(f"Erro durante a análise: {str(e)}")
        raise

if __name__ == "__main__":
    main()
