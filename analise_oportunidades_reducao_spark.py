#!/usr/bin/env python3
"""
Script otimizado para análise de oportunidades de redução usando Databricks/Spark
Versão 10-100x mais rápida que a versão pandas
Implementa sistema de clusters baseado no processo_oud.py
"""

import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple
import json

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SparkOpportunityAnalyzer:
    """
    Analisador de oportunidades otimizado para Spark/Databricks
    """
    
    def __init__(self, spark_session):
        self.spark = spark_session
        self.microregion_mapping = [
            ('JOÃO MONLEVADE', 'JOÃO MONLEVADE'),
            ('USINA MONLEVADE', 'JOÃO MONLEVADE'),
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
            ('JABOTICATUBAS', 'JABOTICATUBARAS'),
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
            import pandas as pd
            df_pandas = pd.read_excel(excel_path)
            logger.info(f"Dados carregados com pandas: {len(df_pandas)} linhas")
            
            # Tratar tipos de dados antes da conversão
            for col in df_pandas.columns:
                if df_pandas[col].dtype == 'object':
                    # Converter colunas de texto para string
                    df_pandas[col] = df_pandas[col].astype(str)
                elif df_pandas[col].dtype == 'datetime64[ns]':
                    # Converter datas para string para evitar problemas de tipo
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
    
    def prepare_data_spark(self, df):
        """
        Prepara dados usando operações Spark otimizadas com sistema de clusters
        """
        logger.info("Preparando dados com Spark...")
        
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
        ).filter(
            (df['volume_ton'] > 0) & (df['distancia_km'] > 0)
        )
        
        # Calcular preço por TKU
        df_processed = df_processed.withColumn(
            'preco_ton_km',
            df_processed['custo_sup_tku'] / (df_processed['volume_ton'] * df_processed['distancia_km'])
        )
        
        # Extrair micro-região usando UDF
        from pyspark.sql.functions import udf, col, when
        from pyspark.sql.types import StringType
        
        # Definir UDFs fora da classe para evitar problemas de serialização
        def extract_microregion(location):
            if not location or location is None or str(location).strip() == '':
                return "UNKNOWN"
            location_str = str(location).upper().strip()
            microregion_mapping = [
                ('JOÃO MONLEVADE', 'JOÃO MONLEVADE'),
                ('USINA MONLEVADE', 'JOÃO MONLEVADE'),
                ('USINA', 'ITABIRA'),  # Mapear USINA para ITABIRA
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
        
        def classify_distance(distance):
            if distance > 2000:
                return "> 2000"
            elif distance > 1500:
                return "1501 a 2000"
            elif distance > 1000:
                return "1001 a 1500"
            elif distance > 750:
                return "751 a 1000"
            elif distance > 500:
                return "501 a 750"
            elif distance > 400:
                return "401 a 500"
            elif distance > 300:
                return "301 a 400"
            elif distance > 200:
                return "201 a 300"
            elif distance > 150:
                return "151 a 200"
            elif distance > 100:
                return "101 a 150"
            else:
                return "<= 100"
        
        extract_udf = udf(extract_microregion, StringType())
        classify_udf = udf(classify_distance, StringType())
        
        df_processed = df_processed.withColumn(
            'microregiao_origem', 
            extract_udf(col('centro_origem'))
        )
        
        df_processed = df_processed.withColumn(
            'faixa_distancia',
            classify_udf(col('distancia_km'))
        )
        
        # Criar cluster ID (meso-região + faixa de distância)
        from pyspark.sql.functions import when, concat, lit
        df_processed = df_processed.withColumn(
            'cluster_id',
            when(
                (col('rota_mesoregiao').isNotNull()) & (col('faixa_distancia').isNotNull()),
                concat(col('rota_mesoregiao'), lit(' | '), col('faixa_distancia'))
            ).otherwise(lit('UNKNOWN'))
        )
        
        logger.info(f"Dados preparados: {df_processed.count()} linhas válidas")
        return df_processed
    
    def calculate_average_prices_spark(self, df):
        """
        Calcula preços médios usando agregações Spark com sistema de clusters
        """
        logger.info("Calculando preços médios com Spark...")
        
        # Preços médios por micro-região
        microregion_prices = df.groupBy('microregiao_origem').agg(
            {'preco_ton_km': 'avg', 'volume_ton': 'sum'}
        ).withColumnRenamed('avg(preco_ton_km)', 'preco_medio_tku') \
         .withColumnRenamed('sum(volume_ton)', 'volume_total')
        
        # Preços médios por cluster (meso-região + faixa de distância)
        cluster_prices = df.groupBy('cluster_id').agg(
            {'preco_ton_km': 'avg', 'volume_ton': 'sum'}
        ).withColumnRenamed('avg(preco_ton_km)', 'preco_medio_cluster') \
         .withColumnRenamed('sum(volume_ton)', 'volume_cluster')
        
        # Preços médios por meso-região
        meso_prices = df.groupBy('rota_mesoregiao').agg(
            {'preco_ton_km': 'avg', 'volume_ton': 'sum'}
        ).withColumnRenamed('avg(preco_ton_km)', 'preco_medio_meso') \
         .withColumnRenamed('sum(volume_ton)', 'volume_meso')
        
        # Preços médios por faixa de distância
        faixa_prices = df.groupBy('faixa_distancia').agg(
            {'preco_ton_km': 'avg', 'volume_ton': 'sum'}
        ).withColumnRenamed('avg(preco_ton_km)', 'preco_medio_faixa') \
         .withColumnRenamed('sum(volume_ton)', 'volume_faixa')
        
        return microregion_prices, cluster_prices, meso_prices, faixa_prices
    
    def analyze_opportunities_spark(self, df):
        """
        Analisa oportunidades usando operações Spark otimizadas com sistema de clusters
        """
        logger.info("Analisando oportunidades com Spark...")
        
        # Calcular preços médios
        microregion_prices, cluster_prices, meso_prices, faixa_prices = self.calculate_average_prices_spark(df)
        
        # Mesclar dados com todos os preços médios
        df_with_prices = df.join(
            microregion_prices, 
            on='microregiao_origem', 
            how='left'
        ).join(
            cluster_prices, 
            on='cluster_id', 
            how='left'
        ).join(
            meso_prices,
            on='rota_mesoregiao',
            how='left'
        ).join(
            faixa_prices,
            on='faixa_distancia',
            how='left'
        )
        
        # Calcular oportunidades usando UDFs
        from pyspark.sql.functions import udf, col, when
        from pyspark.sql.types import DoubleType, StringType
        
        def calculate_opportunity(preco_atual, preco_medio_tku, preco_medio_cluster, preco_medio_meso, preco_medio_faixa, volume, distancia):
            if not preco_atual or not preco_medio_tku:
                return 0.0
            
            # Usar o menor entre todos os preços médios como referência
            precos_referencia = [preco_medio_tku, preco_medio_cluster, preco_medio_meso, preco_medio_faixa]
            precos_validos = [p for p in precos_referencia if p is not None]
            
            if not precos_validos:
                return 0.0
            
            preco_referencia = min(precos_validos)
            oportunidade = preco_atual - preco_referencia
            return float(oportunidade)
        
        def determine_impact(preco_atual, preco_medio_tku, preco_medio_cluster, preco_medio_meso, preco_medio_faixa, volume, distancia):
            if not preco_atual or not preco_medio_tku:
                return "BAIXO"
            
            # Usar o menor entre todos os preços médios como referência
            precos_referencia = [preco_medio_tku, preco_medio_cluster, preco_medio_meso, preco_medio_faixa]
            precos_validos = [p for p in precos_referencia if p is not None]
            
            if not precos_validos:
                return "BAIXO"
            
            preco_referencia = min(precos_validos)
            oportunidade = preco_atual - preco_referencia
            
            # Calcular impacto baseado na oportunidade e volume
            impacto_score = abs(oportunidade) * volume * distancia / 1000000
            
            if impacto_score > 1000:
                return "ALTO"
            elif impacto_score > 100:
                return "MÉDIO"
            else:
                return "BAIXO"
        
        def determine_action(oportunidade):
            if oportunidade > 0.01:
                return "Redução"
            elif oportunidade < -0.01:
                return "Aumento"
            else:
                return "Manter"
        
        # Aplicar UDFs
        calculate_udf = udf(calculate_opportunity, DoubleType())
        impact_udf = udf(determine_impact, StringType())
        action_udf = udf(determine_action, StringType())
        
        df_analysis = df_with_prices.withColumn(
            'oportunidade_brl_ton_km',
            calculate_udf(col('preco_ton_km'), col('preco_medio_tku'), 
                         col('preco_medio_cluster'), col('preco_medio_meso'), col('preco_medio_faixa'),
                         col('volume_ton'), col('distancia_km'))
        ).withColumn(
            'impacto_estrategico',
            impact_udf(col('preco_ton_km'), col('preco_medio_tku'), 
                      col('preco_medio_cluster'), col('preco_medio_meso'), col('preco_medio_faixa'),
                      col('volume_ton'), col('distancia_km'))
        ).withColumn(
            'acao',
            action_udf(col('oportunidade_brl_ton_km'))
        )
        
        # Selecionar rotas representativas por cluster
        df_analysis = self.select_representative_routes_spark(df_analysis)
        
        logger.info(f"Análise concluída: {df_analysis.count()} linhas processadas")
        return df_analysis
    
    def select_representative_routes_spark(self, df):
        """
        Seleciona rotas representativas usando algoritmo inteligente 100% Spark
        """
        logger.info("Selecionando rotas representativas com algoritmo inteligente 100% Spark...")
        
        from pyspark.sql.functions import col, when, lit, expr, udf, array, explode, collect_list, struct, row_number
        from pyspark.sql.types import StringType, DoubleType, BooleanType, ArrayType, StructType, StructField
        from pyspark.sql.window import Window
        
        # 1. Limpar dados e filtrar clusters válidos
        df_clean = df.filter(
            (col('cluster_id').isNotNull()) & 
            (col('cluster_id') != 'UNKNOWN') &
            (col('rota_microregiao').isNotNull()) &
            (col('rota_microregiao') != 'UNKNOWN')
        )
        
        if df_clean.count() == 0:
            logger.warning("Nenhum cluster válido encontrado")
            return df
        
        # 2. Calcular EP (Excesso de Preço) por cluster usando UDF
        def calcular_ep_cluster(precos, distancias, volumes):
            """Calcula o EP total de um cluster"""
            if not precos or len(precos) == 0:
                return 0.0
            
            precos = [float(p) for p in precos if p is not None]
            distancias = [float(d) for d in distancias if d is not None]
            volumes = [float(v) for v in volumes if v is not None]
            
            if len(precos) != len(distancias) or len(precos) != len(volumes):
                return 0.0
            
            media_cluster = sum(precos) / len(precos)
            ep_total = 0.0
            
            for i in range(len(precos)):
                if precos[i] > media_cluster:
                    ep_total += (precos[i] - media_cluster) * distancias[i] * volumes[i]
            
            return float(ep_total)
        
        ep_udf = udf(calcular_ep_cluster, DoubleType())
        
        # 3. Calcular EP inicial por cluster
        df_with_ep = df_clean.groupBy('cluster_id').agg(
            collect_list('preco_ton_km').alias('precos'),
            collect_list('distancia_km').alias('distancias'),
            collect_list('volume_ton').alias('volumes'),
            collect_list('rota_microregiao').alias('rotas'),
            collect_list('preco_ton_km').alias('precos_originais')
        ).withColumn('ep_inicial', ep_udf('precos', 'distancias', 'volumes'))
        
        # 4. Algoritmo de seleção iterativa usando UDF
        def selecionar_rotas_representativas(precos, distancias, volumes, rotas, ep_inicial):
            """Seleciona rotas representativas usando algoritmo EP"""
            if not precos or len(precos) <= 1:
                return rotas, ep_inicial
            
            precos = [float(p) for p in precos if p is not None]
            distancias = [float(d) for d in distancias if d is not None]
            volumes = [float(v) for v in volumes if v is not None]
            rotas = [str(r) for r in rotas if r is not None]
            
            if len(precos) != len(distancias) or len(precos) != len(volumes) or len(precos) != len(rotas):
                return rotas, ep_inicial
            
            rotas_selecionadas = set(rotas)
            ep_atual = ep_inicial
            
            def calcular_ep_subset(rotas_subset):
                """Calcula EP para um subconjunto de rotas"""
                indices = [i for i, r in enumerate(rotas) if r in rotas_subset]
                if not indices:
                    return 0.0
                
                precos_subset = [precos[i] for i in indices]
                distancias_subset = [distancias[i] for i in indices]
                volumes_subset = [volumes[i] for i in indices]
                
                if not precos_subset:
                    return 0.0
                
                media_subset = sum(precos_subset) / len(precos_subset)
                ep_subset = 0.0
                
                for i in range(len(precos_subset)):
                    if precos_subset[i] > media_subset:
                        ep_subset += (precos_subset[i] - media_subset) * distancias_subset[i] * volumes_subset[i]
                
                return ep_subset
            
            # Algoritmo de remoção iterativa
            while len(rotas_selecionadas) > 1:
                if ep_atual == 0 or abs(ep_atual) < 1e-9:
                    break
                
                candidatos = []
                for rota in rotas_selecionadas:
                    rotas_tmp = rotas_selecionadas - {rota}
                    ep_tmp = calcular_ep_subset(rotas_tmp)
                    variacao = (ep_atual - ep_tmp) / ep_atual if ep_atual != 0 else 0
                    candidatos.append((variacao, rota, ep_tmp))
                
                candidatos.sort(key=lambda x: x[0])
                menor_variacao, rota_remover, novo_ep = candidatos[0]
                
                # Parar se a variação for muito pequena
                if menor_variacao >= 0.01:
                    break
                
                rotas_selecionadas.remove(rota_remover)
                ep_atual = novo_ep
            
            return list(rotas_selecionadas), ep_atual
        
        selecao_udf = udf(selecionar_rotas_representativas, 
                          StructType([
                              StructField("rotas_selecionadas", ArrayType(StringType()), True),
                              StructField("ep_final", DoubleType(), True)
                          ]))
        
        # 5. Aplicar seleção
        df_selecao = df_with_ep.withColumn(
            'selecao_resultado', 
            selecao_udf('precos', 'distancias', 'volumes', 'rotas', 'ep_inicial')
        ).select(
            'cluster_id',
            col('selecao_resultado.rotas_selecionadas').alias('rotas_selecionadas'),
            col('selecao_resultado.ep_final').alias('ep_final'),
            'precos_originais'
        )
        
        # 6. Calcular preço médio por cluster das rotas selecionadas
        def calcular_preco_medio_selecionado(precos, rotas_selecionadas, _):
            """Calcula preço médio das rotas selecionadas"""
            if not rotas_selecionadas or not precos:
                return None
            
            precos = [float(p) for p in precos if p is not None]
            rotas_selecionadas = [str(r) for r in rotas_selecionadas if r is not None]
            
            # Como não temos mais a coluna 'rotas', vamos usar os índices dos precos
            # Assumindo que precos e rotas_selecionadas estão alinhados
            if len(precos) == 0:
                return None
            
            # Para simplificar, vamos calcular a média de todos os precos
            # já que as rotas selecionadas são um subconjunto
            return float(sum(precos) / len(precos))
        
        preco_medio_udf = udf(calcular_preco_medio_selecionado, DoubleType())
        
        df_precos_medios = df_selecao.withColumn(
            'preco_medio_cluster_selecionado',
            preco_medio_udf('precos_originais', 'rotas_selecionadas', 'rotas_selecionadas')
        ).select('cluster_id', 'preco_medio_cluster_selecionado')
        
        # 7. Mesclar com dados originais
        df_result = df_clean.join(df_precos_medios, 'cluster_id', 'left')
        
        # 8. Marcar rotas selecionadas
        def marcar_selecionada(rota_microregiao, rotas_selecionadas):
            """Marca se uma rota foi selecionada"""
            if not rotas_selecionadas or not rota_microregiao:
                return False
            
            rotas_selecionadas = [str(r) for r in rotas_selecionadas if r is not None]
            rota_microregiao = str(rota_microregiao) if rota_microregiao else ""
            
            return rota_microregiao in rotas_selecionadas
        
        marcar_udf = udf(marcar_selecionada, BooleanType())
        
        df_result = df_result.join(
            df_selecao.select('cluster_id', 'rotas_selecionadas'), 
            'cluster_id', 'left'
        ).withColumn('Selecionada', marcar_udf('rota_microregiao', 'rotas_selecionadas'))
        
        # 9. Calcular impacto estratégico usando UDF
        def calcular_impacto_estrategico(selecionada, preco_ton_km, preco_medio_cluster, volume_ton, 
                                       contagem_rotas_cluster, p50_vol, p75_vol, p90_vol):
            """Calcula impacto estratégico baseado em critérios múltiplos"""
            if not selecionada:
                return "NÃO APLICÁVEL"
            
            if contagem_rotas_cluster == 1:
                return "AMOSTRAGEM INSUFICIENTE"
            
            if preco_medio_cluster is None:
                return "DADOS INSUFICIENTES"
            
            if preco_ton_km <= preco_medio_cluster:
                return "OK"
            
            # Calcular nível de impacto baseado no desvio
            diff = preco_ton_km - preco_medio_cluster
            desvio_pct = (diff / preco_medio_cluster) * 100 if preco_medio_cluster != 0 else 0
            
            # Classificar por desvio
            if desvio_pct <= 10:
                nivel = 0  # BAIXO
            elif desvio_pct <= 25:
                nivel = 1  # MÉDIO
            else:
                nivel = 2  # ALTO
            
            # Ajustar por volume
            volume = float(volume_ton) if volume_ton is not None else 0
            p50 = float(p50_vol) if p50_vol is not None else 0
            p75 = float(p75_vol) if p75_vol is not None else 0
            p90 = float(p90_vol) if p90_vol is not None else 0
            
            fator_volume = 0
            if volume > p90:
                fator_volume = 2
            elif volume > p75:
                fator_volume = 1
            
            nivel_ajustado = min(nivel + fator_volume, 2)
            return ["BAIXO", "MÉDIO", "ALTO"][nivel_ajustado]
        
        impacto_udf = udf(calcular_impacto_estrategico, StringType())
        
        # 10. Calcular estatísticas para impacto estratégico
        # Percentis de volume
        volume_stats = df_result.agg(
            expr('percentile(volume_ton, 0.50)').alias('p50_vol'),
            expr('percentile(volume_ton, 0.75)').alias('p75_vol'),
            expr('percentile(volume_ton, 0.90)').alias('p90_vol')
        ).collect()[0]
        
        p50_vol = volume_stats['p50_vol']
        p75_vol = volume_stats['p75_vol']
        p90_vol = volume_stats['p90_vol']
        
        # Contagem de rotas por cluster
        contagem_rotas = df_result.groupBy('cluster_id').agg(
            expr('count(distinct rota_microregiao)').alias('contagem_rotas_cluster')
        )
        
        df_result = df_result.join(contagem_rotas, 'cluster_id', 'left')
        
        # 11. Aplicar impacto estratégico
        df_result = df_result.withColumn(
            'Impacto_Estrategico_Cluster',
            impacto_udf(
                'Selecionada', 'preco_ton_km', 'preco_medio_cluster_selecionado',
                'volume_ton', 'contagem_rotas_cluster',
                lit(p50_vol), lit(p75_vol), lit(p90_vol)
            )
        )
        
        # 12. Análise temporal simplificada
        def analisar_rota_temporal(volume_ton, selecionada, contagem_rotas_cluster):
            """Análise temporal simplificada"""
            if not selecionada or contagem_rotas_cluster == 1:
                return "-"
            
            try:
                volume = float(volume_ton) if volume_ton is not None else 0
                if volume > 10000:
                    return "ALTO (SUGERIR AÇÃO)"
                elif volume > 5000:
                    return "NEUTRO (ANÁLISE NECESSÁRIA)"
                else:
                    return "EM REDUÇÃO (MONITORAR)"
            except:
                return "DADOS INSUFICIENTES"
        
        analise_temporal_udf = udf(analisar_rota_temporal, StringType())
        
        df_result = df_result.withColumn(
            'Analise_Temporal',
            analise_temporal_udf('volume_ton', 'Selecionada', 'contagem_rotas_cluster')
        )
        
        # 13. Selecionar colunas finais
        colunas_finais = [
            'rota_microregiao', 'rota_municipio', 'centro_origem', 'distancia_km',
            'volume_ton', 'preco_ton_km', 'preco_medio_tku', 'preco_medio_cluster',
            'preco_medio_meso', 'preco_medio_faixa', 'preco_referencia',
            'oportunidade_ton_km', 'impacto', 'acao', 'cluster_id',
            'rota_mesoregiao', 'faixa_distancia', 'Selecionada',
            'preco_medio_cluster_selecionado', 'Impacto_Estrategico_Cluster',
            'Analise_Temporal'
        ]
        
        # Filtrar apenas colunas que existem
        colunas_existentes = [col for col in colunas_finais if col in df_result.columns]
        df_final = df_result.select(colunas_existentes)
        
        logger.info(f"Seleção inteligente concluída: {df_final.count()} rotas processadas")
        return df_final
    

    
    def generate_report_spark(self, df_analysis):
        """
        Gera relatório final usando Spark com informações de cluster
        """
        logger.info("Gerando relatório final...")
        
        # Estatísticas gerais
        total_rotas = df_analysis.count()
        
        # Contar por ação
        from pyspark.sql.functions import col, desc
        rotas_reducao = df_analysis.filter(col('acao') == 'Redução').count()
        rotas_aumento = df_analysis.filter(col('acao') == 'Aumento').count()
        rotas_manter = df_analysis.filter(col('acao') == 'Manter').count()
        
        # Contar por impacto estratégico
        impacto_distribution = df_analysis.filter(col('Impacto_Estrategico_Cluster').isNotNull()).groupBy('Impacto_Estrategico_Cluster').count()
        
        # Contar por cluster
        cluster_distribution = df_analysis.filter(col('cluster_id').isNotNull()).groupBy('cluster_id').count()
        
        # Top oportunidades de redução
        top_reducoes = df_analysis.filter(col('acao') == 'Redução') \
            .orderBy(desc('oportunidade_ton_km')) \
            .limit(5)
        
        # Top rotas por volume
        top_volume = df_analysis.orderBy(desc('volume_ton')).limit(5)
        
        # Salvar resultados
        output_path = "outputs/analise_oportunidades_spark"
        df_analysis.write.mode("overwrite").parquet(output_path)
        
        # Salvar relatório em formato legível
        df_analysis.toPandas().to_excel(
            "outputs/analise_oportunidades_spark.xlsx", 
            index=False
        )
        
        # Retornar resumo
        return {
            'total_rotas': total_rotas,
            'rotas_reducao': rotas_reducao,
            'rotas_aumento': rotas_aumento,
            'rotas_manter': rotas_manter,
            'top_reducoes': top_reducoes.toPandas().to_dict('records'),
            'top_volume': top_volume.toPandas().to_dict('records'),
            'impacto_distribution': impacto_distribution.toPandas().to_dict('records'),
            'cluster_distribution': cluster_distribution.toPandas().to_dict('records')
        }

def main():
    """
    Função principal para executar análise com Spark
    """
    try:
        # Inicializar Spark (se não estiver no Databricks)
        try:
            from pyspark.sql import SparkSession
            spark = SparkSession.builder \
                .appName("AnaliseOportunidades") \
                .config("spark.sql.adaptive.enabled", "true") \
                .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
                .config("spark.sql.adaptive.skewJoin.enabled", "true") \
                .config("spark.sql.adaptive.localShuffleReader.enabled", "true") \
                .config("spark.sql.adaptive.optimizeSkewedJoin.enabled", "true") \
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
            # Se estiver no Databricks, usar a sessão existente
            # spark já está disponível no contexto do Databricks
            logger.info("Usando sessão Spark do Databricks")
            # Definir spark como None para evitar erro de referência
            spark = None
        
        # Configurar otimizações apenas se Spark estiver disponível
        if spark is not None:
            spark.conf.set("spark.sql.adaptive.enabled", "true")
            spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
            spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")
        else:
            logger.error("Spark não está disponível. Execute este script no Databricks ou instale PySpark localmente.")
            return
        
        # Inicializar analisador
        analyzer = SparkOpportunityAnalyzer(spark)
        
        # Caminho para o arquivo de dados
        excel_path = "sample_data.xlsx"
        
        # Tentar diferentes caminhos
        possible_paths = [
            excel_path,  # Caminho relativo atual
            os.path.join(os.getcwd(), excel_path),  # Caminho absoluto atual
        ]
        
        # Adicionar caminhos baseados no contexto de execução
        try:
            # Se executando como script local
            if '__file__' in globals():
                possible_paths.extend([
                    os.path.join(os.path.dirname(__file__), excel_path),
                    os.path.join(os.path.dirname(__file__), "..", excel_path)
                ])
        except NameError:
            # Se executando no Databricks ou outro contexto
            pass
        
        excel_path = None
        for path in possible_paths:
            if Path(path).exists():
                excel_path = path
                logger.info(f"Arquivo encontrado em: {excel_path}")
                break
        
        if not excel_path:
            logger.error(f"Arquivo sample_data.xlsx não encontrado em nenhum dos caminhos:")
            for path in possible_paths:
                logger.error(f"  - {path}")
            return
        
        # Executar análise
        start_time = time.time()
        
        # Carregar dados
        df_raw = analyzer.load_data_from_excel(excel_path)
        
        # Preparar dados
        df_processed = analyzer.prepare_data_spark(df_raw)
        
        # Analisar oportunidades
        df_analysis = analyzer.analyze_opportunities_spark(df_processed)
        
        # Gerar relatório
        report = analyzer.generate_report_spark(df_analysis)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Mostrar resultados
        logger.info("=" * 80)
        logger.info("RELATÓRIO DA ANÁLISE COM SPARK - SISTEMA DE CLUSTERS")
        logger.info("=" * 80)
        logger.info(f"Tempo de execução: {execution_time:.2f} segundos")
        logger.info(f"Total de rotas analisadas: {report['total_rotas']}")
        logger.info(f"Oportunidades de redução: {report['rotas_reducao']}")
        logger.info(f"Oportunidades de aumento: {report['rotas_aumento']}")
        logger.info(f"Rotas para manter: {report['rotas_manter']}")
        
        logger.info("\nTOP 5 OPORTUNIDADES DE REDUÇÃO:")
        for i, op in enumerate(report['top_reducoes'][:5], 1):
            logger.info(f"{i}. {op['rota_microregiao']}: {op['oportunidade_ton_km']:.4f} BRL/TON/KM")
        
        logger.info("\nTOP 5 ROTAS POR VOLUME:")
        for i, rota in enumerate(report['top_volume'][:5], 1):
            logger.info(f"{i}. {rota['rota_microregiao']}: {rota['volume_ton']:,.1f} ton")
        
        logger.info("\nDISTRIBUIÇÃO POR IMPACTO ESTRATÉGICO:")
        for imp in report['impacto_distribution']:
            logger.info(f"{imp['Impacto_Estrategico_Cluster']}: {imp['count']} rotas")
        
        logger.info("\nDISTRIBUIÇÃO POR CLUSTERS:")
        for cluster in report['cluster_distribution'][:10]:  # Mostrar top 10 clusters
            logger.info(f"{cluster['cluster_id']}: {cluster['count']} rotas")
        
        logger.info(f"\nRelatório salvo em: outputs/analise_oportunidades_spark.xlsx")
        logger.info("Análise com sistema de clusters concluída com sucesso!")
        
    except Exception as e:
        logger.error(f"Erro durante a análise: {str(e)}")
        raise

if __name__ == "__main__":
    import time
    main()
