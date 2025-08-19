#!/usr/bin/env python3
"""
Script para análise de agrupamento hierárquico similar ao dashboard da imagem
Implementa estrutura: Centro Origem > Sub-categoria > Rota Específica
"""

import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, lit, expr, udf, struct, row_number, desc
from pyspark.sql.types import StringType, DoubleType, BooleanType
from pyspark.sql.window import Window

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HierarchicalGroupingAnalyzer:
    """
    Analisador de agrupamento hierárquico para dashboard
    """
    
    def __init__(self, spark_session):
        self.spark = spark_session
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
    
    def prepare_data_hierarchical(self, df):
        """
        Prepara dados para agrupamento hierárquico
        """
        logger.info("Preparando dados para agrupamento hierárquico...")
        
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
            .when(col('centro_origem').like('%ITABIRA%'), 'ITABIRA')
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
        
        logger.info(f"Dados preparados para hierarquia: {df_processed.count()} linhas válidas")
        return df_processed
    
    def create_hierarchical_structure(self, df):
        """
        Cria estrutura hierárquica similar ao dashboard da imagem
        """
        logger.info("Criando estrutura hierárquica...")
        
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
        
        # 4. Calcular preços médios por micro-região para comparação
        df_microregiao_prices = df.groupBy('microregiao_origem').agg(
            {'preco_ton_km': 'avg'}
        ).withColumnRenamed('avg(preco_ton_km)', 'preco_medio_microregiao')
        
        # 5. Mesclar todos os níveis usando a micro-região correta
        # Primeiro, vamos adicionar a micro-região de origem ao df_nivel3
        df_nivel3_with_micro = df_nivel3.join(
            df.select('nivel_1_centro_origem', 'nivel_2_subcategoria', 'nivel_3_rota_especifica', 'microregiao_origem').dropDuplicates(),
            ['nivel_1_centro_origem', 'nivel_2_subcategoria', 'nivel_3_rota_especifica'],
            'left'
        )
        
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
        
        # 6. Calcular métricas finais
        df_hierarchical = df_hierarchical.withColumn(
            'frete_geral_brl_ton',
            when(col('volume_total_nivel3') > 0, 
                 col('custo_total_nivel3') / col('volume_total_nivel3'))
            .otherwise(0.0)
        )
        
        df_hierarchical = df_hierarchical.withColumn(
            'frete_geral_brl_ton_km',
            when(col('volume_total_nivel3') > 0, 
                 col('custo_total_nivel3') / (col('volume_total_nivel3') * col('distancia_media_nivel3')))
            .otherwise(0.0)
        )
        
        # 7. Calcular oportunidade
        df_hierarchical = df_hierarchical.withColumn(
            'oportunidade_brl_ton_km',
            when(col('preco_medio_microregiao').isNotNull(),
                 col('frete_geral_brl_ton_km') - col('preco_medio_microregiao'))
            .otherwise(0.0)
        )
        
        # Debug: verificar se os valores estão sendo calculados
        logger.info(f"Debug - Valores de oportunidade:")
        logger.info(f"  - frete_geral_brl_ton_km: {df_hierarchical.agg({'frete_geral_brl_ton_km': 'count'}).collect()[0]}")
        logger.info(f"  - preco_medio_microregiao: {df_hierarchical.agg({'preco_medio_microregiao': 'count'}).collect()[0]}")
        logger.info(f"  - oportunidade_brl_ton_km: {df_hierarchical.agg({'oportunidade_brl_ton_km': 'count'}).collect()[0]}")
        logger.info(f"  - Valores NaN em oportunidade: {df_hierarchical.filter(col('oportunidade_brl_ton_km').isNull()).count()}")
        
        # 8. Determinar ação
        df_hierarchical = df_hierarchical.withColumn(
            'acao',
            when(col('oportunidade_brl_ton_km') > 0.01, 'Redução')
            .when(col('oportunidade_brl_ton_km') < -0.01, 'Aumento')
            .otherwise('Manter')
        )
        
        # 9. Ordenar por hierarquia
        df_hierarchical = df_hierarchical.orderBy(
            'nivel_1_centro_origem',
            'nivel_2_subcategoria',
            'nivel_3_rota_especifica'
        )
        
        logger.info(f"Estrutura hierárquica criada: {df_hierarchical.count()} linhas")
        return df_hierarchical
    
    def generate_hierarchical_report(self, df_hierarchical):
        """
        Gera relatório hierárquico similar ao dashboard da imagem
        """
        logger.info("Gerando relatório hierárquico...")
        
        # Selecionar colunas para o relatório final
        df_report = df_hierarchical.select(
            'nivel_1_centro_origem',
            'nivel_2_subcategoria', 
            'nivel_3_rota_especifica',
            'volume_total_nivel3',
            'frete_geral_brl_ton',
            'frete_geral_brl_ton_km',
            'oportunidade_brl_ton_km',
            'acao'
        )
        
        # Renomear colunas para o formato final (usando aliases simples para evitar problemas)
        df_report = df_report.withColumnRenamed('nivel_1_centro_origem', 'Centro_Origem') \
                            .withColumnRenamed('nivel_2_subcategoria', 'Sub_Categoria') \
                            .withColumnRenamed('nivel_3_rota_especifica', 'Rota_Especifica') \
                            .withColumnRenamed('volume_total_nivel3', 'Volume_TON') \
                            .withColumnRenamed('frete_geral_brl_ton', 'Fr_Geral_BRL_TON') \
                            .withColumnRenamed('frete_geral_brl_ton_km', 'Fr_Geral_BRL_TON_KM') \
                            .withColumnRenamed('oportunidade_brl_ton_km', 'Oport_BRL_TON_KM')
        
        # Salvar resultados
        output_path = "outputs/analise_hierarquica"
        df_report.write.mode("overwrite").parquet(output_path)
        
        # Salvar relatório em formato legível
        df_report.toPandas().to_excel(
            "outputs/dashboard_hierarquico.xlsx", 
            index=False
        )
        
        # Retornar resumo
        total_rotas = df_report.count()
        rotas_reducao = df_report.filter(col('acao') == 'Redução').count()
        rotas_aumento = df_report.filter(col('acao') == 'Aumento').count()
        rotas_manter = df_report.filter(col('acao') == 'Manter').count()
        
        # Top oportunidades de redução
        top_reducoes = df_report.filter(col('acao') == 'Redução') \
            .orderBy(desc('Oport_BRL_TON_KM')) \
            .limit(5)
        
        # Top rotas por volume
        top_volume = df_report.orderBy(desc('Volume_TON')).limit(5)
        
        return {
            'total_rotas': total_rotas,
            'rotas_reducao': rotas_reducao,
            'rotas_aumento': rotas_aumento,
            'rotas_manter': rotas_manter,
            'top_reducoes': top_reducoes.toPandas().to_dict('records'),
            'top_volume': top_volume.toPandas().to_dict('records')
        }

def main():
    """
    Função principal para executar análise hierárquica
    """
    try:
        # Inicializar Spark
        try:
            from pyspark.sql import SparkSession
            spark = SparkSession.builder \
                .appName("AnaliseHierarquica") \
                .config("spark.sql.adaptive.enabled", "true") \
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
        
        # Preparar dados para hierarquia
        df_processed = analyzer.prepare_data_hierarchical(df_raw)
        
        # Criar estrutura hierárquica
        df_hierarchical = analyzer.create_hierarchical_structure(df_processed)
        
        # Gerar relatório
        report = analyzer.generate_hierarchical_report(df_hierarchical)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Mostrar resultados
        logger.info("=" * 80)
        logger.info("RELATÓRIO DA ANÁLISE HIERÁRQUICA")
        logger.info("=" * 80)
        logger.info(f"Tempo de execução: {execution_time:.2f} segundos")
        logger.info(f"Total de rotas analisadas: {report['total_rotas']}")
        logger.info(f"Oportunidades de redução: {report['rotas_reducao']}")
        logger.info(f"Oportunidades de aumento: {report['rotas_aumento']}")
        logger.info(f"Rotas para manter: {report['rotas_manter']}")
        
        logger.info("\nTOP 5 OPORTUNIDADES DE REDUÇÃO:")
        for i, op in enumerate(report['top_reducoes'][:5], 1):
            logger.info(f"{i}. {op['Rota_Especifica']}: {op['Oport_BRL_TON_KM']:.4f} BRL/TON/KM")
        
        logger.info("\nTOP 5 ROTAS POR VOLUME:")
        for i, rota in enumerate(report['top_volume'][:5], 1):
            logger.info(f"{i}. {rota['Rota_Especifica']}: {rota['Volume_TON']:,.1f} ton")
        
        logger.info(f"\nRelatório salvo em: outputs/dashboard_hierarquico.xlsx")
        logger.info("Análise hierárquica concluída com sucesso!")
        
    except Exception as e:
        logger.error(f"Erro durante a análise: {str(e)}")
        raise

if __name__ == "__main__":
    main()
