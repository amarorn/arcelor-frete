import logging
import time
import os
import sys

# Configurar caminhos para funcionar tanto localmente quanto no Databricks
try:
    # Se executando localmente
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)
except NameError:
    # Se executando no Databricks, usar caminho relativo
    project_root = ".."
    sys.path.insert(0, project_root)

from utils.adapters.polars_adapter import PolarsAdapter
from utils.ml.frete_price_predictor import FretePricePredictor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_fase1_pipeline(data_file: str = None):
    if data_file is None:
        # Buscar na pasta raiz do projeto
        try:
            # Tentar caminho local
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_file = os.path.join(project_root, "sample_data.xlsx")
        except NameError:
            # Caminho para Databricks
            data_file = "../sample_data.xlsx"
    
    start_time = time.time()
    logger.info("Iniciando Pipeline da Fase 1: Migração Polars + ML")
    logger.info(f"Arquivo de dados: {data_file}")
    
    try:
        logger.info("Inicializando adaptador Polars...")
        polars_adapter = PolarsAdapter()
        
        logger.info("Carregando dados...")
        df_raw = polars_adapter.read_excel(data_file)
        
        logger.info("Preparando dados...")
        df_processed = polars_adapter.prepare_data(df_raw)
        
        logger.info("Realizando análise exploratória...")
        
        route_metrics = polars_adapter.calculate_route_metrics(df_processed)
        logger.info(f"Top 5 rotas por volume:")
        logger.info(route_metrics.head().to_pandas())
        
        carrier_metrics = polars_adapter.calculate_carrier_metrics(df_processed)
        logger.info(f"Top 5 transportadoras por volume:")
        logger.info(carrier_metrics.head().to_pandas())
        
        logger.info("Iniciando treinamento do modelo ML...")
        predictor = FretePricePredictor()
        
        metrics = predictor.train(df_processed)
        
        predictor.save_model()
        
        logger.info("Gerando relatório de otimização...")
        optimization_report = predictor.generate_optimization_report(df_processed)
        
        logger.info("Exportando resultados...")
        
        # Criar pasta outputs se não existir
        try:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            outputs_dir = os.path.join(project_root, "outputs")
        except NameError:
            outputs_dir = "../outputs"
        
        os.makedirs(outputs_dir, exist_ok=True)
        
        polars_adapter.export_to_excel(
            route_metrics, 
            os.path.join(outputs_dir, "fase1_metricas_rotas.xlsx")
        )
        
        polars_adapter.export_to_excel(
            carrier_metrics, 
            os.path.join(outputs_dir, "fase1_metricas_transportadoras.xlsx")
        )
        
        import json
        with open(os.path.join(outputs_dir, "fase1_relatorio_otimizacao.json"), "w") as f:
            json.dump(optimization_report, f, indent=2, default=str)
        
        execution_time = time.time() - start_time
        logger.info("Pipeline da Fase 1 concluído com sucesso!")
        logger.info(f"Tempo de execução: {execution_time:.2f} segundos")
        
        logger.info("RESUMO DOS RESULTADOS:")
        logger.info(f"   • Dados processados: {df_processed.shape[0]:,} registros")
        logger.info(f"   • Rotas analisadas: {df_processed['rota'].n_unique()}")
        logger.info(f"   • Transportadoras: {df_processed['transportadora'].n_unique()}")
        logger.info(f"   • Modelo ML - R² treino: {metrics['train_r2']:.4f}")
        logger.info(f"   • Modelo ML - R² teste: {metrics['test_r2']:.4f}")
        logger.info(f"   • Oportunidades identificadas: {optimization_report['num_oportunidades']}")
        logger.info(f"   • Economia potencial: R$ {optimization_report['economia_potencial_total']:,.2f}")
        
        return {
            'success': True,
            'execution_time': execution_time,
            'metrics': metrics,
            'optimization_report': optimization_report
        }
        
    except Exception as e:
        logger.error(f"Erro no pipeline: {e}")
        return {'success': False, 'error': str(e)}


def benchmark_performance(data_file: str = None):
    if data_file is None:
        # Buscar na pasta raiz do projeto
        try:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_file = os.path.join(project_root, "sample_data.xlsx")
        except NameError:
            data_file = "../sample_data.xlsx"
    
    logger.info("Iniciando benchmark de performance...")
    logger.info(f"Arquivo de dados: {data_file}")
    
    try:
        import pandas as pd
        import polars as pl
        
        start_time = time.time()
        df_pandas = pd.read_excel(data_file)
        pandas_time = time.time() - start_time
        
        start_time = time.time()
        df_polars = pl.read_excel(data_file)
        polars_time = time.time() - start_time
        
        speedup = pandas_time / polars_time
        
        logger.info("RESULTADOS DO BENCHMARK:")
        logger.info(f"   • Pandas: {pandas_time:.2f}s")
        logger.info(f"   • Polars: {polars_time:.2f}s")
        logger.info(f"   • Speedup: {speedup:.1f}x mais rápido com Polars!")
        
        return {
            'pandas_time': pandas_time,
            'polars_time': polars_time,
            'speedup': speedup
        }
        
    except Exception as e:
        logger.error(f"Erro no benchmark: {e}")
        return None


if __name__ == "__main__":
    # Criar pastas necessárias
    try:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        outputs_dir = os.path.join(project_root, "outputs")
        models_dir = os.path.join(project_root, "models")
    except NameError:
        outputs_dir = "../outputs"
        models_dir = "../models"
    
    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    benchmark_results = benchmark_performance()
    
    results = run_fase1_pipeline()
    
    if results['success']:
        logger.info("Fase 1 implementada com sucesso!")
        logger.info("Verifique os arquivos em: outputs/ e models/")
    else:
        logger.error("Falha na implementação da Fase 1")
        logger.error(f"Erro: {results['error']}")
