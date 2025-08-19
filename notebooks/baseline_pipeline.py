

import logging
import time
import os
import sys

# Configurar caminhos
try:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)
except NameError:
    project_root = ".."
    sys.path.insert(0, project_root)

import polars as pl
from utils.adapters.polars_adapter import PolarsAdapter
from utils.ml.baseline_price_predictor import BaselinePricePredictor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_baseline_pipeline(data_file: str = None):
    if data_file is None:
        try:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_file = os.path.join(project_root, "sample_data.xlsx")
        except NameError:
            data_file = "../sample_data.xlsx"
    
    start_time = time.time()
    logger.info("Iniciando Pipeline Baseline: Feature Engineering + Múltiplos Modelos")
    logger.info(f"Arquivo de dados: {data_file}")
    
    try:
        logger.info("Inicializando adaptador Polars...")
        polars_adapter = PolarsAdapter()
        
        logger.info("Carregando dados...")
        df_raw = polars_adapter.read_excel(data_file)
        
        logger.info("Preparando dados básicos...")
        df_processed = polars_adapter.prepare_data(df_raw)
        
        logger.info("Inicializando predictor baseline...")
        baseline_predictor = BaselinePricePredictor()
        
        logger.info("Criando features avançadas...")
        df_enhanced = baseline_predictor.create_advanced_features(df_processed)
        
        logger.info("Analisando features criadas...")
        numeric_cols = [col for col in df_enhanced.columns if df_enhanced[col].dtype in [pl.Float64, pl.Int64, pl.Int8]]
        logger.info(f"Features numéricas criadas: {len(numeric_cols)}")
        logger.info(f"Features categóricas: {len(baseline_predictor.categorical_features)}")
        
        logger.info("Iniciando treinamento dos modelos baseline...")
        results = baseline_predictor.train_baseline_models(df_enhanced)
        
        logger.info("Otimizando hiperparâmetros...")
        tuning_results = baseline_predictor.hyperparameter_tuning(df_enhanced)
        
        logger.info("Salvando modelo baseline...")
        baseline_predictor.save_baseline_model()
        
        logger.info("Gerando relatório baseline...")
        baseline_report = baseline_predictor.generate_baseline_report(df_enhanced, results)
        
        logger.info("Exportando resultados...")
        
        try:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            outputs_dir = os.path.join(project_root, "outputs")
        except NameError:
            outputs_dir = "../outputs"
        
        os.makedirs(outputs_dir, exist_ok=True)
        
        # Salvar relatório JSON
        import json
        with open(os.path.join(outputs_dir, "baseline_relatorio.json"), "w") as f:
            json.dump(baseline_report, f, indent=2, default=str)
        
        # Salvar dados com features
        polars_adapter.export_to_excel(
            df_enhanced, 
            os.path.join(outputs_dir, "baseline_dados_com_features.xlsx")
        )
        
        # Salvar ranking dos modelos
        if baseline_report['model_ranking']:
            ranking_df = pl.DataFrame(baseline_report['model_ranking'])
            polars_adapter.export_to_excel(
                ranking_df,
                os.path.join(outputs_dir, "baseline_ranking_modelos.xlsx")
            )
        
        execution_time = time.time() - start_time
        logger.info("Pipeline Baseline concluído com sucesso!")
        logger.info(f"Tempo de execução: {execution_time:.2f} segundos")
        
        logger.info("RESUMO DOS RESULTADOS:")
        logger.info(f"Total de modelos treinados: {baseline_report['training_summary']['total_models']}")
        logger.info(f"Modelos com sucesso: {baseline_report['training_summary']['successful_models']}")
        logger.info(f"Melhor modelo: {baseline_report['best_model']['name']}")
        logger.info(f"Score R²: {baseline_report['best_model']['score']:.4f}")
        
        if baseline_report['model_ranking']:
            logger.info("Top 3 modelos:")
            for i, model in enumerate(baseline_report['model_ranking'][:3]):
                logger.info(f"{i+1}. {model['model']}: R² = {model['test_r2']:.4f}")
        
        return baseline_report
        
    except Exception as e:
        logger.error(f"Erro no pipeline baseline: {e}")
        raise


def run_feature_analysis(data_file: str = None):
    if data_file is None:
        try:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_file = os.path.join(project_root, "sample_data.xlsx")
        except NameError:
            data_file = "../sample_data.xlsx"
    
    logger.info("Iniciando análise de features...")
    
    try:
        polars_adapter = PolarsAdapter()
        df_raw = polars_adapter.read_excel(data_file)
        df_processed = polars_adapter.prepare_data(df_raw)
        
        baseline_predictor = BaselinePricePredictor()
        df_enhanced = baseline_predictor.create_advanced_features(df_processed)
        
        logger.info("Análise estatística das features...")
        
        numeric_cols = [col for col in df_enhanced.columns if df_enhanced[col].dtype in [pl.Float64, pl.Int64, pl.Int8]]
        numeric_cols = [col for col in numeric_cols if col != 'preco_ton_km_calculado']
        
        logger.info(f"Features numéricas criadas: {len(numeric_cols)}")
        for col in numeric_cols[:10]:
            try:
                stats = df_enhanced.select([
                    pl.col(col).mean().alias('mean'),
                    pl.col(col).std().alias('std'),
                    pl.col(col).min().alias('min'),
                    pl.col(col).max().alias('max')
                ])
                
                if stats.height > 0:
                    logger.info(f"{col}: mean={stats[0, 'mean']:.4f}, std={stats[0, 'std']:.4f}")
            except Exception as e:
                logger.warning(f"Erro ao calcular estatísticas para {col}: {e}")
        
        logger.info("Analisando correlação com target...")
        target_correlations = {}
        
        for col in numeric_cols[:10]:
            try:
                corr = df_enhanced.select([
                    pl.corr(pl.col(col), pl.col('preco_ton_km_calculado'))
                ]).collect()
                
                if corr and corr[0][0] is not None:
                    target_correlations[col] = abs(corr[0][0])
            except:
                continue
        
        sorted_correlations = sorted(target_correlations.items(), key=lambda x: x[1], reverse=True)
        
        logger.info("Top 10 features por correlação com target:")
        for i, (feature, corr) in enumerate(sorted_correlations[:10]):
            logger.info(f"{i+1}. {feature}: {corr:.4f}")
        
        return {
            'total_features': len(numeric_cols),
            'top_correlations': sorted_correlations[:10]
        }
        
    except Exception as e:
        logger.error(f"Erro na análise de features: {e}")
        raise


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Pipeline Baseline de Preços")
    parser.add_argument("--data", help="Caminho para arquivo de dados")
    parser.add_argument("--analysis", action="store_true", help="Executar análise de features")
    
    args = parser.parse_args()
    
    if args.analysis:
        run_feature_analysis(args.data)
    else:
        run_baseline_pipeline(args.data)
