"""
Pipeline da Fase 2: Integração Modelo de Frete + Seletor de Transportadora
Combina previsão de preços com seleção inteligente de transportadoras
"""

import logging
import time
import os
import sys
import polars as pl

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
from utils.ml.transportadora_selector import TransportadoraSelector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_fase2_pipeline(data_file: str = None):
    """
    Executa o pipeline completo da Fase 2
    """
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
    logger.info("🚀 Iniciando Pipeline da Fase 2: Frete + Seleção de Transportadora")
    logger.info(f"📁 Arquivo de dados: {data_file}")
    
    try:
        # 1. Inicializar adaptador Polars
        logger.info("📊 Inicializando adaptador Polars...")
        polars_adapter = PolarsAdapter()
        
        # 2. Carregar e preparar dados
        logger.info("📥 Carregando dados...")
        df_raw = polars_adapter.read_excel(data_file)
        
        logger.info("🔧 Preparando dados...")
        df_processed = polars_adapter.prepare_data(df_raw)
        
        # 3. Análise exploratória
        logger.info("📈 Realizando análise exploratória...")
        
        route_metrics = polars_adapter.calculate_route_metrics(df_processed)
        logger.info(f"🏆 Top 5 rotas por volume:")
        logger.info(route_metrics.head().to_pandas())
        
        carrier_metrics = polars_adapter.calculate_carrier_metrics(df_processed)
        logger.info(f"🚛 Top 5 transportadoras por volume:")
        logger.info(carrier_metrics.head().to_pandas())
        
        # 4. Treinar modelo de previsão de frete (Fase 1)
        logger.info("🤖 Iniciando treinamento do modelo de frete...")
        frete_predictor = FretePricePredictor()
        
        frete_metrics = frete_predictor.train(df_processed)
        logger.info(f"✅ Modelo de frete treinado - R²: {frete_metrics['test_r2']:.4f}")
        
        frete_predictor.save_model()
        
        # 5. Treinar modelo de seleção de transportadora (Fase 2)
        logger.info("🎯 Iniciando treinamento do seletor de transportadora...")
        transportadora_selector = TransportadoraSelector()
        
        # Preparar dados para o seletor (adicionar colunas necessárias)
        df_for_selector = df_processed.with_columns([
            pl.col('transportadora').alias('transportadora'),
            pl.col('rota').alias('rota'),
            pl.col('volume_ton').alias('volume_ton'),
            pl.col('distancia_km').alias('distancia_km'),
            pl.col('preco_ton_km_calculado').alias('preco_ton_km'),
            pl.col('frete_brl').alias('frete_brl'),
            pl.col('mes').alias('mes'),
            pl.col('trimestre').alias('trimestre'),
            pl.col('ano').alias('ano')
        ])
        
        selector_metrics = transportadora_selector.train(df_for_selector)
        logger.info(f"✅ Seletor de transportadora treinado - Acurácia: {selector_metrics['test_accuracy']:.4f}")
        
        transportadora_selector.save_model()
        
        # 6. Gerar relatórios integrados
        logger.info("📋 Gerando relatórios integrados...")
        
        # Relatório de otimização de frete
        frete_report = frete_predictor.generate_optimization_report(df_processed)
        
        # Relatório de seleção de transportadora
        transportadora_report = transportadora_selector.generate_selection_report(df_for_selector)
        
        # 7. Análise integrada: combinar previsões de frete com seleção de transportadora
        logger.info("🔗 Realizando análise integrada...")
        
        # Fazer previsões de frete
        frete_predictions = frete_predictor.predict(df_processed)
        
        # Fazer previsões de transportadora
        df_with_carrier_predictions = transportadora_selector.predict_transportadora(df_for_selector)
        
        # Combinar previsões
        df_integrated = df_processed.with_columns([
            pl.Series("preco_frete_previsto", frete_predictions),
            pl.Series("economia_potencial", 
                     (pl.col("preco_ton_km_calculado") - pl.Series("preco_frete_previsto", frete_predictions)) * 
                     pl.col("volume_ton") * pl.col("distancia_km"))
        ])
        
        # Juntar com previsões de transportadora
        df_integrated = df_integrated.join(
            df_with_carrier_predictions.select(['rota', 'transportadora_recomendada_ml', 'score_confianca_ml']),
            on='rota', how='left'
        )
        
        # 8. Exportar resultados
        logger.info("💾 Exportando resultados...")
        
        # Criar pasta outputs se não existir
        try:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            outputs_dir = os.path.join(project_root, "outputs")
        except NameError:
            outputs_dir = "../outputs"
        
        os.makedirs(outputs_dir, exist_ok=True)
        
        # Exportar métricas de rotas e transportadoras
        polars_adapter.export_to_excel(
            route_metrics, 
            os.path.join(outputs_dir, "fase2_metricas_rotas.xlsx")
        )
        
        polars_adapter.export_to_excel(
            carrier_metrics, 
            os.path.join(outputs_dir, "fase2_metricas_transportadoras.xlsx")
        )
        
        # Exportar dados integrados
        polars_adapter.export_to_excel(
            df_integrated, 
            os.path.join(outputs_dir, "fase2_dados_integrados.xlsx")
        )
        
        # Exportar relatórios em JSON
        import json
        
        # Relatório de frete
        with open(os.path.join(outputs_dir, "fase2_relatorio_frete.json"), "w") as f:
            json.dump(frete_report, f, indent=2, default=str)
        
        # Relatório de seleção de transportadora
        with open(os.path.join(outputs_dir, "fase2_relatorio_transportadora.json"), "w") as f:
            json.dump(transportadora_report, f, indent=2, default=str)
        
        # Relatório integrado
        integrated_report = {
            'resumo_fase2': {
                'total_registros_processados': len(df_processed),
                'total_rotas_analisadas': df_processed['rota'].n_unique(),
                'total_transportadoras_analisadas': df_processed['transportadora'].n_unique(),
                'modelo_frete_r2': frete_metrics['test_r2'],
                'seletor_transportadora_acuracia': selector_metrics['test_accuracy']
            },
            'economia_potencial_total': float(df_integrated['economia_potencial'].sum()),
            'top_rotas_economia': df_integrated.group_by('rota').agg([
                pl.col('economia_potencial').sum().alias('economia_total'),
                pl.col('volume_ton').sum().alias('volume_total'),
                pl.col('transportadora_recomendada_ml').first().alias('transportadora_recomendada')
            ]).sort('economia_total', descending=True).head(10).to_pandas().to_dict('records'),
            'metricas_modelos': {
                'frete': frete_metrics,
                'transportadora': selector_metrics
            }
        }
        
        with open(os.path.join(outputs_dir, "fase2_relatorio_integrado.json"), "w") as f:
            json.dump(integrated_report, f, indent=2, default=str)
        
        execution_time = time.time() - start_time
        logger.info("🎉 Pipeline da Fase 2 concluído com sucesso!")
        logger.info(f"⏱️ Tempo de execução: {execution_time:.2f} segundos")
        
        logger.info("📊 RESUMO DOS RESULTADOS:")
        logger.info(f"   • Modelo de Frete - R²: {frete_metrics['test_r2']:.4f}")
        logger.info(f"   • Seletor de Transportadora - Acurácia: {selector_metrics['test_accuracy']:.4f}")
        logger.info(f"   • Economia Potencial Total: R$ {integrated_report['economia_potencial_total']:,.2f}")
        logger.info(f"   • Rotas Analisadas: {integrated_report['resumo_fase2']['total_rotas_analisadas']}")
        logger.info(f"   • Transportadoras Analisadas: {integrated_report['resumo_fase2']['total_transportadoras_analisadas']}")
        
        return {
            'success': True,
            'execution_time': execution_time,
            'frete_metrics': frete_metrics,
            'selector_metrics': selector_metrics,
            'integrated_report': integrated_report
        }
        
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"❌ Erro no pipeline da Fase 2: {e}")
        
        return {
            'success': False,
            'execution_time': execution_time,
            'error': str(e)
        }


def benchmark_fase2_performance(data_file: str = None):
    """
    Benchmark de performance da Fase 2
    """
    logger.info("🏃 Iniciando benchmark de performance da Fase 2...")
    
    start_time = time.time()
    result = run_fase2_pipeline(data_file)
    end_time = time.time()
    
    if result['success']:
        logger.info("✅ Benchmark concluído com sucesso!")
        logger.info(f"⏱️ Tempo total: {result['execution_time']:.2f} segundos")
        logger.info(f"📊 Performance dos modelos:")
        logger.info(f"   • Frete: R² = {result['frete_metrics']['test_r2']:.4f}")
        logger.info(f"   • Transportadora: Acurácia = {result['selector_metrics']['test_accuracy']:.4f}")
        logger.info(f"💰 Economia potencial: R$ {result['integrated_report']['economia_potencial_total']:,.2f}")
    else:
        logger.error(f"❌ Benchmark falhou: {result['error']}")
    
    return result


if __name__ == "__main__":
    # Executar pipeline da Fase 2
    result = run_fase2_pipeline()
    
    if result['success']:
        print("\n🎉 Pipeline da Fase 2 executado com sucesso!")
        print(f"⏱️ Tempo de execução: {result['execution_time']:.2f} segundos")
        print(f"📊 Resultados disponíveis na pasta 'outputs/'")
    else:
        print(f"\n❌ Erro na execução: {result['error']}")
        sys.exit(1)
