#!/usr/bin/env python3
"""
Exemplo de uso do TKUTrendAnalyzer
Demonstra como analisar tendências de TKU para rotas específicas
"""

import logging
import sys
import os
from pathlib import Path

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Adicionar o diretório raiz ao path
try:
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))
except NameError:
    project_root = "."

# Importar o analisador
from utils.ml.tku_trend_analyzer import TKUTrendAnalyzer
import polars as pl

def create_sample_data():
    """Cria dados de exemplo para demonstração"""
    import polars as pl
    from datetime import datetime, timedelta
    import numpy as np
    
    # Configurar seed para reprodutibilidade
    np.random.seed(42)
    
    # Criar datas de exemplo (últimos 18 meses)
    start_date = datetime.now() - timedelta(days=540)
    dates = [start_date + timedelta(days=i*7) for i in range(78)]  # 78 semanas
    
    # Criar dados de exemplo
    data = []
    for i, date in enumerate(dates):
        # Simular tendência de alta com sazonalidade
        base_tku = 0.8 + (i * 0.002)  # Tendência de alta gradual
        
        # Adicionar sazonalidade (dezembro/janeiro mais caros)
        seasonal_factor = 1.0
        if date.month in [12, 1]:
            seasonal_factor = 1.3  # 30% mais caro
        elif date.month in [6, 7]:
            seasonal_factor = 0.9  # 10% mais barato
        
        # Adicionar ruído aleatório
        noise = np.random.normal(0, 0.05)
        
        # TKU final
        tku = (base_tku * seasonal_factor) + noise
        
        # Volume e distância
        volume = np.random.uniform(50, 200)
        distancia = np.random.uniform(100, 500)
        
        # Calcular frete
        frete = tku * volume * distancia
        
        data.append({
            'data_faturamento': date,
            'rota_municipio': 'SABARÁ',
            'rota_microregiao': 'METROPOLITANA',
            'rota_mesoregiao': 'METROPOLITANA',
            'microregiao_origem': 'JOÃO MONLEVADE',
            'volume_ton': volume,
            'distancia_km': distancia,
            'frete_brl': frete,
            'modal': 'RODOVIARIO',
            'tipo_rodovia': 'BR-381',
            'tipo_veiculo': 'TRUCK'
        })
    
    return pl.DataFrame(data)

def demonstrate_tku_analysis():
    """Demonstra a análise de tendências de TKU"""
    logger.info("🚀 Iniciando demonstração do TKUTrendAnalyzer")
    
    # 1. Criar dados de exemplo
    logger.info("📊 Criando dados de exemplo...")
    df = create_sample_data()
    logger.info(f"Dados criados: {len(df)} entregas")
    
    # 2. Inicializar analisador
    logger.info("🔧 Inicializando analisador...")
    analyzer = TKUTrendAnalyzer()
    
    # 3. Analisar rota específica
    rota_id = 'SABARÁ'
    logger.info(f"📈 Analisando tendências para rota: {rota_id}")
    
    analysis = analyzer.analyze_route_tku_trends(df, rota_id)
    
    # 4. Exibir resultados
    logger.info("=" * 80)
    logger.info("📋 RELATÓRIO DE ANÁLISE DE TENDÊNCIAS TKU")
    logger.info("=" * 80)
    
    if 'erro' in analysis:
        logger.error(f"❌ Erro na análise: {analysis['erro']}")
        return
    
    # Informações básicas
    logger.info(f"🛣️  Rota: {analysis['rota_id']}")
    logger.info(f"📅 Período analisado: {analysis['periodo_analise']['inicio'][:10]} a {analysis['periodo_analise']['fim'][:10]}")
    logger.info(f"📦 Total de entregas: {analysis['periodo_analise']['total_entregas']}")
    logger.info(f"💰 TKU Atual: R$ {analysis['tku_atual']:.4f}/TON.KM")
    logger.info(f"📊 Variação (3 meses): {analysis['variacao_percentual']:+.1f}%")
    logger.info(f"🎯 Status Geral: {analysis['status_geral']}")
    
    # Análise de tendências
    logger.info("\n📈 ANÁLISE DE TENDÊNCIAS:")
    logger.info("-" * 40)
    
    for periodo, tendencia in analysis['tendencias'].items():
        if tendencia.get('dados_disponiveis'):
            logger.info(f"• {periodo}:")
            logger.info(f"  - Tendência: {tendencia['tendencia']}")
            logger.info(f"  - Velocidade: {tendencia['velocidade']}")
            logger.info(f"  - Confiança: {tendencia['confianca']:.1f}%")
            logger.info(f"  - TKU Atual: R$ {tendencia['tku_atual']:.4f}")
            logger.info(f"  - Variação: {tendencia['variacao_percentual']:+.1f}%")
        else:
            logger.info(f"• {periodo}: Dados insuficientes")
    
    # Análise de sazonalidade
    logger.info("\n🌍 ANÁLISE DE SAZONALIDADE:")
    logger.info("-" * 40)
    
    sazonalidade = analysis['sazonalidade']
    if sazonalidade.get('padrao_sazonal_detectado'):
        logger.info(f"✅ Padrão sazonal detectado!")
        logger.info(f"📊 TKU médio geral: R$ {sazonalidade['tku_medio_geral']:.4f}")
        logger.info(f"📈 Variação sazonal: R$ {sazonalidade['variacao_sazonal']:.4f}")
        
        if sazonalidade['meses_alta_temporada']:
            logger.info(f"🔥 Meses de alta temporada: {', '.join(sazonalidade['meses_alta_temporada'])}")
        
        if sazonalidade['meses_baixa_temporada']:
            logger.info(f"❄️  Meses de baixa temporada: {', '.join(sazonalidade['meses_baixa_temporada'])}")
    else:
        logger.info("❌ Padrão sazonal não detectado")
    
    # Benchmark de mercado
    logger.info("\n🏆 BENCHMARK DE MERCADO:")
    logger.info("-" * 40)
    
    benchmark = analysis['benchmark']
    logger.info(f"📍 Micro-região analisada: {benchmark['microregiao_analisada']}")
    logger.info(f"💰 TKU da rota: R$ {benchmark['tku_rota']:.4f}")
    logger.info(f"📊 TKU benchmark: R$ {benchmark['tku_benchmark']:.4f}")
    logger.info(f"📈 Posicionamento: {benchmark['posicionamento']}")
    logger.info(f"💡 Recomendação: {benchmark['recomendacao']}")
    logger.info(f"🎯 Ação necessária: {benchmark['acao_necessaria']}")
    logger.info(f"💰 Economia potencial: R$ {benchmark['economia_potencial']:.4f}/TON.KM")
    
    # Recomendações
    logger.info("\n💡 RECOMENDAÇÕES:")
    logger.info("-" * 40)
    
    for i, rec in enumerate(analysis['recomendacoes'], 1):
        logger.info(f"{i}. {rec['acao']}")
        logger.info(f"   • Tipo: {rec['tipo']}")
        logger.info(f"   • Prioridade: {rec['prioridade']}")
        logger.info(f"   • Justificativa: {rec['justificativa']}")
        logger.info(f"   • Economia esperada: {rec['economia_esperada']}")
        logger.info(f"   • Prazo: {rec['prazo']}")
        logger.info(f"   • Dificuldade: {rec['dificuldade']}")
        logger.info("")  # Linha em branco
    
    # Resumo executivo
    logger.info("\n📋 RESUMO EXECUTIVO:")
    logger.info("-" * 40)
    
    summary = analyzer.get_analysis_summary(rota_id)
    logger.info(f"🛣️  Rota: {summary['rota_id']}")
    logger.info(f"🎯 Status: {summary['status_geral']}")
    logger.info(f"💰 TKU Atual: R$ {summary['tku_atual']:.4f}")
    logger.info(f"📊 Variação: {summary['variacao_percentual']:+.1f}%")
    logger.info(f"📈 Tendência 3M: {summary['tendencia_3m']}")
    logger.info(f"🏆 Posicionamento: {summary['posicionamento_mercado']}")
    logger.info(f"💡 Ação Principal: {summary['recomendacao_principal']}")
    
    logger.info("\n✅ Demonstração concluída com sucesso!")

def demonstrate_multiple_routes():
    """Demonstra análise de múltiplas rotas"""
    logger.info("\n🔄 DEMONSTRAÇÃO DE MÚLTIPLAS ROTAS")
    logger.info("=" * 80)
    
    # Criar dados para múltiplas rotas
    df = create_sample_data()
    
    # Adicionar mais rotas
    rotas_adicionais = [
        {'rota': 'CONTAGEM', 'tendencia': 'estavel', 'tku_base': 0.75},
        {'rota': 'SANTA LUZIA', 'tendencia': 'caindo', 'tku_base': 0.70},
        {'rota': 'BETIM', 'tendencia': 'subindo', 'tku_base': 0.90}
    ]
    
    # Expandir dados com novas rotas
    analyzer = TKUTrendAnalyzer()
    
    for rota_info in rotas_adicionais:
        logger.info(f"\n📊 Analisando rota: {rota_info['rota']}")
        
        # Criar dados específicos para esta rota
        df_rota = df.with_columns([
            pl.lit(rota_info['rota']).alias('rota_municipio'),
            pl.lit(rota_info['rota']).alias('rota_microregiao')
        ])
        
        # Ajustar TKU baseado na tendência
        if rota_info['tendencia'] == 'subindo':
            df_rota = df_rota.with_columns([
                (pl.col('frete_brl') * 1.2).alias('frete_brl')  # +20%
            ])
        elif rota_info['tendencia'] == 'caindo':
            df_rota = df_rota.with_columns([
                (pl.col('frete_brl') * 0.8).alias('frete_brl')  # -20%
            ])
        
        # Combinar com dados originais
        df = pl.concat([df, df_rota])
    
    # Analisar cada rota
    rotas_para_analisar = ['SABARÁ', 'CONTAGEM', 'SANTA LUZIA', 'BETIM']
    
    for rota_id in rotas_para_analisar:
        logger.info(f"\n{'='*60}")
        logger.info(f"📊 ANÁLISE COMPLETA: {rota_id}")
        logger.info(f"{'='*60}")
        
        analysis = analyzer.analyze_route_tku_trends(df, rota_id)
        
        if 'erro' not in analysis:
            summary = analyzer.get_analysis_summary(rota_id)
            logger.info(f"Status: {summary['status_geral']}")
            logger.info(f"TKU: R$ {summary['tku_atual']:.4f}")
            logger.info(f"Tendência: {summary['tendencia_3m']}")
            logger.info(f"Posicionamento: {summary['posicionamento_mercado']}")
            logger.info(f"Ação: {summary['recomendacao_principal']}")
        else:
            logger.error(f"Erro na análise: {analysis['erro']}")

if __name__ == "__main__":
    try:
        # Demonstração básica
        demonstrate_tku_analysis()
        
        # Demonstração de múltiplas rotas
        demonstrate_multiple_routes()
        
    except Exception as e:
        logger.error(f"Erro durante demonstração: {str(e)}")
        import traceback
        traceback.print_exc()
