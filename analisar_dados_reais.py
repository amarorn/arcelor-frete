#!/usr/bin/env python3
"""
Script para analisar dados reais do sample_data_parquet.parquet
usando o TKUTrendAnalyzer
"""

import polars as pl
import logging
from utils.ml.tku_trend_analyzer import TKUTrendAnalyzer
import json
from datetime import datetime

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_parquet_data():
    """Carrega dados do arquivo Parquet"""
    logger.info("📊 Carregando dados do arquivo Parquet...")
    
    try:
        df = pl.read_parquet('sample_data_parquet.parquet')
        logger.info(f"✅ Dados carregados: {len(df):,} registros, {len(df.columns)} colunas")
        return df
    except Exception as e:
        logger.error(f"❌ Erro ao carregar dados: {str(e)}")
        return None


def analyze_route_performance(df):
    """Analisa performance das rotas"""
    logger.info("\n📈 ANÁLISE DE PERFORMANCE DAS ROTAS")
    logger.info("=" * 60)
    
    # Estatísticas gerais
    total_rotas = df['rota_municipio'].n_unique()
    total_entregas = len(df)
    periodo_inicio = df['data_faturamento'].min()
    periodo_fim = df['data_faturamento'].max()
    
    logger.info(f"🛣️  Total de rotas únicas: {total_rotas}")
    logger.info(f"📦 Total de entregas: {total_entregas:,}")
    logger.info(f"📅 Período: {periodo_inicio} a {periodo_fim}")
    
    # Top 10 rotas por volume
    logger.info("\n🏆 TOP 10 ROTAS POR VOLUME:")
    top_rotas_volume = df.group_by('rota_municipio').agg([
        pl.col('volume_ton').sum().alias('volume_total_ton'),
        pl.col('frete_brl').sum().alias('frete_total_brl'),
        pl.len().alias('num_entregas')
    ]).sort('volume_total_ton', descending=True).head(10)
    
    for i, row in enumerate(top_rotas_volume.iter_rows(named=True), 1):
        logger.info(f"  {i:2d}. {row['rota_municipio']}")
        logger.info(f"      📦 Volume: {row['volume_total_ton']:,.0f} ton")
        logger.info(f"      💰 Frete: R$ {row['frete_total_brl']:,.2f}")
        logger.info(f"      🚚 Entregas: {row['num_entregas']}")
    
    return top_rotas_volume


def analyze_tku_distribution(df):
    """Analisa distribuição de TKU"""
    logger.info("\n💰 ANÁLISE DE DISTRIBUIÇÃO TKU")
    logger.info("=" * 60)
    
    # Estatísticas de TKU
    tku_stats = df.select([
        pl.col('tku_calculado').mean().alias('tku_medio'),
        pl.col('tku_calculado').std().alias('tku_desvio'),
        pl.col('tku_calculado').min().alias('tku_min'),
        pl.col('tku_calculado').max().alias('tku_max'),
        pl.col('tku_calculado').quantile(0.25).alias('tku_q25'),
        pl.col('tku_calculado').quantile(0.50).alias('tku_q50'),
        pl.col('tku_calculado').quantile(0.75).alias('tku_q75')
    ])
    
    stats = tku_stats.to_dicts()[0]
    
    logger.info(f"📊 TKU Médio: R$ {stats['tku_medio']:.4f}/TON.KM")
    logger.info(f"📈 TKU Desvio Padrão: R$ {stats['tku_desvio']:.4f}")
    logger.info(f"📉 TKU Mínimo: R$ {stats['tku_min']:.4f}")
    logger.info(f"📈 TKU Máximo: R$ {stats['tku_max']:.4f}")
    logger.info(f"📊 TKU Q25: R$ {stats['tku_q25']:.4f}")
    logger.info(f"📊 TKU Q50 (Mediana): R$ {stats['tku_q50']:.4f}")
    logger.info(f"📊 TKU Q75: R$ {stats['tku_q75']:.4f}")
    
    # Classificar rotas por TKU
    logger.info("\n🏷️  CLASSIFICAÇÃO DE ROTAS POR TKU:")
    
    # Rotas com TKU alto (acima do Q75)
    rotas_alto_tku = df.filter(pl.col('tku_calculado') > stats['tku_q75']).group_by('rota_municipio').agg([
        pl.col('tku_calculado').mean().alias('tku_medio'),
        pl.len().alias('num_entregas')
    ]).sort('tku_medio', descending=True).head(5)
    
    logger.info("🔥 ROTAS COM TKU ALTO (acima do Q75):")
    for row in rotas_alto_tku.iter_rows(named=True):
        logger.info(f"  • {row['rota_municipio']}: R$ {row['tku_medio']:.4f} ({row['num_entregas']} entregas)")
    
    # Rotas com TKU baixo (abaixo do Q25)
    rotas_baixo_tku = df.filter(pl.col('tku_calculado') < stats['tku_q25']).group_by('rota_municipio').agg([
        pl.col('tku_calculado').mean().alias('tku_medio'),
        pl.len().alias('num_entregas')
    ]).sort('tku_medio').head(5)
    
    logger.info("\n❄️  ROTAS COM TKU BAIXO (abaixo do Q25):")
    for row in rotas_baixo_tku.iter_rows(named=True):
        logger.info(f"  • {row['rota_municipio']}: R$ {row['tku_medio']:.4f} ({row['num_entregas']} entregas)")
    
    return stats


def analyze_specific_routes(df, analyzer):
    """Analisa rotas específicas usando o TKUTrendAnalyzer"""
    logger.info("\n🧠 ANÁLISE DETALHADA DE ROTAS ESPECÍFICAS")
    logger.info("=" * 60)
    
    # Selecionar rotas para análise detalhada
    rotas_para_analisar = [
        'JOÃO MONLEVADE-MANAUS',      # Rota com mais volume
        'JOÃO MONLEVADE-BIGUAÇU',     # Rota representativa
        'JOÃO MONLEVADE-JOÃO MONLEVADE'  # Rota local
    ]
    
    resultados_analise = {}
    
    for rota in rotas_para_analisar:
        logger.info(f"\n📊 ANALISANDO ROTA: {rota}")
        logger.info("-" * 40)
        
        try:
            # Filtrar dados da rota
            df_rota = df.filter(pl.col('rota_municipio') == rota)
            
            if df_rota.is_empty():
                logger.warning(f"⚠️  Nenhum dado encontrado para rota: {rota}")
                continue
            
            # Estatísticas básicas da rota
            stats_rota = df_rota.select([
                pl.col('volume_ton').sum().alias('volume_total'),
                pl.col('frete_brl').sum().alias('frete_total'),
                pl.col('distancia_km').mean().alias('distancia_media'),
                pl.col('tku_calculado').mean().alias('tku_medio'),
                pl.len().alias('num_entregas')
            ]).to_dicts()[0]
            
            logger.info(f"📦 Volume total: {stats_rota['volume_total']:,.0f} ton")
            logger.info(f"💰 Frete total: R$ {stats_rota['frete_total']:,.2f}")
            logger.info(f"🛣️  Distância média: {stats_rota['distancia_media']:.0f} km")
            logger.info(f"💵 TKU médio: R$ {stats_rota['tku_medio']:.4f}")
            logger.info(f"🚚 Número de entregas: {stats_rota['num_entregas']}")
            
            # Análise completa com TKUTrendAnalyzer
            analysis = analyzer.analyze_route_tku_trends(df, rota)
            
            if 'erro' not in analysis:
                logger.info(f"🎯 Status: {analysis['status_geral']}")
                logger.info(f"📊 Variação 3M: {analysis['variacao_percentual']:+.1f}%")
                
                # Verificar tendências
                if 'tendencias' in analysis:
                    tendencia_3m = analysis['tendencias'].get('3_meses', {})
                    if tendencia_3m.get('dados_disponiveis'):
                        logger.info(f"📈 Tendência 3M: {tendencia_3m['tendencia']}")
                        logger.info(f"⚡ Velocidade: {tendencia_3m['velocidade']}")
                        logger.info(f"🎯 Confiança: {tendencia_3m['confianca']:.1f}%")
                
                # Verificar benchmark
                if 'benchmark' in analysis:
                    benchmark = analysis['benchmark']
                    logger.info(f"🏆 Posicionamento: {benchmark['posicionamento']}")
                    if 'economia_potencial' in benchmark:
                        logger.info(f"💰 Economia potencial: R$ {benchmark['economia_potencial']:.4f}")
                
                # Verificar recomendações
                if 'recomendacoes' in analysis and analysis['recomendacoes']:
                    rec = analysis['recomendacoes'][0]
                    logger.info(f"💡 Recomendação: {rec['acao']}")
                    logger.info(f"   • Tipo: {rec['tipo']}")
                    logger.info(f"   • Prioridade: {rec['prioridade']}")
                
                resultados_analise[rota] = analysis
                
            else:
                logger.error(f"❌ Erro na análise: {analysis['erro']}")
                
        except Exception as e:
            logger.error(f"❌ Erro ao analisar rota {rota}: {str(e)}")
    
    return resultados_analise


def generate_summary_report(df, resultados_analise):
    """Gera relatório resumido da análise"""
    logger.info("\n📋 RELATÓRIO RESUMIDO DA ANÁLISE")
    logger.info("=" * 60)
    
    # Estatísticas gerais
    total_rotas = df['rota_municipio'].n_unique()
    total_entregas = len(df)
    volume_total = df['volume_ton'].sum()
    frete_total = df['frete_brl'].sum()
    tku_medio = df['tku_calculado'].mean()
    
    logger.info(f"📊 RESUMO GERAL:")
    logger.info(f"  • Total de rotas: {total_rotas}")
    logger.info(f"  • Total de entregas: {total_entregas:,}")
    logger.info(f"  • Volume total: {volume_total:,.0f} ton")
    logger.info(f"  • Frete total: R$ {frete_total:,.2f}")
    logger.info(f"  • TKU médio: R$ {tku_medio:.4f}")
    
    # Resumo das análises
    if resultados_analise:
        logger.info(f"\n🎯 RESUMO DAS ANÁLISES:")
        for rota, analysis in resultados_analise.items():
            if 'erro' not in analysis:
                status = analysis.get('status_geral', 'N/A')
                tku_atual = analysis.get('tku_atual', 0)
                variacao = analysis.get('variacao_percentual', 0)
                
                logger.info(f"  • {rota}:")
                logger.info(f"    - Status: {status}")
                logger.info(f"    - TKU: R$ {tku_atual:.4f}")
                logger.info(f"    - Variação: {variacao:+.1f}%")
    
    # Recomendações gerais
    logger.info(f"\n💡 RECOMENDAÇÕES GERAIS:")
    logger.info(f"  1. Monitorar rotas com TKU acima da mediana")
    logger.info(f"  2. Analisar padrões sazonais para otimização")
    logger.info(f"  3. Implementar alertas para tendências de alta")
    logger.info(f"  4. Revisar contratos de rotas críticas")
    logger.info(f"  5. Usar TKUTrendAnalyzer para análises semanais")


def main():
    """Função principal"""
    logger.info("🚀 ANÁLISE DE DADOS REAIS COM TKUTrendAnalyzer")
    logger.info("=" * 60)
    
    # 1. Carregar dados
    df = load_parquet_data()
    if df is None:
        return
    
    # 2. Análise de performance das rotas
    top_rotas = analyze_route_performance(df)
    
    # 3. Análise de distribuição TKU
    tku_stats = analyze_tku_distribution(df)
    
    # 4. Inicializar TKUTrendAnalyzer
    logger.info("\n🔧 Inicializando TKUTrendAnalyzer...")
    analyzer = TKUTrendAnalyzer()
    
    # 5. Análise detalhada de rotas específicas
    resultados_analise = analyze_specific_routes(df, analyzer)
    
    # 6. Gerar relatório resumido
    generate_summary_report(df, resultados_analise)
    
    logger.info("\n✅ Análise concluída com sucesso!")
    logger.info("💡 Use o TKUTrendAnalyzer para análises contínuas das suas rotas")


if __name__ == "__main__":
    main()
