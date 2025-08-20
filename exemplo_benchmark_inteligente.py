#!/usr/bin/env python3
"""
Exemplo de uso do Sistema de Benchmark Inteligente
Demonstra diferentes estratégias para cálculo automático do volume mínimo
"""

import logging
from analise_agrupamento_hierarquico import HierarchicalGroupingAnalyzer
from pyspark.sql import SparkSession

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def demonstrar_benchmark_inteligente():
    """
    Demonstra diferentes estratégias de benchmark inteligente
    """
    try:
        # Inicializar Spark
        spark = SparkSession.builder \
            .appName("BenchmarkInteligenteDemo") \
            .master("local[*]") \
            .getOrCreate()
        
        # Inicializar analisador
        analyzer = HierarchicalGroupingAnalyzer(spark)
        
        logger.info("=" * 80)
        logger.info("DEMONSTRAÇÃO DO SISTEMA DE BENCHMARK INTELIGENTE")
        logger.info("=" * 80)
        
        # ESTRATÉGIA 1: Benchmark Conservador (Percentil 90)
        logger.info("\n🎯 ESTRATÉGIA 1: BENCHMARK CONSERVADOR")
        logger.info("Objetivo: Máxima qualidade, poucas rotas")
        analyzer.configurar_benchmark_inteligente(
            percentil_benchmark=90,
            min_rotas_benchmark=5,
            max_rotas_benchmark=30,
            adaptativo_benchmark=True
        )
        
        # ESTRATÉGIA 2: Benchmark Equilibrado (Percentil 75)
        logger.info("\n⚖️ ESTRATÉGIA 2: BENCHMARK EQUILIBRADO")
        logger.info("Objetivo: Boa qualidade, quantidade moderada de rotas")
        analyzer.configurar_benchmark_inteligente(
            percentil_benchmark=75,
            min_rotas_benchmark=10,
            max_rotas_benchmark=50,
            adaptativo_benchmark=True
        )
        
        # ESTRATÉGIA 3: Benchmark Inclusivo (Percentil 50)
        logger.info("\n📊 ESTRATÉGIA 3: BENCHMARK INCLUSIVO")
        logger.info("Objetivo: Mais rotas, qualidade moderada")
        analyzer.configurar_benchmark_inteligente(
            percentil_benchmark=50,
            min_rotas_benchmark=20,
            max_rotas_benchmark=80,
            adaptativo_benchmark=True
        )
        
        # ESTRATÉGIA 4: Benchmark Adaptativo (Percentil 25)
        logger.info("\n🔄 ESTRATÉGIA 4: BENCHMARK ADAPTATIVO")
        logger.info("Objetivo: Máxima quantidade de rotas, qualidade ajustável")
        analyzer.configurar_benchmark_inteligente(
            percentil_benchmark=25,
            min_rotas_benchmark=30,
            max_rotas_benchmark=100,
            adaptativo_benchmark=True
        )
        
        # ESTRATÉGIA 5: Benchmark Manual (Desativado)
        logger.info("\n🔧 ESTRATÉGIA 5: BENCHMARK MANUAL")
        logger.info("Objetivo: Controle total, volume fixo")
        analyzer.configurar_benchmark_inteligente(
            percentil_benchmark=75,
            min_rotas_benchmark=10,
            max_rotas_benchmark=50,
            adaptativo_benchmark=False
        )
        
        logger.info("\n✅ Todas as estratégias foram configuradas!")
        logger.info("Para usar uma estratégia específica, configure o analisador antes da análise")
        
        return True
        
    except Exception as e:
        logger.error(f"Erro na demonstração: {str(e)}")
        return False

def comparar_estrategias_benchmark(analyzer, df_sample):
    """
    Compara diferentes estratégias de benchmark usando dados de exemplo
    
    Args:
        analyzer: Instância do analisador configurada
        df_sample: DataFrame com dados de exemplo
    """
    logger.info("\n" + "=" * 80)
    logger.info("COMPARAÇÃO DE ESTRATÉGIAS DE BENCHMARK")
    logger.info("=" * 80)
    
    estrategias = [
        {"nome": "Conservador (90%)", "percentil": 90, "min_rotas": 5, "max_rotas": 30},
        {"nome": "Equilibrado (75%)", "percentil": 75, "min_rotas": 10, "max_rotas": 50},
        {"nome": "Inclusivo (50%)", "percentil": 50, "min_rotas": 20, "max_rotas": 80},
        {"nome": "Adaptativo (25%)", "percentil": 25, "min_rotas": 30, "max_rotas": 100}
    ]
    
    resultados = []
    
    for estrategia in estrategias:
        logger.info(f"\n🔍 Testando estratégia: {estrategia['nome']}")
        
        # Configurar estratégia
        analyzer.configurar_benchmark_inteligente(
            percentil_benchmark=estrategia['percentil'],
            min_rotas_benchmark=estrategia['min_rotas'],
            max_rotas_benchmark=estrategia['max_rotas'],
            adaptativo_benchmark=True
        )
        
        # Calcular volume mínimo
        volume_minimo = analyzer.calcular_volume_minimo_inteligente(df_sample)
        
        # Analisar distribuição
        analise = analyzer.analisar_distribuicao_volumes(df_sample)
        
        if "erro" not in analise:
            rotas_qualificadas = analise["faixas_volume"]["1000+"] if "1000+" in analise["faixas_volume"] else 0
            
            resultado = {
                "estrategia": estrategia['nome'],
                "percentil": estrategia['percentil'],
                "volume_minimo": volume_minimo,
                "rotas_qualificadas": rotas_qualificadas,
                "total_rotas": analise["total_rotas"],
                "percentual_qualificadas": (rotas_qualificadas / analise["total_rotas"]) * 100 if analise["total_rotas"] > 0 else 0
            }
            
            resultados.append(resultado)
            
            logger.info(f"  - Volume mínimo: {volume_minimo:,.1f} ton")
            logger.info(f"  - Rotas qualificadas: {rotas_qualificadas}")
            logger.info(f"  - Percentual: {resultado['percentual_qualificadas']:.1f}%")
        else:
            logger.warning(f"  - Erro na análise: {analise['erro']}")
    
    # Resumo comparativo
    logger.info("\n" + "=" * 80)
    logger.info("RESUMO COMPARATIVO DAS ESTRATÉGIAS")
    logger.info("=" * 80)
    
    for resultado in resultados:
        logger.info(f"{resultado['estrategia']:20} | "
                   f"Vol: {resultado['volume_minimo']:6,.0f} ton | "
                   f"Rotas: {resultado['rotas_qualificadas']:3} | "
                   f"Perc: {resultado['percentual_qualificadas']:5.1f}%")
    
    return resultados

def recomendar_estrategia_otima(resultados):
    """
    Recomenda a estratégia ótima baseada nos resultados
    
    Args:
        resultados: Lista de resultados das estratégias testadas
    """
    logger.info("\n" + "=" * 80)
    logger.info("RECOMENDAÇÃO DE ESTRATÉGIA ÓTIMA")
    logger.info("=" * 80)
    
    if not resultados:
        logger.warning("Nenhum resultado para análise")
        return None
    
    # Critérios de avaliação
    melhor_equilibrio = None
    melhor_score = 0
    
    for resultado in resultados:
        # Score baseado em múltiplos critérios
        score_volume = 1.0 / (1.0 + resultado['volume_minimo'] / 1000)  # Preferir volumes menores
        score_rotas = min(resultado['rotas_qualificadas'] / 50, 1.0)     # Preferir 50+ rotas
        score_percentual = min(resultado['percentual_qualificadas'] / 30, 1.0)  # Preferir 30%+
        
        # Score composto
        score_total = (score_volume * 0.3 + score_rotas * 0.4 + score_percentual * 0.3)
        
        if score_total > melhor_score:
            melhor_score = score_total
            melhor_equilibrio = resultado
    
    if melhor_equilibrio:
        logger.info(f"🏆 ESTRATÉGIA RECOMENDADA: {melhor_equilibrio['estrategia']}")
        logger.info(f"   - Volume mínimo: {melhor_equilibrio['volume_minimo']:,.1f} ton")
        logger.info(f"   - Rotas qualificadas: {melhor_equilibrio['rotas_qualificadas']}")
        logger.info(f"   - Percentual de cobertura: {melhor_equilibrio['percentual_qualificadas']:.1f}%")
        logger.info(f"   - Score de qualidade: {melhor_score:.3f}")
        
        # Configuração recomendada
        if "Conservador" in melhor_equilibrio['estrategia']:
            logger.info("   - Configuração: Máxima qualidade, poucas rotas")
        elif "Equilibrado" in melhor_equilibrio['estrategia']:
            logger.info("   - Configuração: Boa qualidade, quantidade moderada")
        elif "Inclusivo" in melhor_equilibrio['estrategia']:
            logger.info("   - Configuração: Mais rotas, qualidade moderada")
        elif "Adaptativo" in melhor_equilibrio['estrategia']:
            logger.info("   - Configuração: Máxima cobertura, qualidade ajustável")
    
    return melhor_equilibrio

def main():
    """
    Função principal para demonstrar o benchmark inteligente
    """
    try:
        logger.info("🚀 Iniciando demonstração do Sistema de Benchmark Inteligente")
        
        # Demonstrar configurações
        if not demonstrar_benchmark_inteligente():
            return
        
        logger.info("\n✅ Demonstração concluída com sucesso!")
        logger.info("\nPara usar em produção:")
        logger.info("1. Configure o analisador com a estratégia desejada")
        logger.info("2. Execute a análise normalmente")
        logger.info("3. O sistema calculará automaticamente o volume mínimo")
        logger.info("4. Monitore os logs para ver o volume calculado")
        
    except Exception as e:
        logger.error(f"Erro na demonstração: {str(e)}")

if __name__ == "__main__":
    main()
