#!/usr/bin/env python3
"""
Script principal para análise de oportunidades de redução de preços
Baseado no preço médio por TKU por micro-região
"""

import polars as pl
import logging
from pathlib import Path
from utils.adapters.polars_adapter import PolarsAdapter
from utils.ml.opportunity_analyzer import OpportunityAnalyzer

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data_from_excel(excel_path: str) -> pl.DataFrame:
    """
    Carrega dados do arquivo Excel usando Polars e prepara para análise
    """
    logger.info(f"Carregando dados de: {excel_path}")
    
    # Usar PolarsAdapter para ler o arquivo
    adapter = PolarsAdapter()
    df_polars = adapter.read_excel(excel_path)
    
    # Preparar dados com Polars
    df_polars = adapter.prepare_data(df_polars)
    
    # Mapear colunas para o formato esperado pelo OpportunityAnalyzer
    df_polars_mapped = df_polars.select([
        pl.col('rota').alias('centro_origem'),
        pl.col('volume_ton'),
        pl.col('distancia_km'),
        pl.coalesce(pl.col('preco_ton_km'), pl.col('preco_ton_km_calculado')).alias('preco_ton_km'),
        pl.col('frete_brl').alias('custo_sup_tku'),
        pl.col('data_faturamento'),
        pl.col('01.Rota_MesoOrigem_MesoDestino').alias('rota_mesoregiao')
    ])
    
    # Filtrar linhas com volume e distância válidos
    df_polars_mapped = df_polars_mapped.filter(
        (pl.col('volume_ton') > 0) & 
        (pl.col('distancia_km') > 0)
    )
    
    logger.info(f"Dados carregados: {len(df_polars_mapped)} rotas válidas")
    return df_polars_mapped

def main():
    """
    Função principal para executar a análise de oportunidades
    """
    # Caminho para o arquivo de dados
    excel_path = "sample_data.xlsx"
    
    # Verificar se o arquivo existe
    if not Path(excel_path).exists():
        logger.error(f"Arquivo {excel_path} não encontrado!")
        return
    
    try:
        # Carregar dados com Polars
        df_polars = load_data_from_excel(excel_path)
        
        if df_polars.is_empty():
            logger.error("Nenhum dado válido encontrado!")
            return
        
        # Converter para pandas para compatibilidade com OpportunityAnalyzer
        df = df_polars.to_pandas()
        
        # Inicializar analisador de oportunidades
        analyzer = OpportunityAnalyzer(
            threshold_opportunity=0.02,
            min_volume_threshold=100.0
        )
        
        # Executar análise
        logger.info("Executando análise de oportunidades...")
        df_resultado = analyzer.analyze_reduction_opportunities(df)
        
        # Mostrar resumo dos resultados
        logger.info("=" * 80)
        logger.info("RESUMO DA ANÁLISE DE OPORTUNIDADES")
        logger.info("=" * 80)
        
        total_rotas = len(df_resultado)
        rotas_reducao = len(df_resultado[df_resultado['Ação'] == 'Redução'])
        rotas_aumento = len(df_resultado[df_resultado['Ação'] == 'Aumento'])
        rotas_manter = len(df_resultado[df_resultado['Ação'] == 'Manter'])
        rotas_representativas = len(df_resultado[df_resultado['Rota Representativa'] == True])
        
        logger.info(f"Total de rotas analisadas: {total_rotas}")
        logger.info(f"Oportunidades de redução: {rotas_reducao}")
        logger.info(f"Oportunidades de aumento: {rotas_aumento}")
        logger.info(f"Rotas para manter: {rotas_manter}")
        logger.info(f"Rotas representativas selecionadas: {rotas_representativas}")
        
        # Mostrar TOP 5 oportunidades de redução
        logger.info("\n" + "=" * 80)
        logger.info("TOP 5 OPORTUNIDADES DE REDUÇÃO")
        logger.info("=" * 80)
        
        top_reducoes = df_resultado[df_resultado['Ação'] == 'Redução'].head(5)
        
        for idx, row in top_reducoes.iterrows():
            logger.info(
                f"{row['Centro Origem']}: "
                f"Oportunidade: {row['Oport. (BRL/TON/KM)']:.4f} BRL/TON/KM, "
                f"Impacto: {row['Impacto Estratégico']}, "
                f"Preço atual: {row['Custo Sup (TKU)']:.2f} TKU, "
                f"Novo valor sugerido: {row['Novo Valor Sugerido (BRL/TON/KM)']:.4f} BRL/TON/KM"
            )
        
        # Mostrar distribuição por impacto estratégico
        logger.info("\n" + "=" * 80)
        logger.info("DISTRIBUIÇÃO POR IMPACTO ESTRATÉGICO")
        logger.info("=" * 80)
        
        impacto_counts = df_resultado['Impacto Estratégico'].value_counts()
        for impacto, count in impacto_counts.items():
            logger.info(f"{impacto}: {count} rotas")
        
        # Mostrar análise temporal das rotas representativas
        logger.info("\n" + "=" * 80)
        logger.info("ANÁLISE TEMPORAL DAS ROTAS REPRESENTATIVAS")
        logger.info("=" * 80)
        
        rotas_temp = df_resultado[df_resultado['Rota Representativa'] == True]
        analise_counts = rotas_temp['Análise Temporal'].value_counts()
        
        for analise, count in analise_counts.items():
            if analise != "-":
                logger.info(f"{analise}: {count} rotas")
        
        # Salvar resultado em Excel
        output_path = Path("outputs/analise_oportunidades_reducao_integrada.xlsx")
        output_path.parent.mkdir(exist_ok=True)
        
        df_resultado.to_excel(output_path, index=False, sheet_name='Análise Oportunidades')
        
        logger.info(f"\nRelatório salvo em: {output_path}")
        logger.info("Análise concluída com sucesso!")
        
        # Mostrar primeiras linhas do resultado
        logger.info("\n" + "=" * 80)
        logger.info("PRIMEIRAS LINHAS DO RELATÓRIO")
        logger.info("=" * 80)
        
        # Selecionar colunas principais para exibição
        colunas_exibicao = [
            'Centro Origem', 'Volume (TON)', 'Distância (KM)', 'Custo Sup (TKU)',
            '04.01 - Média MicroRegião - Preço SUP (BRL/TON/KM)',
            '04.02 - Média Cluster - Preço SUP (BRL/TON/KM)',
            'Oport. (BRL/TON/KM)', 'Impacto Estratégico', 'Ação', 'Análise Temporal'
        ]
        
        colunas_disponiveis = [col for col in colunas_exibicao if col in df_resultado.columns]
        df_exibicao = df_resultado[colunas_disponiveis].head(10)
        
        logger.info(f"\n{df_exibicao.to_string()}")
        
    except Exception as e:
        logger.error(f"Erro durante a análise: {str(e)}")
        raise

if __name__ == "__main__":
    main()
