#!/usr/bin/env python3
"""
Script para converter sample_data.xlsx para formato Parquet
e mapear colunas para compatibilidade com o TKUTrendAnalyzer
"""

import pandas as pd
import polars as pl
import logging
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_and_analyze_excel_data():
    """Carrega e analisa dados do Excel"""
    logger.info("📊 Carregando dados do sample_data.xlsx...")
    
    try:
        # Carregar dados do Excel
        df_excel = pd.read_excel('sample_data.xlsx')
        
        logger.info(f"✅ Dados carregados com sucesso!")
        logger.info(f"📈 Total de registros: {len(df_excel):,}")
        logger.info(f"🏗️  Total de colunas: {len(df_excel.columns)}")
        
        # Mostrar informações das colunas
        logger.info("\n📋 COLUNAS DISPONÍVEIS:")
        for i, col in enumerate(df_excel.columns):
            logger.info(f"  {i+1:2d}. {col}")
        
        # Verificar tipos de dados
        logger.info("\n🔍 TIPOS DE DADOS:")
        for col in df_excel.columns:
            dtype = df_excel[col].dtype
            non_null = df_excel[col].count()
            total = len(df_excel)
            pct = (non_null / total) * 100
            logger.info(f"  {col}: {dtype} ({non_null:,}/{total:,} = {pct:.1f}%)")
        
        return df_excel
        
    except Exception as e:
        logger.error(f"❌ Erro ao carregar dados: {str(e)}")
        return None


def map_columns_for_tku_analyzer(df):
    """Mapeia colunas do Excel para o formato esperado pelo TKUTrendAnalyzer"""
    logger.info("\n🔄 Mapeando colunas para o TKUTrendAnalyzer...")
    
    # Mapeamento de colunas
    column_mapping = {
        # Colunas principais
        '00.dt_doc_faturamento': 'data_faturamento',
        '00.nm_centro_origem_unidade': 'centro_origem',
        'dc_centro_unidade_descricao': 'centro_origem_descricao',
        '00.nm_modal': 'modal',
        'nm_tipo_rodovia': 'tipo_rodovia',
        'nm_veiculo': 'tipo_veiculo',
        
        # Rotas
        '01.Rota_UFOrigem_UFDestino': 'rota_uf',
        '01.Rota_MesoOrigem_MesoDestino': 'rota_mesoregiao',
        '01.Rota_MicroOrigem_MicroDestino': 'rota_microregiao',
        '01.Rota_MuniOrigem_MuniDestino': 'rota_municipio',
        
        # Identificadores
        'id_transportadora': 'id_transportadora',
        'id_fatura': 'id_fatura',
        'id_transporte': 'id_transporte',
        
        # Métricas principais
        '02.01.00 - Volume (ton)': 'volume_ton',
        '02.01.01 - Frete Geral (BRL)': 'frete_brl',
        '02.01.02 - DISTANCIA (KM)': 'distancia_km',
        '02.03.00 - Preço_Frete Geral (BRL) / TON': 'preco_por_ton',
        '02.03.02 - Preço_Frete Geral (BRL / TON / KM)': 'tku_calculado'
    }
    
    # Criar novo DataFrame com colunas mapeadas
    df_mapped = df.copy()
    
    # Renomear colunas
    df_mapped = df_mapped.rename(columns=column_mapping)
    
    # Adicionar colunas derivadas
    df_mapped['microregiao_origem'] = df_mapped['centro_origem']
    
    # Verificar se TKU já está calculado, senão calcular
    if 'tku_calculado' in df_mapped.columns:
        # Verificar se há valores válidos
        tku_valid = df_mapped['tku_calculado'].notna().sum()
        logger.info(f"✅ TKU já calculado: {tku_valid:,} valores válidos")
    else:
        logger.info("🔄 Calculando TKU...")
        df_mapped['tku_calculado'] = df_mapped['frete_brl'] / (df_mapped['volume_ton'] * df_mapped['distancia_km'])
    
    # Filtrar apenas registros com dados válidos
    initial_count = len(df_mapped)
    df_mapped = df_mapped.dropna(subset=['volume_ton', 'distancia_km', 'frete_brl'])
    final_count = len(df_mapped)
    
    logger.info(f"📊 Registros válidos: {final_count:,}/{initial_count:,} ({final_count/initial_count*100:.1f}%)")
    
    return df_mapped


def convert_to_parquet(df, output_filename='sample_data_parquet.parquet'):
    """Converte DataFrame para formato Parquet"""
    logger.info(f"\n💾 Convertendo para Parquet: {output_filename}")
    
    try:
        # Converter para Polars (mais eficiente para Parquet)
        df_polars = pl.from_pandas(df)
        
        # Salvar como Parquet
        df_polars.write_parquet(output_filename)
        
        # Verificar arquivo criado
        file_size = Path(output_filename).stat().st_size / (1024 * 1024)  # MB
        logger.info(f"✅ Arquivo Parquet criado com sucesso!")
        logger.info(f"📁 Nome: {output_filename}")
        logger.info(f"💾 Tamanho: {file_size:.2f} MB")
        logger.info(f"📊 Registros: {len(df_polars):,}")
        logger.info(f"🏗️  Colunas: {len(df_polars.columns)}")
        
        return output_filename
        
    except Exception as e:
        logger.error(f"❌ Erro ao converter para Parquet: {str(e)}")
        return None


def create_sample_analysis(df):
    """Cria análise de exemplo usando o TKUTrendAnalyzer"""
    logger.info("\n🧠 Testando TKUTrendAnalyzer com dados reais...")
    
    try:
        from utils.ml.tku_trend_analyzer import TKUTrendAnalyzer
        
        # Converter para Polars
        df_polars = pl.from_pandas(df)
        
        # Inicializar analisador
        analyzer = TKUTrendAnalyzer()
        
        # Identificar rotas únicas
        rotas_unicas = df_polars['rota_municipio'].unique().to_list()
        logger.info(f"🛣️  Rotas encontradas: {len(rotas_unicas)}")
        logger.info(f"📍 Exemplos: {rotas_unicas[:5]}")
        
        # Analisar uma rota específica
        if rotas_unicas:
            rota_teste = rotas_unicas[0]
            logger.info(f"\n📊 Analisando rota de teste: {rota_teste}")
            
            analysis = analyzer.analyze_route_tku_trends(df_polars, rota_teste)
            
            if 'erro' not in analysis:
                logger.info(f"✅ Análise concluída para {rota_teste}")
                logger.info(f"🎯 Status: {analysis['status_geral']}")
                logger.info(f"💰 TKU Atual: R$ {analysis['tku_atual']:.4f}")
                logger.info(f"📊 Variação: {analysis['variacao_percentual']:+.1f}%")
                logger.info(f"📈 Tendência 3M: {analysis['tendencias']['3_meses']['tendencia']}")
                
                # Verificar recomendações
                if analysis['recomendacoes']:
                    rec = analysis['recomendacoes'][0]
                    logger.info(f"💡 Recomendação: {rec['acao']}")
                    logger.info(f"   • Tipo: {rec['tipo']}")
                    logger.info(f"   • Prioridade: {rec['prioridade']}")
            else:
                logger.warning(f"⚠️  Erro na análise: {analysis['erro']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erro ao testar TKUTrendAnalyzer: {str(e)}")
        return False


def create_data_catalog_metadata(df, output_filename):
    """Cria metadados para o catálogo de dados"""
    logger.info("\n📚 Criando metadados para o catálogo...")
    
    try:
        # Informações básicas
        metadata = {
            "dataset_info": {
                "nome": "Dados de Frete ArcelorMittal",
                "descricao": "Dados históricos de fretes com análise de tendências TKU",
                "fonte": "sample_data.xlsx",
                "formato_original": "Excel (.xlsx)",
                "formato_atual": "Parquet (.parquet)",
                "data_criacao": datetime.now().isoformat(),
                "versao": "1.0"
            },
            "estatisticas": {
                "total_registros": len(df),
                "total_colunas": len(df.columns),
                "periodo_dados": {
                    "inicio": df['data_faturamento'].min().isoformat() if 'data_faturamento' in df.columns else "N/A",
                    "fim": df['data_faturamento'].max().isoformat() if 'data_faturamento' in df.columns else "N/A"
                }
            },
            "colunas": {}
        }
        
        # Detalhes das colunas
        for col in df.columns:
            metadata["colunas"][col] = {
                "tipo": str(df[col].dtype),
                "valores_unicos": df[col].nunique(),
                "valores_nulos": df[col].isna().sum(),
                "percentual_nulos": (df[col].isna().sum() / len(df)) * 100
            }
        
        # Salvar metadados
        metadata_filename = output_filename.replace('.parquet', '_metadata.json')
        import json
        with open(metadata_filename, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"✅ Metadados salvos em: {metadata_filename}")
        
        return metadata_filename
        
    except Exception as e:
        logger.error(f"❌ Erro ao criar metadados: {str(e)}")
        return None


def main():
    """Função principal"""
    logger.info("🚀 INICIANDO CONVERSÃO EXCEL → PARQUET")
    logger.info("=" * 60)
    
    # 1. Carregar dados do Excel
    df_excel = load_and_analyze_excel_data()
    if df_excel is None:
        return
    
    # 2. Mapear colunas para o TKUTrendAnalyzer
    df_mapped = map_columns_for_tku_analyzer(df_excel)
    
    # 3. Converter para Parquet
    output_filename = convert_to_parquet(df_mapped)
    if output_filename is None:
        return
    
    # 4. Criar metadados para o catálogo
    metadata_filename = create_data_catalog_metadata(df_mapped, output_filename)
    
    # 5. Testar com o TKUTrendAnalyzer
    success = create_sample_analysis(df_mapped)
    
    # 6. Resumo final
    logger.info("\n" + "=" * 60)
    logger.info("🎉 CONVERSÃO CONCLUÍDA COM SUCESSO!")
    logger.info("=" * 60)
    logger.info(f"📁 Arquivo Parquet: {output_filename}")
    if metadata_filename:
        logger.info(f"📚 Metadados: {metadata_filename}")
    logger.info(f"📊 Total de registros: {len(df_mapped):,}")
    logger.info(f"🏗️  Total de colunas: {len(df_mapped.columns)}")
    logger.info(f"✅ TKUTrendAnalyzer testado: {'Sim' if success else 'Não'}")
    
    logger.info("\n💡 PRÓXIMOS PASSOS:")
    logger.info("1. Faça upload do arquivo .parquet para seu catálogo")
    logger.info("2. Use o TKUTrendAnalyzer para análises em tempo real")
    logger.info("3. Configure alertas automáticos para rotas críticas")
    logger.info("4. Gere relatórios semanais/mensais de tendências")


if __name__ == "__main__":
    main()
