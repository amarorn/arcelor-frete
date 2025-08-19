#!/usr/bin/env python3
"""
Script CORRIGIDO para converter sample_data.xlsx para formato Parquet
compatível com Databricks/Spark
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
        # Carregar dados do Excel com parse_dates específico
        df_excel = pd.read_excel('sample_data.xlsx', parse_dates=['00.dt_doc_faturamento', 'Date'])
        
        logger.info(f"✅ Dados carregados com sucesso!")
        logger.info(f"📈 Total de registros: {len(df_excel):,}")
        logger.info(f"🏗️  Total de colunas: {len(df_excel.columns)}")
        
        # Verificar tipos de dados
        logger.info("\n🔍 TIPOS DE DADOS ORIGINAIS:")
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


def fix_data_types_for_databricks(df):
    """Corrige tipos de dados para compatibilidade com Databricks"""
    logger.info("\n🔧 Corrigindo tipos de dados para compatibilidade com Databricks...")
    
    df_fixed = df.copy()
    
    # 1. Corrigir colunas de data
    date_columns = ['00.dt_doc_faturamento', 'Date']
    for col in date_columns:
        if col in df_fixed.columns:
            # Converter para datetime e depois para string ISO format
            df_fixed[col] = pd.to_datetime(df_fixed[col], errors='coerce')
            # Converter para string ISO para evitar problemas de timezone
            df_fixed[col] = df_fixed[col].dt.strftime('%Y-%m-%d')
            logger.info(f"✅ Coluna {col} convertida para string ISO: YYYY-MM-DD")
    
    # 2. Corrigir colunas numéricas
    numeric_columns = [
        '02.01.00 - Volume (ton)',
        '02.01.01 - Frete Geral (BRL)',
        '02.01.02 - DISTANCIA (KM)',
        '02.03.00 - Preço_Frete Geral (BRL) / TON',
        '02.03.02 - Preço_Frete Geral (BRL / TON / KM)'
    ]
    
    for col in numeric_columns:
        if col in df_fixed.columns:
            # Converter para float64 e tratar valores nulos
            df_fixed[col] = pd.to_numeric(df_fixed[col], errors='coerce')
            # Substituir infinitos por NaN
            df_fixed[col] = df_fixed[col].replace([float('inf'), float('-inf')], pd.NA)
            logger.info(f"✅ Coluna {col} convertida para float64")
    
    # 3. Corrigir colunas de texto
    text_columns = [
        '00.nm_centro_origem_unidade',
        'dc_centro_unidade_descricao',
        '00.nm_modal',
        'nm_tipo_rodovia',
        'nm_veiculo',
        '01.Rota_UFOrigem_UFDestino',
        '01.Rota_MesoOrigem_MesoDestino',
        '01.Rota_MicroOrigem_MicroDestino',
        '01.Rota_MuniOrigem_MuniDestino',
        'id_transportadora',
        'id_fatura',
        'id_transporte'
    ]
    
    for col in text_columns:
        if col in df_fixed.columns:
            # Converter para string e tratar valores nulos
            df_fixed[col] = df_fixed[col].astype(str)
            df_fixed[col] = df_fixed[col].replace('nan', '')
            df_fixed[col] = df_fixed[col].replace('None', '')
            logger.info(f"✅ Coluna {col} convertida para string")
    
    return df_fixed


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
        
        # Limpar valores inválidos de TKU
        df_mapped['tku_calculado'] = pd.to_numeric(df_mapped['tku_calculado'], errors='coerce')
        df_mapped['tku_calculado'] = df_mapped['tku_calculado'].replace([float('inf'), float('-inf')], pd.NA)
    else:
        logger.info("🔄 Calculando TKU...")
        # Garantir que as colunas são numéricas antes do cálculo
        df_mapped['volume_ton'] = pd.to_numeric(df_mapped['volume_ton'], errors='coerce')
        df_mapped['distancia_km'] = pd.to_numeric(df_mapped['distancia_km'], errors='coerce')
        df_mapped['frete_brl'] = pd.to_numeric(df_mapped['frete_brl'], errors='coerce')
        
        # Calcular TKU apenas para registros válidos
        mask = (df_mapped['volume_ton'].notna() & 
                df_mapped['distancia_km'].notna() & 
                df_mapped['frete_brl'].notna() &
                (df_mapped['volume_ton'] > 0) & 
                (df_mapped['distancia_km'] > 0))
        
        df_mapped.loc[mask, 'tku_calculado'] = (
            df_mapped.loc[mask, 'frete_brl'] / 
            (df_mapped.loc[mask, 'volume_ton'] * df_mapped.loc[mask, 'distancia_km'])
        )
    
    # Filtrar apenas registros com dados válidos
    initial_count = len(df_mapped)
    df_mapped = df_mapped.dropna(subset=['volume_ton', 'distancia_km', 'frete_brl'])
    final_count = len(df_mapped)
    
    logger.info(f"📊 Registros válidos: {final_count:,}/{initial_count:,} ({final_count/initial_count*100:.1f}%)")
    
    return df_mapped


def convert_to_parquet_databricks_compatible(df, output_filename='sample_data_databricks_compatible.parquet'):
    """Converte DataFrame para formato Parquet compatível com Databricks"""
    logger.info(f"\n💾 Convertendo para Parquet compatível com Databricks: {output_filename}")
    
    try:
        # 1. Converter para Polars com tipos específicos
        df_polars = pl.from_pandas(df)
        
        # 2. Definir schema explícito para compatibilidade
        schema = {
            'data_faturamento': pl.Utf8,  # String para evitar problemas de timestamp
            'centro_origem': pl.Utf8,
            'centro_origem_descricao': pl.Utf8,
            'modal': pl.Utf8,
            'tipo_rodovia': pl.Utf8,
            'tipo_veiculo': pl.Utf8,
            'rota_uf': pl.Utf8,
            'rota_mesoregiao': pl.Utf8,
            'rota_microregiao': pl.Utf8,
            'rota_municipio': pl.Utf8,
            'id_transportadora': pl.Utf8,
            'id_fatura': pl.Utf8,
            'id_transporte': pl.Utf8,
            'volume_ton': pl.Float64,
            'frete_brl': pl.Float64,
            'distancia_km': pl.Float64,
            'preco_por_ton': pl.Float64,
            'tku_calculado': pl.Float64,
            'microregiao_origem': pl.Utf8
        }
        
        # 3. Aplicar schema
        df_polars = df_polars.cast(schema)
        
        # 4. Salvar como Parquet com configurações específicas
        df_polars.write_parquet(
            output_filename,
            compression='snappy',  # Compressão padrão do Spark
            use_pyarrow=True       # Usar PyArrow para melhor compatibilidade
        )
        
        # 5. Verificar arquivo criado
        file_size = Path(output_filename).stat().st_size / (1024 * 1024)  # MB
        logger.info(f"✅ Arquivo Parquet compatível com Databricks criado!")
        logger.info(f"📁 Nome: {output_filename}")
        logger.info(f"💾 Tamanho: {file_size:.2f} MB")
        logger.info(f"📊 Registros: {len(df_polars):,}")
        logger.info(f"🏗️  Colunas: {len(df_polars.columns)}")
        
        # 6. Verificar schema do arquivo criado
        logger.info("\n🔍 SCHEMA DO ARQUIVO PARQUET:")
        df_check = pl.read_parquet(output_filename)
        for col, dtype in df_check.schema.items():
            logger.info(f"  {col}: {dtype}")
        
        return output_filename
        
    except Exception as e:
        logger.error(f"❌ Erro ao converter para Parquet: {str(e)}")
        return None


def create_databricks_compatible_metadata(df, output_filename):
    """Cria metadados específicos para Databricks"""
    logger.info("\n📚 Criando metadados para Databricks...")
    
    try:
        # Informações específicas para Databricks
        metadata = {
            "databricks_info": {
                "compatibility": "Databricks/Spark Compatible",
                "parquet_version": "2.6",
                "compression": "snappy",
                "created_with": "Polars + PyArrow",
                "notes": "Timestamps convertidos para strings ISO para evitar conflitos de tipo"
            },
            "dataset_info": {
                "nome": "Dados de Frete ArcelorMittal - Compatível com Databricks",
                "descricao": "Dados históricos de fretes com análise de tendências TKU",
                "fonte": "sample_data.xlsx",
                "formato_original": "Excel (.xlsx)",
                "formato_atual": "Parquet (.parquet)",
                "data_criacao": datetime.now().isoformat(),
                "versao": "2.0 - Databricks Compatible"
            },
            "estatisticas": {
                "total_registros": len(df),
                "total_colunas": len(df.columns),
                "periodo_dados": {
                    "inicio": df['data_faturamento'].min() if 'data_faturamento' in df.columns else "N/A",
                    "fim": df['data_faturamento'].max() if 'data_faturamento' in df.columns else "N/A"
                }
            },
            "colunas": {}
        }
        
        # Detalhes das colunas com tipos específicos
        for col in df.columns:
            metadata["colunas"][col] = {
                "tipo_pandas": str(df[col].dtype),
                "tipo_parquet": "Utf8" if df[col].dtype == 'object' else "Float64",
                "valores_unicos": df[col].nunique(),
                "valores_nulos": df[col].isna().sum(),
                "percentual_nulos": (df[col].isna().sum() / len(df)) * 100
            }
        
        # Salvar metadados
        metadata_filename = output_filename.replace('.parquet', '_metadata.json')
        import json
        with open(metadata_filename, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"✅ Metadados para Databricks salvos em: {metadata_filename}")
        
        return metadata_filename
        
    except Exception as e:
        logger.error(f"❌ Erro ao criar metadados: {str(e)}")
        return None


def test_databricks_compatibility(output_filename):
    """Testa compatibilidade com Databricks"""
    logger.info("\n🧪 Testando compatibilidade com Databricks...")
    
    try:
        # Ler o arquivo Parquet criado
        df_test = pl.read_parquet(output_filename)
        
        # Verificar se consegue fazer operações básicas
        logger.info("✅ Leitura do Parquet: OK")
        
        # Verificar tipos de dados
        logger.info("✅ Schema do Parquet:")
        for col, dtype in df_test.schema.items():
            logger.info(f"  {col}: {dtype}")
        
        # Verificar se não há tipos problemáticos
        problematic_types = ['Timestamp', 'Int64']
        has_problematic = any(str(dtype) in str(dtype) for dtype in df_test.schema.values())
        
        if not has_problematic:
            logger.info("✅ Nenhum tipo problemático encontrado")
            logger.info("✅ Arquivo compatível com Databricks/Spark")
            return True
        else:
            logger.warning("⚠️  Tipos problemáticos encontrados")
            return False
            
    except Exception as e:
        logger.error(f"❌ Erro no teste de compatibilidade: {str(e)}")
        return False


def main():
    """Função principal"""
    logger.info("🚀 CONVERSÃO EXCEL → PARQUET COMPATÍVEL COM DATABRICKS")
    logger.info("=" * 70)
    
    # 1. Carregar dados do Excel
    df_excel = load_and_analyze_excel_data()
    if df_excel is None:
        return
    
    # 2. Corrigir tipos de dados para compatibilidade
    df_fixed = fix_data_types_for_databricks(df_excel)
    
    # 3. Mapear colunas para o TKUTrendAnalyzer
    df_mapped = map_columns_for_tku_analyzer(df_fixed)
    
    # 4. Converter para Parquet compatível com Databricks
    output_filename = convert_to_parquet_databricks_compatible(df_mapped)
    if output_filename is None:
        return
    
    # 5. Criar metadados específicos para Databricks
    metadata_filename = create_databricks_compatible_metadata(df_mapped, output_filename)
    
    # 6. Testar compatibilidade
    compatibility_test = test_databricks_compatibility(output_filename)
    
    # 7. Resumo final
    logger.info("\n" + "=" * 70)
    if compatibility_test:
        logger.info("🎉 CONVERSÃO COMPATÍVEL COM DATABRICKS CONCLUÍDA!")
    else:
        logger.info("⚠️  CONVERSÃO CONCLUÍDA COM AVISOS")
    logger.info("=" * 70)
    
    logger.info(f"📁 Arquivo Parquet: {output_filename}")
    if metadata_filename:
        logger.info(f"📚 Metadados: {metadata_filename}")
    logger.info(f"📊 Total de registros: {len(df_mapped):,}")
    logger.info(f"🏗️  Total de colunas: {len(df_mapped.columns)}")
    logger.info(f"✅ Compatibilidade Databricks: {'Sim' if compatibility_test else 'Verificar'}")
    
    logger.info("\n💡 PRÓXIMOS PASSOS:")
    logger.info("1. Faça upload do arquivo .parquet para seu catálogo Databricks")
    logger.info("2. Use os metadados para documentação")
    logger.info("3. Teste com uma query simples no Databricks")
    logger.info("4. Use o TKUTrendAnalyzer para análises")


if __name__ == "__main__":
    main()
