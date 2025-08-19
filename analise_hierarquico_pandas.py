#!/usr/bin/env python3
"""
Script para análise de agrupamento hierárquico usando pandas
Implementa estrutura exata: Centro Origem > Sub-categoria > Rota Específica
"""

import logging
import os
import time
from pathlib import Path
import pandas as pd
import numpy as np

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HierarchicalPandasAnalyzer:
    """
    Analisador de agrupamento hierárquico usando pandas
    """
    
    def __init__(self):
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
        Carrega dados do Excel usando pandas
        """
        logger.info(f"Carregando dados de: {excel_path}")
        
        df = pd.read_excel(excel_path)
        logger.info(f"Dados carregados: {len(df)} linhas")
        
        # Renomear colunas para facilitar o uso
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
        
        df = df.rename(columns=column_mapping)
        return df
    
    def prepare_data_hierarchical(self, df):
        """
        Prepara dados para agrupamento hierárquico
        """
        logger.info("Preparando dados para agrupamento hierárquico...")
        
        # Filtrar dados válidos
        df_processed = df[
            (df['volume_ton'] > 0) & 
            (df['distancia_km'] > 0)
        ].copy()
        
        # Calcular preço por TKU
        df_processed['preco_ton_km'] = df_processed['frete_brl'] / (df_processed['volume_ton'] * df_processed['distancia_km'])
        
        # Extrair micro-região
        def extract_microregion(location):
            if pd.isna(location) or str(location).strip() == '':
                return "UNKNOWN"
            location_str = str(location).upper().strip()
            for key, microregion in self.microregion_mapping:
                if key in location_str:
                    return microregion
            return location_str
        
        df_processed['microregiao_origem'] = df_processed['centro_origem'].apply(extract_microregion)
        
        # Criar estrutura hierárquica
        df_processed['nivel_1_centro_origem'] = df_processed['centro_origem'].apply(
            lambda x: 'Usina' if 'USINA' in str(x).upper() else 
                     'ITABIRA' if 'ITABIRA' in str(x).upper() else str(x)
        )
        
        # Criar sub-categoria (nível 2)
        df_processed['nivel_2_subcategoria'] = df_processed['rota_microregiao'].fillna(
            df_processed['rota_municipio']
        )
        
        # Criar rota específica (nível 3)
        df_processed['nivel_3_rota_especifica'] = df_processed['rota_municipio'].fillna(
            df_processed['rota_microregiao']
        )
        
        logger.info(f"Dados preparados: {len(df_processed)} linhas válidas")
        return df_processed
    
    def create_hierarchical_structure(self, df):
        """
        Cria estrutura hierárquica similar ao dashboard da imagem
        """
        logger.info("Criando estrutura hierárquica...")
        
        # 1. Agrupar por nível 3 (Rota Específica) - nível mais detalhado
        df_nivel3 = df.groupby(['nivel_1_centro_origem', 'nivel_2_subcategoria', 'nivel_3_rota_especifica']).agg({
            'volume_ton': 'sum',
            'frete_brl': 'sum',
            'distancia_km': 'mean',
            'preco_ton_km': 'mean'
        }).reset_index()
        
        # 2. Calcular preços médios por micro-região para comparação
        # Usar o mapeamento correto para encontrar a micro-região correspondente
        df_microregiao_prices = df.groupby('microregiao_origem')['preco_ton_km'].mean().reset_index()
        df_microregiao_prices.columns = ['microregiao_origem', 'preco_medio_microregiao']
        
        # 3. Mesclar com preços médios usando a micro-região correta
        # Primeiro, vamos adicionar a micro-região de origem ao df_nivel3
        df_nivel3_with_micro = df_nivel3.merge(
            df[['nivel_1_centro_origem', 'nivel_2_subcategoria', 'nivel_3_rota_especifica', 'microregiao_origem']].drop_duplicates(),
            on=['nivel_1_centro_origem', 'nivel_2_subcategoria', 'nivel_3_rota_especifica'],
            how='left'
        )
        
        # Agora fazer o merge com os preços médios
        df_hierarchical = df_nivel3_with_micro.merge(
            df_microregiao_prices,
            on='microregiao_origem',
            how='left'
        )
        
        # 4. Calcular métricas finais
        df_hierarchical['frete_geral_brl_ton'] = (
            df_hierarchical['frete_brl'] / df_hierarchical['volume_ton']
        )
        
        df_hierarchical['frete_geral_brl_ton_km'] = (
            df_hierarchical['frete_brl'] / (df_hierarchical['volume_ton'] * df_hierarchical['distancia_km'])
        )
        
        # 5. Calcular oportunidade
        df_hierarchical['oportunidade_brl_ton_km'] = (
            df_hierarchical['frete_geral_brl_ton_km'] - df_hierarchical['preco_medio_microregiao']
        )
        
        # Debug: verificar se os valores estão sendo calculados
        logger.info(f"Debug - Valores de oportunidade:")
        logger.info(f"  - frete_geral_brl_ton_km: {df_hierarchical['frete_geral_brl_ton_km'].describe()}")
        logger.info(f"  - preco_medio_microregiao: {df_hierarchical['preco_medio_microregiao'].describe()}")
        logger.info(f"  - oportunidade_brl_ton_km: {df_hierarchical['oportunidade_brl_ton_km'].describe()}")
        logger.info(f"  - Valores NaN em oportunidade: {df_hierarchical['oportunidade_brl_ton_km'].isna().sum()}")
        
        # 6. Determinar ação
        df_hierarchical['acao'] = df_hierarchical['oportunidade_brl_ton_km'].apply(
            lambda x: 'Redução' if x > 0.01 else 'Aumento' if x < -0.01 else 'Manter'
        )
        
        # 7. Ordenar por hierarquia
        df_hierarchical = df_hierarchical.sort_values([
            'nivel_1_centro_origem',
            'nivel_2_subcategoria',
            'nivel_3_rota_especifica'
        ])
        
        logger.info(f"Estrutura hierárquica criada: {len(df_hierarchical)} linhas")
        return df_hierarchical
    
    def generate_hierarchical_report(self, df_hierarchical):
        """
        Gera relatório hierárquico similar ao dashboard da imagem
        """
        logger.info("Gerando relatório hierárquico...")
        
        # Selecionar colunas para o relatório final
        df_report = df_hierarchical[[
            'nivel_1_centro_origem',
            'nivel_2_subcategoria', 
            'nivel_3_rota_especifica',
            'volume_ton',
            'frete_geral_brl_ton',
            'frete_geral_brl_ton_km',
            'oportunidade_brl_ton_km',
            'acao'
        ]].copy()
        
        # Renomear colunas para o formato final
        df_report.columns = [
            'Centro Origem',
            'Sub-Categoria', 
            'Rota Específica',
            'Volume (TON)',
            'Fr Geral (BRL / TON)',
            'Fr Geral (BRL / TON / KM)',
            'Oport. (BRL/TON/KM)',
            'Ação'
        ]
        
        # Salvar relatório
        output_path = "outputs/dashboard_hierarquico_pandas.xlsx"
        df_report.to_excel(output_path, index=False)
        
        # Retornar resumo
        total_rotas = len(df_report)
        rotas_reducao = len(df_report[df_report['Ação'] == 'Redução'])
        rotas_aumento = len(df_report[df_report['Ação'] == 'Aumento'])
        rotas_manter = len(df_report[df_report['Ação'] == 'Manter'])
        
        # Top oportunidades de redução
        top_reducoes = df_report[df_report['Ação'] == 'Redução'].nlargest(5, 'Oport. (BRL/TON/KM)')
        
        # Top rotas por volume
        top_volume = df_report.nlargest(5, 'Volume (TON)')
        
        return {
            'total_rotas': total_rotas,
            'rotas_reducao': rotas_reducao,
            'rotas_aumento': rotas_aumento,
            'rotas_manter': rotas_manter,
            'top_reducoes': top_reducoes.to_dict('records'),
            'top_volume': top_volume.to_dict('records')
        }

def main():
    """
    Função principal para executar análise hierárquica com pandas
    """
    try:
        # Inicializar analisador
        analyzer = HierarchicalPandasAnalyzer()
        
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
        logger.info("RELATÓRIO DA ANÁLISE HIERÁRQUICA COM PANDAS")
        logger.info("=" * 80)
        logger.info(f"Tempo de execução: {execution_time:.2f} segundos")
        logger.info(f"Total de rotas analisadas: {report['total_rotas']}")
        logger.info(f"Oportunidades de redução: {report['rotas_reducao']}")
        logger.info(f"Oportunidades de aumento: {report['rotas_aumento']}")
        logger.info(f"Rotas para manter: {report['rotas_manter']}")
        
        logger.info("\nTOP 5 OPORTUNIDADES DE REDUÇÃO:")
        for i, op in enumerate(report['top_reducoes'][:5], 1):
            logger.info(f"{i}. {op['Rota Específica']}: {op['Oport. (BRL/TON/KM)']:.4f} BRL/TON/KM")
        
        logger.info("\nTOP 5 ROTAS POR VOLUME:")
        for i, rota in enumerate(report['top_volume'][:5], 1):
            logger.info(f"{i}. {rota['Rota Específica']}: {rota['Volume (TON)']:,.1f} ton")
        
        logger.info(f"\nRelatório salvo em: outputs/dashboard_hierarquico_pandas.xlsx")
        logger.info("Análise hierárquica com pandas concluída com sucesso!")
        
    except Exception as e:
        logger.error(f"Erro durante a análise: {str(e)}")
        raise

if __name__ == "__main__":
    main()
