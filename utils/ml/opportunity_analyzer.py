"""
Analisador de Oportunidades de Redução de Preços
Baseado no preço médio por TKU por micro-região
"""

import pandas as pd
import numpy as np
import logging
from sklearn.linear_model import LinearRegression
from typing import Protocol

logger = logging.getLogger(__name__)

class IMicroRegionPriceCalculator(Protocol):
    def calculate_average_price_by_microregion(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula o preço médio por TKU por micro-região
        """
        ...

class IOpportunityAnalyzer(Protocol):
    def analyze_reduction_opportunities(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analisa oportunidades de redução de preços baseado no preço médio por TKU por micro-região
        """
        ...

class MicroRegionPriceCalculator(IMicroRegionPriceCalculator):
    """
    Calcula preços médios por micro-região e meso-região
    """
    
    def __init__(self, 
                 origin_col: str = 'centro_origem',
                 volume_col: str = 'volume_ton',
                 distance_col: str = 'distancia_km',
                 cost_col: str = 'custo_sup_tku',
                 price_col: str = 'preco_ton_km'):
        
        self.origin_col = origin_col
        self.volume_col = volume_col
        self.distance_col = distance_col
        self.cost_col = cost_col
        self.price_col = price_col
        
        # Mapeamento de micro-regiões (ordem específica para priorizar matches)
        self.microregion_mapping = [
            ('JOÃO MONLEVADE', 'JOÃO MONLEVADE'),
            ('USINA MONLEVADE', 'JOÃO MONLEVADE'),
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
            ('MARIANA', 'MARIANA'),
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
    
    def _extract_microregion(self, location: str) -> str:
        """
        Extrai a micro-região de uma string de localização
        """
        if pd.isna(location):
            return "UNKNOWN"
        
        location_str = str(location).upper()
        
        for key, microregion in self.microregion_mapping:
            if key in location_str:
                return microregion
        
        return location_str
    
    def _classificar_faixa_distancia(self, distancia: float) -> str:
        """
        Classifica a distância em faixas para clustering
        """
        if distancia > 2000:
            return "> 2000"
        if 1501 <= distancia <= 2000:
            return "1501 a 2000"
        if 1001 <= distancia <= 1500:
            return "1001 a 1500"
        if 751 <= distancia <= 1000:
            return "751 a 1000"
        if 501 <= distancia <= 750:
            return "501 a 750"
        if 401 <= distancia <= 500:
            return "401 a 500"
        if 301 <= distancia <= 400:
            return "301 a 400"
        if 201 <= distancia <= 300:
            return "201 a 300"
        if 151 <= distancia <= 200:
            return "151 a 200"
        if 101 <= distancia <= 150:
            return "101 a 150"
        return "<= 100"
    
    def _analisar_tendencia_temporal(self, df_rota: pd.DataFrame, rota: str) -> str:
        """
        Analisa tendência temporal de preços para uma rota específica
        """
        if len(df_rota) < 2:
            return "DADOS INSUFICIENTES"
        
        # Ordenar por data
        df_rota = df_rota.sort_values('data_faturamento')
        
        # Calcular índice (preço por TKU) se não existir
        if 'preco_ton_km' not in df_rota.columns:
            df_rota['preco_ton_km'] = (
                df_rota['custo_sup_tku'] / 
                (df_rota['volume_ton'] * df_rota['distancia_km'])
            )
        
        # Filtrar valores válidos
        df_rota = df_rota[df_rota['preco_ton_km'].notna()]
        if len(df_rota) < 2:
            return "DADOS INSUFICIENTES"
        
        # Calcular estatísticas
        media = df_rota['preco_ton_km'].mean()
        ultimo = df_rota['preco_ton_km'].iloc[-1]
        minimo = df_rota['preco_ton_km'].min()
        
        # Calcular tendência usando regressão linear
        X = np.arange(len(df_rota)).reshape(-1, 1)
        y = df_rota['preco_ton_km'].values
        modelo = LinearRegression().fit(X, y)
        tendencia = float(modelo.coef_[0])
        
        # Calcular variações
        delta_media = (ultimo - media) / media if media > 0 else 0
        delta_min = (ultimo - minimo) / minimo if minimo > 0 else 0
        
        # Classificar tendência
        if tendencia < -0.001:  # Tendência de redução
            return "EM REDUÇÃO (MONITORAR)"
        elif delta_media > 0.2 and delta_min > 0.3:  # Preço alto
            return "ALTO (SUGERIR AÇÃO)"
        else:
            return "NEUTRO (ANÁLISE NECESSÁRIA)"
    
    def calculate_average_price_by_microregion(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula o preço médio por TKU por micro-região e meso-região
        """
        logger.info("Calculando preço médio por micro-região e meso-região...")
        
        # Criar cópia para não modificar o original
        df_analysis = df.copy()
        
        # Extrair micro-região de origem
        df_analysis['microregiao_origem'] = df_analysis[self.origin_col].apply(
            self._extract_microregion
        )
        
        # Extrair meso-região (assumindo que está na coluna rota_mesoregiao)
        if 'rota_mesoregiao' in df_analysis.columns:
            df_analysis['mesoregiao'] = df_analysis['rota_mesoregiao']
        else:
            # Fallback: usar micro-região como meso-região
            df_analysis['mesoregiao'] = df_analysis['microregiao_origem']
        
        # Classificar faixa de distância
        df_analysis['faixa_distancia'] = df_analysis[self.distance_col].apply(
            self._classificar_faixa_distancia
        )
        
        # Calcular preço por TKU (se não existir)
        if self.price_col not in df_analysis.columns:
            df_analysis[self.price_col] = (
                df_analysis[self.cost_col] / 
                (df_analysis[self.volume_col] * df_analysis[self.distance_col])
            )
        
        # Calcular preço médio por micro-região
        microregion_avg = df_analysis.groupby('microregiao_origem')[self.price_col].agg([
            'mean', 'std', 'count'
        ]).reset_index()
        
        microregion_avg.columns = [
            'microregiao_origem', 
            'preco_medio_tku', 
            'desvio_padrao_tku', 
            'quantidade_rotas'
        ]
        
        # Calcular preço médio por cluster (meso-região + faixa de distância)
        df_analysis['cluster_id'] = df_analysis['mesoregiao'] + ' | ' + df_analysis['faixa_distancia']
        
        cluster_avg = df_analysis.groupby('cluster_id')[self.price_col].agg([
            'mean', 'std', 'count'
        ]).reset_index()
        
        cluster_avg.columns = [
            'cluster_id', 
            'preco_medio_cluster', 
            'desvio_padrao_cluster', 
            'quantidade_rotas_cluster'
        ]
        
        # Mesclar as duas análises
        result = microregion_avg.merge(
            df_analysis[['microregiao_origem', 'cluster_id']].drop_duplicates(),
            on='microregiao_origem',
            how='left'
        ).merge(
            cluster_avg,
            on='cluster_id',
            how='left'
        )
        
        logger.info(f"Preços médios calculados para {len(result)} micro-regiões")
        return result


class OpportunityAnalyzer(IOpportunityAnalyzer):
    """
    Analisa oportunidades de redução de preços com funcionalidades integradas
    """
    
    def __init__(self, 
                 threshold_opportunity: float = 0.02,
                 min_volume_threshold: float = 100.0):
        
        self.threshold_opportunity = threshold_opportunity
        self.min_volume_threshold = min_volume_threshold
        self.price_calculator = MicroRegionPriceCalculator()
    
    def _calculate_opportunity(self, 
                             current_price: float, 
                             avg_microregion_price: float,
                             avg_cluster_price: float,
                             volume: float,
                             distance: float) -> tuple:
        """
        Calcula a oportunidade de redução em BRL/TON/KM e classifica o impacto
        """
        if pd.isna(current_price) or pd.isna(volume) or pd.isna(distance):
            return 0.0, "NÃO APLICÁVEL"
        
        if volume < self.min_volume_threshold or distance <= 0:
            return 0.0, "NÃO APLICÁVEL"
        
        # Usar preço do cluster se disponível, senão usar micro-região
        reference_price = avg_cluster_price if pd.notna(avg_cluster_price) else avg_microregion_price
        
        if pd.isna(reference_price):
            return 0.0, "NÃO APLICÁVEL"
        
        # Oportunidade = preço atual - preço de referência
        opportunity = current_price - reference_price
        
        # Classificar impacto baseado no volume e na diferença
        if opportunity <= 0:
            return opportunity, "OK"
        
        # Calcular percentual de diferença
        percent_diff = (opportunity / reference_price) * 100
        
        # Classificar impacto
        if percent_diff <= 10:
            impacto = "BAIXO"
        elif percent_diff <= 25:
            impacto = "MÉDIO"
        else:
            impacto = "ALTO"
        
        # Ajustar impacto baseado no volume
        if volume > 1000:  # Volume alto
            if impacto == "BAIXO":
                impacto = "MÉDIO"
            elif impacto == "MÉDIO":
                impacto = "ALTO"
        
        return opportunity, impacto
    
    def _determine_action(self, opportunity: float) -> str:
        """
        Determina a ação baseada na oportunidade calculada
        """
        if opportunity > self.threshold_opportunity:
            return "Redução"
        elif opportunity < -self.threshold_opportunity:
            return "Aumento"
        else:
            return "Manter"
    
    def _selecionar_rotas_representativas(self, df_cluster: pd.DataFrame) -> pd.DataFrame:
        """
        Seleciona rotas representativas de cada cluster para cálculo de médias
        """
        if len(df_cluster) <= 1:
            return df_cluster
        
        # Calcular EP (Excesso de Preço) total do cluster
        media_cluster = df_cluster['preco_ton_km'].mean()
        subset_contrib = df_cluster[df_cluster['preco_ton_km'] > media_cluster]
        
        if subset_contrib.empty:
            return df_cluster
        
        EP_total = ((subset_contrib['preco_ton_km'] - media_cluster) *
                    subset_contrib['distancia_km'] *
                    subset_contrib['volume_ton']).sum()
        
        if EP_total == 0:
            return df_cluster
        
        # Selecionar rotas que minimizam o EP total
        rotas_selecionadas = set(df_cluster.index)
        
        while len(rotas_selecionadas) > 1:
            candidatos = []
            for rota_idx in rotas_selecionadas:
                rotas_tmp = rotas_selecionadas - {rota_idx}
                subset_tmp = df_cluster.loc[list(rotas_tmp)]
                
                if len(subset_tmp) == 0:
                    continue
                
                media_tmp = subset_tmp['preco_ton_km'].mean()
                subset_contrib_tmp = subset_tmp[subset_tmp['preco_ton_km'] > media_tmp]
                
                if subset_contrib_tmp.empty:
                    EP_tmp = 0.0
                else:
                    EP_tmp = ((subset_contrib_tmp['preco_ton_km'] - media_tmp) *
                              subset_contrib_tmp['distancia_km'] *
                              subset_contrib_tmp['volume_ton']).sum()
                
                variacao = (EP_total - EP_tmp) / EP_total if EP_total != 0 else 0
                candidatos.append((variacao, rota_idx, EP_tmp))
            
            if not candidatos:
                break
                
            candidatos.sort(key=lambda x: x[0])
            menor_variacao, rota_remover, novo_EP_total = candidatos[0]
            
            # Parar se a variação for muito pequena
            if menor_variacao >= 0.01:
                break
                
            rotas_selecionadas.remove(rota_remover)
            EP_total = novo_EP_total
        
        return df_cluster.loc[list(rotas_selecionadas)]
    
    def analyze_reduction_opportunities(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analisa oportunidades de redução de preços com funcionalidades integradas
        """
        logger.info("Iniciando análise de oportunidades de redução...")
        
        # Calcular preços médios por micro-região e cluster
        price_analysis = self.price_calculator.calculate_average_price_by_microregion(df)
        
        # Criar cópia para análise
        df_analysis = df.copy()
        
        # Extrair micro-região de origem
        df_analysis['microregiao_origem'] = df_analysis['centro_origem'].apply(
            self.price_calculator._extract_microregion
        )
        
        # Extrair meso-região
        if 'rota_mesoregiao' in df_analysis.columns:
            df_analysis['mesoregiao'] = df_analysis['rota_mesoregiao']
        else:
            df_analysis['mesoregiao'] = df_analysis['microregiao_origem']
        
        # Classificar faixa de distância
        df_analysis['faixa_distancia'] = df_analysis['distancia_km'].apply(
            self.price_calculator._classificar_faixa_distancia
        )
        
        # Criar cluster ID
        df_analysis['cluster_id'] = df_analysis['mesoregiao'] + ' | ' + df_analysis['faixa_distancia']
        
        # Calcular preço por TKU se não existir
        if 'preco_ton_km' not in df_analysis.columns:
            df_analysis['preco_ton_km'] = (
                df_analysis['custo_sup_tku'] / 
                (df_analysis['volume_ton'] * df_analysis['distancia_km'])
            )
        
        # Mesclar com preços médios
        df_analysis = df_analysis.merge(
            price_analysis[['microregiao_origem', 'preco_medio_tku', 'cluster_id', 'preco_medio_cluster']], 
            on=['microregiao_origem', 'cluster_id'], 
            how='left'
        )
        
        # Selecionar rotas representativas por cluster para cálculo de médias
        df_analysis['selecionada'] = False
        for cluster_id, grupo in df_analysis.groupby('cluster_id'):
            rotas_representativas = self._selecionar_rotas_representativas(grupo)
            df_analysis.loc[rotas_representativas.index, 'selecionada'] = True
        
        # Calcular oportunidade e impacto
        oportunidades = []
        for idx, row in df_analysis.iterrows():
            oportunidade, impacto = self._calculate_opportunity(
                row['preco_ton_km'],
                row['preco_medio_tku'],
                row['preco_medio_cluster'],
                row['volume_ton'],
                row['distancia_km']
            )
            oportunidades.append((oportunidade, impacto))
        
        df_analysis['oportunidade_brl_ton_km'], df_analysis['impacto_estrategico'] = zip(*oportunidades)
        
        # Determinar ação
        df_analysis['acao'] = df_analysis['oportunidade_brl_ton_km'].apply(
            self._determine_action
        )
        
        # Análise temporal (se houver dados de data)
        if 'data_faturamento' in df_analysis.columns:
            df_analysis['analise_temporal'] = df_analysis.apply(
                lambda row: self.price_calculator._analisar_tendencia_temporal(
                    df_analysis[df_analysis['centro_origem'] == row['centro_origem']], 
                    row['centro_origem']
                ) if row['selecionada'] else "-",
                axis=1
            )
        else:
            df_analysis['analise_temporal'] = "-"
        
        # Ordenar por oportunidade (maior para menor)
        df_analysis = df_analysis.sort_values('oportunidade_brl_ton_km', ascending=False)
        
        # Selecionar colunas relevantes para o relatório
        columns_mapping = {
            'centro_origem': 'Centro Origem',
            'volume_ton': 'Volume (TON)',
            'distancia_km': 'Distância (KM)',
            'custo_sup_tku': 'Custo Sup (TKU)',
            'preco_medio_tku': '04.01 - Média MicroRegião - Preço SUP (BRL/TON/KM)',
            'preco_medio_cluster': '04.02 - Média Cluster - Preço SUP (BRL/TON/KM)',
            'oportunidade_brl_ton_km': 'Oport. (BRL/TON/KM)',
            'impacto_estrategico': 'Impacto Estratégico',
            'acao': 'Ação',
            'analise_temporal': 'Análise Temporal',
            'cluster_id': 'Cluster (Meso + Faixa)',
            'selecionada': 'Rota Representativa'
        }
        
        # Renomear colunas para o formato do relatório
        df_report = df_analysis[columns_mapping.keys()].copy()
        df_report.columns = columns_mapping.values()
        
        # Calcular novo valor sugerido baseado na média do cluster ou micro-região
        df_report['Novo Valor Sugerido (BRL/TON/KM)'] = df_report.apply(
            lambda row: (
                row['04.02 - Média Cluster - Preço SUP (BRL/TON/KM)']
                if pd.notna(row['04.02 - Média Cluster - Preço SUP (BRL/TON/KM)'])
                else row['04.01 - Média MicroRegião - Preço SUP (BRL/TON/KM)']
            ) if row['Ação'] == 'Redução'
            else row['Custo Sup (TKU)'] / (row['Volume (TON)'] * row['Distância (KM)']),
            axis=1
        )
        
        # Formatar números para melhor visualização
        numeric_columns = [
            'Volume (TON)', 'Distância (KM)', 'Custo Sup (TKU)',
            '04.01 - Média MicroRegião - Preço SUP (BRL/TON/KM)',
            '04.02 - Média Cluster - Preço SUP (BRL/TON/KM)',
            'Oport. (BRL/TON/KM)', 'Novo Valor Sugerido (BRL/TON/KM)'
        ]
        
        for col in numeric_columns:
            if col in df_report.columns:
                df_report[col] = df_report[col].round(4)
        
        # Formatar colunas específicas
        df_report['Volume (TON)'] = df_report['Volume (TON)'].round(2)
        df_report['Distância (KM)'] = df_report['Distância (KM)'].round(2)
        
        logger.info(f"Análise concluída. {len(df_report)} rotas analisadas")
        logger.info(f"Oportunidades de redução identificadas: {len(df_report[df_report['Ação'] == 'Redução'])}")
        logger.info(f"Rotas representativas selecionadas: {len(df_report[df_report['Rota Representativa'] == True])}")
        
        return df_report
