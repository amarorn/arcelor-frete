"""
Testes para o analisador de oportunidades de redução de preços
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from utils.ml.opportunity_analyzer import MicroRegionPriceCalculator, OpportunityAnalyzer


class TestMicroRegionPriceCalculator:
    """Testes para MicroRegionPriceCalculator"""
    
    def setup_method(self):
        self.calculator = MicroRegionPriceCalculator()
    
    def test_extract_microregion(self):
        """Testa extração de micro-regiões"""
        assert self.calculator._extract_microregion("JOÃO MONLEVADE-SABARÁ") == "JOÃO MONLEVADE"
        assert self.calculator._extract_microregion("ITABIRA-BELO HORIZONTE") == "ITABIRA"
        assert self.calculator._extract_microregion("USINA MONLEVADE-CONTAGEM") == "JOÃO MONLEVADE"
        assert self.calculator._extract_microregion("UNKNOWN_LOCATION") == "UNKNOWN_LOCATION"
    
    def test_classificar_faixa_distancia(self):
        """Testa classificação de faixas de distância"""
        assert self.calculator._classificar_faixa_distancia(50) == "<= 100"
        assert self.calculator._classificar_faixa_distancia(150) == "101 a 150"
        assert self.calculator._classificar_faixa_distancia(500) == "401 a 500"
        assert self.calculator._classificar_faixa_distancia(2500) == "> 2000"
    
    def test_analisar_tendencia_temporal(self):
        """Testa análise de tendência temporal"""
        # Criar dados de teste com tendência de redução mais forte
        dates = [datetime.now() - timedelta(days=i) for i in range(5, 0, -1)]
        df_teste = pd.DataFrame({
            'data_faturamento': dates,
            'custo_sup_tku': [200, 150, 100, 50, 0],  # Tendência de redução mais forte
            'volume_ton': [100, 100, 100, 100, 100],
            'distancia_km': [50, 50, 50, 50, 50]
        })
        
        resultado = self.calculator._analisar_tendencia_temporal(df_teste, "TESTE")
        assert "REDUÇÃO" in resultado
        
        # Teste com dados insuficientes
        df_insuficiente = df_teste.head(1)
        resultado = self.calculator._analisar_tendencia_temporal(df_insuficiente, "TESTE")
        assert resultado == "DADOS INSUFICIENTES"
    
    def test_calculate_average_price_by_microregion(self):
        """Testa cálculo de preços médios por micro-região e cluster"""
        df = pd.DataFrame({
            'centro_origem': ['JOÃO MONLEVADE-SABARÁ', 'JOÃO MONLEVADE-CONTAGEM', 'ITABIRA-BH'],
            'volume_ton': [100, 200, 150],
            'distancia_km': [50, 60, 100],
            'custo_sup_tku': [500, 1200, 1500],
            'rota_mesoregiao': ['METROPOLITANA', 'METROPOLITANA', 'VALE DO AÇO']
        })
        
        resultado = self.calculator.calculate_average_price_by_microregion(df)
        
        assert len(resultado) > 0
        assert 'preco_medio_tku' in resultado.columns
        assert 'preco_medio_cluster' in resultado.columns
        assert 'cluster_id' in resultado.columns

class TestOpportunityAnalyzer:
    """Testes para OpportunityAnalyzer"""
    
    def setup_method(self):
        self.analyzer = OpportunityAnalyzer()
    
    def test_calculate_opportunity(self):
        """Testa cálculo de oportunidade e classificação de impacto"""
        oportunidade, impacto = self.analyzer._calculate_opportunity(
            current_price=0.15,  # Preço atual
            avg_microregion_price=0.10,  # Média micro-região
            avg_cluster_price=0.12,  # Média cluster
            volume=500,  # Volume alto
            distance=100
        )
        
        assert oportunidade > 0  # Deve ser positiva (oportunidade de redução)
        assert impacto in ["BAIXO", "MÉDIO", "ALTO"]
        
        # Teste com volume baixo
        oportunidade, impacto = self.analyzer._calculate_opportunity(
            current_price=0.15,
            avg_microregion_price=0.10,
            avg_cluster_price=0.12,
            volume=50,  # Volume baixo
            distance=100
        )
        
        assert impacto == "NÃO APLICÁVEL"
    
    def test_determine_action(self):
        """Testa determinação de ação"""
        assert self.analyzer._determine_action(0.03) == "Redução"  # > threshold
        assert self.analyzer._determine_action(-0.03) == "Aumento"  # < -threshold
        assert self.analyzer._determine_action(0.01) == "Manter"  # Entre thresholds
    
    def test_selecionar_rotas_representativas(self):
        """Testa seleção de rotas representativas"""
        df_cluster = pd.DataFrame({
            'preco_ton_km': [0.10, 0.12, 0.15, 0.20],
            'distancia_km': [100, 100, 100, 100],
            'volume_ton': [100, 100, 100, 100]
        })
        
        rotas_selecionadas = self.analyzer._selecionar_rotas_representativas(df_cluster)
        
        assert len(rotas_selecionadas) <= len(df_cluster)
        assert len(rotas_selecionadas) > 0
    
    def test_analyze_reduction_opportunities_integrated(self):
        """Testa análise completa integrada"""
        # Criar dados de teste mais realistas com oportunidades de redução
        df = pd.DataFrame({
            'centro_origem': [
                'JOÃO MONLEVADE-SABARÁ', 'JOÃO MONLEVADE-CONTAGEM', 'JOÃO MONLEVADE-SANTA LUZIA',
                'ITABIRA-BELO HORIZONTE', 'ITABIRA-CONTAGEM', 'ITABIRA-SABARÁ'
            ],
            'volume_ton': [800, 1200, 600, 900, 1100, 700],
            'distancia_km': [35, 45, 40, 120, 110, 125],
            'custo_sup_tku': [7000, 12000, 5600, 24000, 22000, 18000],  # Custos mais altos para gerar oportunidades
            'data_faturamento': [datetime.now()] * 6,
            'rota_mesoregiao': ['METROPOLITANA', 'METROPOLITANA', 'METROPOLITANA', 
                               'VALE DO AÇO', 'VALE DO AÇO', 'VALE DO AÇO']
        })
        
        # Usar threshold menor para garantir oportunidades
        analyzer = OpportunityAnalyzer(threshold_opportunity=0.01)
        resultado = analyzer.analyze_reduction_opportunities(df)
        
        # Verificar colunas esperadas
        colunas_esperadas = [
            'Centro Origem', 'Volume (TON)', 'Distância (KM)', 'Custo Sup (TKU)',
            '04.01 - Média MicroRegião - Preço SUP (BRL/TON/KM)',
            '04.02 - Média Cluster - Preço SUP (BRL/TON/KM)',
            'Oport. (BRL/TON/KM)', 'Impacto Estratégico', 'Ação', 'Análise Temporal',
            'Cluster (Meso + Faixa)', 'Rota Representativa', 'Novo Valor Sugerido (BRL/TON/KM)'
        ]
        
        for coluna in colunas_esperadas:
            assert coluna in resultado.columns, f"Coluna {coluna} não encontrada"
        
        # Verificar que há oportunidades de redução
        rotas_reducao = resultado[resultado['Ação'] == 'Redução']
        assert len(rotas_reducao) > 0, "Deve haver pelo menos uma oportunidade de redução"
        
        # Verificar classificação de impacto
        impactos = resultado['Impacto Estratégico'].unique()
        assert len(impactos) > 1, "Deve haver diferentes níveis de impacto"
        
        # Verificar rotas representativas
        rotas_representativas = resultado[resultado['Rota Representativa'] == True]
        assert len(rotas_representativas) > 0, "Deve haver rotas representativas selecionadas"

class TestIntegration:
    """Testes de integração entre componentes"""
    
    def test_full_workflow(self):
        """Testa fluxo completo de análise"""
        # Dados de teste
        df = pd.DataFrame({
            'centro_origem': ['JOÃO MONLEVADE-TESTE', 'ITABIRA-TESTE'],
            'volume_ton': [1000, 800],
            'distancia_km': [100, 150],
            'custo_sup_tku': [10000, 12000],
            'data_faturamento': [datetime.now(), datetime.now()],
            'rota_mesoregiao': ['METROPOLITANA', 'VALE DO AÇO']
        })
        
        # Executar análise completa
        analyzer = OpportunityAnalyzer()
        resultado = analyzer.analyze_reduction_opportunities(df)
        
        # Verificações básicas
        assert len(resultado) == 2
        assert 'Ação' in resultado.columns
        assert 'Impacto Estratégico' in resultado.columns
        assert 'Novo Valor Sugerido (BRL/TON/KM)' in resultado.columns
        
        # Verificar que os cálculos fazem sentido
        for idx, row in resultado.iterrows():
            if row['Ação'] == 'Redução':
                assert row['Oport. (BRL/TON/KM)'] > 0
                assert row['Novo Valor Sugerido (BRL/TON/KM)'] > 0

if __name__ == "__main__":
    pytest.main([__file__])
