"""
Testes unitários para o TKUTrendAnalyzer
"""

import pytest
import polars as pl
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

# Importar a classe a ser testada
from utils.ml.tku_trend_analyzer import TKUTrendAnalyzer


@pytest.fixture
def sample_data():
    """Cria dados de exemplo para testes"""
    # Configurar seed para reprodutibilidade
    np.random.seed(42)
    
    # Criar datas de exemplo (últimos 12 meses)
    start_date = datetime.now() - timedelta(days=365)
    dates = [start_date + timedelta(days=i*7) for i in range(52)]  # 52 semanas
    
    data = []
    for i, date in enumerate(dates):
        # Simular tendência de alta
        base_tku = 0.8 + (i * 0.001)  # Tendência de alta gradual
        
        # Adicionar sazonalidade
        seasonal_factor = 1.0
        if date.month in [12, 1]:
            seasonal_factor = 1.2  # 20% mais caro
        elif date.month in [6, 7]:
            seasonal_factor = 0.9  # 10% mais barato
        
        # Adicionar ruído
        noise = np.random.normal(0, 0.03)
        
        # TKU final
        tku = (base_tku * seasonal_factor) + noise
        
        # Volume e distância
        volume = np.random.uniform(50, 150)
        distancia = np.random.uniform(100, 400)
        
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


@pytest.fixture
def analyzer():
    """Cria instância do analisador para testes"""
    return TKUTrendAnalyzer()


class TestTKUTrendAnalyzer:
    """Testes para a classe TKUTrendAnalyzer"""
    
    def test_initialization(self, analyzer):
        """Testa inicialização da classe"""
        assert analyzer is not None
        assert hasattr(analyzer, 'periods')
        assert hasattr(analyzer, 'trend_thresholds')
        assert hasattr(analyzer, 'analysis_cache')
        
        # Verificar configurações padrão
        assert analyzer.periods['short_term'] == 90
        assert analyzer.periods['medium_term'] == 180
        assert analyzer.periods['long_term'] == 365
        
        # Verificar thresholds
        assert analyzer.trend_thresholds['stable_threshold'] == 0.05
        assert analyzer.trend_thresholds['moderate_threshold'] == 0.15
        assert analyzer.trend_thresholds['fast_threshold'] == 0.25
    
    def test_filter_route_data(self, analyzer, sample_data):
        """Testa filtragem de dados por rota"""
        # Testar com rota existente
        df_filtered = analyzer._filter_route_data(sample_data, 'SABARÁ')
        assert not df_filtered.is_empty()
        assert len(df_filtered) == len(sample_data)
        
        # Testar com rota inexistente
        df_filtered = analyzer._filter_route_data(sample_data, 'ROTA_INEXISTENTE')
        assert df_filtered.is_empty()
    
    def test_calculate_tku_historical(self, analyzer, sample_data):
        """Testa cálculo de TKU histórico"""
        df_tku = analyzer._calculate_tku_historical(sample_data)
        
        # Verificar se a coluna foi criada
        assert 'tku_historico' in df_tku.columns
        
        # Verificar se todos os valores são finitos e positivos
        tku_values = df_tku['tku_historico'].to_numpy()
        assert np.all(np.isfinite(tku_values))
        assert np.all(tku_values > 0)
        
        # Verificar se o cálculo está correto
        for row in df_tku.iter_rows(named=True):
            expected_tku = row['frete_brl'] / (row['volume_ton'] * row['distancia_km'])
            assert abs(row['tku_historico'] - expected_tku) < 1e-10
    
    def test_analyze_short_term_trend(self, analyzer, sample_data):
        """Testa análise de tendência de curto prazo"""
        # Calcular TKU histórico primeiro
        df_tku = analyzer._calculate_tku_historical(sample_data)
        
        # Analisar tendência de 3 meses
        trend_analysis = analyzer._analyze_short_term_trend(df_tku)
        
        # Verificar estrutura da resposta
        assert 'periodo' in trend_analysis
        assert 'tendencia' in trend_analysis
        assert 'velocidade' in trend_analysis
        assert 'confianca' in trend_analysis
        assert 'tku_atual' in trend_analysis
        assert 'variacao_percentual' in trend_analysis
        
        # Verificar valores
        assert trend_analysis['periodo'] == '3 meses'
        assert trend_analysis['dados_disponiveis'] == True
        assert trend_analysis['tku_atual'] > 0
        assert trend_analysis['confianca'] >= 0
        assert trend_analysis['confianca'] <= 100
    
    def test_analyze_medium_term_trend(self, analyzer, sample_data):
        """Testa análise de tendência de médio prazo"""
        df_tku = analyzer._calculate_tku_historical(sample_data)
        trend_analysis = analyzer._analyze_medium_term_trend(df_tku)
        
        assert trend_analysis['periodo'] == '6 meses'
        assert trend_analysis['dados_disponiveis'] == True
    
    def test_analyze_long_term_trend(self, analyzer, sample_data):
        """Testa análise de tendência de longo prazo"""
        df_tku = analyzer._calculate_tku_historical(sample_data)
        trend_analysis = analyzer._analyze_long_term_trend(df_tku)
        
        assert trend_analysis['periodo'] == '12 meses'
        assert trend_analysis['dados_disponiveis'] == True
    
    def test_calculate_trend_metrics(self, analyzer, sample_data):
        """Testa cálculo de métricas de tendência"""
        df_tku = analyzer._calculate_tku_historical(sample_data)
        
        # Filtrar dados dos últimos 3 meses
        cutoff_date = datetime.now() - timedelta(days=90)
        df_period = df_tku.filter(pl.col('data_faturamento') >= pl.lit(cutoff_date))
        
        if not df_period.is_empty():
            direction, speed, confidence = analyzer._calculate_trend_metrics(df_period)
            
            # Verificar se os valores são válidos
            assert direction in ['ESTÁVEL', 'SUBINDO', 'CAINDO', 'ERRO']
            assert speed in ['ESTÁVEL', 'LENTO', 'MÉDIO', 'RÁPIDO', 'N/A']
            assert 0 <= confidence <= 100
    
    def test_analyze_seasonality(self, analyzer, sample_data):
        """Testa análise de sazonalidade"""
        df_tku = analyzer._calculate_tku_historical(sample_data)
        seasonality = analyzer._analyze_seasonality(df_tku)
        
        # Verificar estrutura da resposta
        assert 'tku_medio_geral' in seasonality
        assert 'analise_mensal' in seasonality
        assert 'meses_alta_temporada' in seasonality
        assert 'meses_baixa_temporada' in seasonality
        assert 'variacao_sazonal' in seasonality
        assert 'padrao_sazonal_detectado' in seasonality
        
        # Verificar valores
        assert seasonality['tku_medio_geral'] > 0
        assert seasonality['variacao_sazonal'] >= 0
        assert isinstance(seasonality['padrao_sazonal_detectado'], bool)
    
    def test_analyze_route_benchmark(self, analyzer, sample_data):
        """Testa análise de benchmark de rota"""
        df_tku = analyzer._calculate_tku_historical(sample_data)
        benchmark = analyzer._analyze_route_benchmark(sample_data, 'SABARÁ', df_tku)
        
        # Verificar estrutura da resposta
        assert 'tku_rota' in benchmark
        assert 'tku_benchmark' in benchmark
        assert 'posicionamento' in benchmark
        assert 'recomendacao' in benchmark
        assert 'acao_necessaria' in benchmark
        assert 'economia_potencial' in benchmark
        
        # Verificar valores
        assert benchmark['tku_rota'] > 0
        assert benchmark['tku_benchmark'] > 0
        assert benchmark['economia_potencial'] >= 0
    
    def test_generate_recommendations(self, analyzer):
        """Testa geração de recomendações"""
        # Criar dados de teste para tendências
        trends_analysis = {
            '3_meses': {
                'tendencia': 'SUBINDO',
                'velocidade': 'MÉDIO',
                'dados_disponiveis': True
            },
            '6_meses': {
                'tendencia': 'SUBINDO',
                'dados_disponiveis': True
            }
        }
        
        benchmark_analysis = {
            'posicionamento': 'REGULAR (Top 75%)',
            'economia_potencial': 0.15
        }
        
        recommendations = analyzer._generate_recommendations(trends_analysis, benchmark_analysis)
        
        # Verificar se foram geradas recomendações
        assert len(recommendations) > 0
        
        # Verificar estrutura das recomendações
        for rec in recommendations:
            assert 'tipo' in rec
            assert 'acao' in rec
            assert 'justificativa' in rec
            assert 'economia_esperada' in rec
            assert 'prazo' in rec
            assert 'dificuldade' in rec
    
    def test_analyze_route_tku_trends_complete(self, analyzer, sample_data):
        """Testa análise completa de uma rota"""
        analysis = analyzer.analyze_route_tku_trends(sample_data, 'SABARÁ')
        
        # Verificar se não há erro
        assert 'erro' not in analysis
        
        # Verificar estrutura da análise completa
        required_keys = [
            'rota_id', 'data_analise', 'periodo_analise', 'tku_atual',
            'tku_anterior', 'variacao_percentual', 'tendencias',
            'sazonalidade', 'benchmark', 'recomendacoes', 'status_geral'
        ]
        
        for key in required_keys:
            assert key in analysis
        
        # Verificar valores específicos
        assert analysis['rota_id'] == 'SABARÁ'
        assert analysis['tku_atual'] > 0
        assert analysis['variacao_percentual'] is not None
        assert len(analysis['tendencias']) == 3  # 3, 6, 12 meses
        assert len(analysis['recomendacoes']) > 0
    
    def test_cache_functionality(self, analyzer, sample_data):
        """Testa funcionalidade de cache"""
        # Primeira análise
        analysis1 = analyzer.analyze_route_tku_trends(sample_data, 'SABARÁ')
        
        # Verificar se foi salva no cache
        cached_analysis = analyzer.get_cached_analysis('SABARÁ')
        assert cached_analysis is not None
        assert cached_analysis['rota_id'] == 'SABARÁ'
        
        # Segunda análise (deve usar cache)
        analysis2 = analyzer.analyze_route_tku_trends(sample_data, 'SABARÁ')
        
        # Verificar se os resultados são idênticos
        assert analysis1['rota_id'] == analysis2['rota_id']
        assert analysis1['tku_atual'] == analysis2['tku_atual']
        
        # Limpar cache
        analyzer.clear_cache()
        assert analyzer.get_cached_analysis('SABARÁ') is None
    
    def test_get_analysis_summary(self, analyzer, sample_data):
        """Testa obtenção de resumo da análise"""
        # Executar análise primeiro
        analyzer.analyze_route_tku_trends(sample_data, 'SABARÁ')
        
        # Obter resumo
        summary = analyzer.get_analysis_summary('SABARÁ')
        
        # Verificar se não há erro
        assert 'erro' not in summary
        
        # Verificar estrutura do resumo
        required_keys = [
            'rota_id', 'status_geral', 'tku_atual', 'variacao_percentual',
            'tendencia_3m', 'posicionamento_mercado', 'recomendacao_principal'
        ]
        
        for key in required_keys:
            assert key in summary
    
    def test_error_handling(self, analyzer):
        """Testa tratamento de erros"""
        # Testar com DataFrame vazio
        empty_df = pl.DataFrame()
        analysis = analyzer.analyze_route_tku_trends(empty_df, 'ROTA_TESTE')
        
        assert 'erro' in analysis
        assert analysis['status_geral'] == 'SEM DADOS'
        
        # Testar com dados inválidos
        invalid_df = pl.DataFrame({
            'data_faturamento': ['2024-01-01'],
            'volume_ton': [0],  # Volume zero
            'distancia_km': [0],  # Distância zero
            'frete_brl': [100]
        })
        
        analysis = analyzer.analyze_route_tku_trends(invalid_df, 'ROTA_INVALIDA')
        assert 'erro' in analysis or analysis['status_geral'] == 'SEM DADOS'
    
    def test_percentage_change_calculation(self, analyzer, sample_data):
        """Testa cálculo de variação percentual"""
        df_tku = analyzer._calculate_tku_historical(sample_data)
        
        # Calcular variação percentual
        variacao = analyzer._calculate_percentage_change(df_tku)
        
        # Verificar se é um número
        assert isinstance(variacao, (int, float))
        assert not np.isnan(variacao)
    
    def test_route_status_determination(self, analyzer):
        """Testa determinação do status da rota"""
        # Testar diferentes cenários
        scenarios = [
            # Tendência subindo rapidamente
            ({
                '3_meses': {'tendencia': 'SUBINDO', 'velocidade': 'RÁPIDO'},
                '6_meses': {'tendencia': 'SUBINDO', 'velocidade': 'MÉDIO'}
            }, {
                'posicionamento': 'BOM (Top 50%)'
            }, 'CRÍTICO - Ação urgente necessária'),
            
            # Tendência estável
            ({
                '3_meses': {'tendencia': 'ESTÁVEL', 'velocidade': 'ESTÁVEL'},
                '6_meses': {'tendencia': 'ESTÁVEL', 'velocidade': 'ESTÁVEL'}
            }, {
                'posicionamento': 'EXCELENTE (Top 25%)'
            }, 'ESTÁVEL - Manter estratégia atual'),
            
            # Tendência caindo
            ({
                '3_meses': {'tendencia': 'CAINDO', 'velocidade': 'LENTO'},
                '6_meses': {'tendencia': 'CAINDO', 'velocidade': 'LENTO'}
            }, {
                'posicionamento': 'BOM (Top 50%)'
            }, 'POSITIVO - Tendência favorável')
        ]
        
        for trends, benchmark, expected_status in scenarios:
            status = analyzer._get_route_status(trends, benchmark)
            assert status == expected_status


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
