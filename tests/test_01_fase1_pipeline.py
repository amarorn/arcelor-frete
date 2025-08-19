"""
Testes unitários para o pipeline da Fase 1
"""

import pytest
import polars as pl
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Importar o pipeline
from notebooks.fase1_pipeline import run_fase1_pipeline, benchmark_performance


def create_mock_processed_data():
    """Helper para criar dados mockados processados com as colunas corretas"""
    return pl.DataFrame({
        'rota': ['ROTA_A_B', 'ROTA_C_D'] * 50,
        'transportadora': ['TRANSP_1', 'TRANSP_2', 'TRANSP_3'] * 33 + ['TRANSP_1'],
        'volume_ton': [50.0] * 100,
        'distancia_km': [500.0] * 100,
        'frete_brl': [25000.0] * 100,
        'preco_ton_km_calculado': [0.1] * 100,
        'mes': [1] * 100,
        'trimestre': [1] * 100,
        'ano': [2024] * 100
    })


class TestFase1Pipeline:
    """Testes para o pipeline principal da Fase 1"""
    
    def test_benchmark_performance_success(self, sample_data):
        """Testa benchmark de performance bem-sucedido"""
        # Criar arquivo Excel temporário
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
            sample_data.to_excel(tmp.name, index=False)
            temp_file = tmp.name
        
        try:
            # Executar benchmark
            result = benchmark_performance(temp_file)
            
            assert result is not None
            assert 'pandas_time' in result
            assert 'polars_time' in result
            assert 'speedup' in result
            
            # Verificar se os tempos são positivos
            assert result['pandas_time'] > 0
            assert result['polars_time'] > 0
            assert result['speedup'] > 0
            
            # Verificar se Polars é mais rápido
            assert result['speedup'] >= 1.0
            
        finally:
            # Limpar arquivo temporário
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def test_benchmark_performance_file_not_found(self):
        """Testa benchmark com arquivo inexistente"""
        result = benchmark_performance("arquivo_inexistente.xlsx")
        assert result is None
    
    @patch('notebooks.fase1_pipeline.PolarsAdapter')
    @patch('notebooks.fase1_pipeline.FretePricePredictor')
    def test_run_fase1_pipeline_success(self, mock_predictor, mock_adapter, sample_data):
        """Testa execução bem-sucedida do pipeline"""
        # Mock do adaptador
        mock_adapter_instance = Mock()
        mock_adapter.return_value = mock_adapter_instance
        
        # Mock dos dados
        mock_df_raw = pl.from_pandas(sample_data)
        
        # Mock dos dados processados com as colunas corretas
        mock_df_processed = pl.DataFrame({
            'rota': ['ROTA_A_B', 'ROTA_C_D'] * 50,
            'transportadora': ['TRANSP_1', 'TRANSP_2', 'TRANSP_3'] * 33 + ['TRANSP_1'],
            'volume_ton': [50.0] * 100,
            'distancia_km': [500.0] * 100,
            'frete_brl': [25000.0] * 100,
            'preco_ton_km_calculado': [0.1] * 100,
            'mes': [1] * 100,
            'trimestre': [1] * 100,
            'ano': [2024] * 100
        })
        
        mock_adapter_instance.read_excel.return_value = mock_df_raw
        mock_adapter_instance.prepare_data.return_value = mock_df_processed
        
        # Mock das métricas
        mock_route_metrics = pl.DataFrame({
            'rota': ['ROTA_A_B', 'ROTA_C_D'],
            'volume_total': [1000, 800],
            'frete_total': [50000, 40000]
        })
        
        mock_carrier_metrics = pl.DataFrame({
            'transportadora': ['TRANSP_1', 'TRANSP_2'],
            'volume_total': [1200, 600],
            'frete_total': [60000, 30000]
        })
        
        mock_adapter_instance.calculate_route_metrics.return_value = mock_route_metrics
        mock_adapter_instance.calculate_carrier_metrics.return_value = mock_carrier_metrics
        
        # Mock do predictor
        mock_predictor_instance = Mock()
        mock_predictor.return_value = mock_predictor_instance
        
        mock_metrics = {
            'train_r2': 0.75,
            'test_r2': 0.70,
            'train_mae': 0.3,
            'test_mae': 0.35
        }
        
        mock_optimization_report = {
            'economia_potencial_total': 1000000.0,
            'num_oportunidades': 150,
            'top_rotas_otimizacao': [],
            'top_transportadoras_negociacao': [],
            'metricas_modelo': {}
        }
        
        mock_predictor_instance.train.return_value = mock_metrics
        mock_predictor_instance.generate_optimization_report.return_value = mock_optimization_report
        
        # Mock do export
        mock_adapter_instance.export_to_excel.return_value = None
        
        # Executar pipeline
        result = run_fase1_pipeline("test_data.xlsx")
        
        # Verificar resultado
        assert result['success'] is True
        assert 'execution_time' in result
        assert 'metrics' in result
        assert 'optimization_report' in result
        
        # Verificar se os métodos foram chamados
        mock_adapter_instance.read_excel.assert_called_once_with("test_data.xlsx")
        mock_adapter_instance.prepare_data.assert_called_once_with(mock_df_raw)
        mock_predictor_instance.train.assert_called_once_with(mock_df_processed)
        mock_predictor_instance.save_model.assert_called_once()
        mock_predictor_instance.generate_optimization_report.assert_called_once_with(mock_df_processed)
    
    @patch('notebooks.fase1_pipeline.PolarsAdapter')
    def test_run_fase1_pipeline_adapter_error(self, mock_adapter):
        """Testa erro no adaptador"""
        # Mock do adaptador com erro
        mock_adapter_instance = Mock()
        mock_adapter.return_value = mock_adapter_instance
        
        mock_adapter_instance.read_excel.side_effect = Exception("Erro no adaptador")
        
        # Executar pipeline
        result = run_fase1_pipeline("test_data.xlsx")
        
        # Verificar resultado de erro
        assert result['success'] is False
        assert 'error' in result
        assert "Erro no adaptador" in result['error']
    
    @patch('notebooks.fase1_pipeline.PolarsAdapter')
    @patch('notebooks.fase1_pipeline.FretePricePredictor')
    def test_run_fase1_pipeline_ml_error(self, mock_predictor, mock_adapter, sample_data):
        """Testa erro no modelo de ML"""
        # Mock do adaptador funcionando
        mock_adapter_instance = Mock()
        mock_adapter.return_value = mock_adapter_instance
        
        mock_df_raw = pl.from_pandas(sample_data)
        mock_df_processed = pl.from_pandas(sample_data)
        
        mock_adapter_instance.read_excel.return_value = mock_df_raw
        mock_adapter_instance.prepare_data.return_value = mock_df_processed
        mock_adapter_instance.calculate_route_metrics.return_value = pl.DataFrame()
        mock_adapter_instance.calculate_carrier_metrics.return_value = pl.DataFrame()
        
        # Mock do predictor com erro
        mock_predictor_instance = Mock()
        mock_predictor.return_value = mock_predictor_instance
        
        mock_predictor_instance.train.side_effect = Exception("Erro no modelo ML")
        
        # Executar pipeline
        result = run_fase1_pipeline("test_data.xlsx")
        
        # Verificar resultado de erro
        assert result['success'] is False
        assert 'error' in result
        assert "Erro no modelo ML" in result['error']
    
    @patch('notebooks.fase1_pipeline.PolarsAdapter')
    @patch('notebooks.fase1_pipeline.FretePricePredictor')
    def test_run_fase1_pipeline_export_error(self, mock_predictor, mock_adapter, sample_data):
        """Testa erro na exportação"""
        # Mock do adaptador funcionando
        mock_adapter_instance = Mock()
        mock_adapter.return_value = mock_adapter_instance
        
        mock_df_raw = pl.from_pandas(sample_data)
        mock_df_processed = pl.from_pandas(sample_data)
        
        mock_adapter_instance.read_excel.return_value = mock_df_raw
        mock_adapter_instance.prepare_data.return_value = mock_df_processed
        mock_adapter_instance.calculate_route_metrics.return_value = pl.DataFrame()
        mock_adapter_instance.calculate_carrier_metrics.return_value = pl.DataFrame()
        
        # Mock do predictor funcionando
        mock_predictor_instance = Mock()
        mock_predictor.return_value = mock_predictor_instance
        
        mock_metrics = {'train_r2': 0.75, 'test_r2': 0.70}
        mock_optimization_report = {
            'economia_potencial_total': 1000000.0,
            'num_oportunidades': 150,
            'top_rotas_otimizacao': [],
            'top_transportadoras_negociacao': [],
            'metricas_modelo': {}
        }
        
        mock_predictor_instance.train.return_value = mock_metrics
        mock_predictor_instance.generate_optimization_report.return_value = mock_optimization_report
        
        # Mock do export com erro
        mock_adapter_instance.export_to_excel.side_effect = Exception("Erro na exportação")
        
        # Executar pipeline
        result = run_fase1_pipeline("test_data.xlsx")
        
        # Verificar resultado de erro
        assert result['success'] is False
        assert 'error' in result
        assert "Erro na exportação" in result['error']
    
    def test_pipeline_execution_time(self, sample_data):
        """Testa se o tempo de execução é medido corretamente"""
        with patch('notebooks.fase1_pipeline.PolarsAdapter') as mock_adapter, \
             patch('notebooks.fase1_pipeline.FretePricePredictor') as mock_predictor:
            
            # Mock básico funcionando
            mock_adapter_instance = Mock()
            mock_adapter.return_value = mock_adapter_instance
            
            mock_df_raw = pl.from_pandas(sample_data)
            mock_df_processed = create_mock_processed_data()
            
            mock_adapter_instance.read_excel.return_value = mock_df_raw
            mock_adapter_instance.prepare_data.return_value = mock_df_processed
            mock_adapter_instance.calculate_route_metrics.return_value = pl.DataFrame()
            mock_adapter_instance.calculate_carrier_metrics.return_value = pl.DataFrame()
            mock_adapter_instance.export_to_excel.return_value = None
            
            mock_predictor_instance = Mock()
            mock_predictor.return_value = mock_predictor_instance
            
            mock_metrics = {'train_r2': 0.75, 'test_r2': 0.70}
            mock_optimization_report = {
                'economia_potencial_total': 1000000.0,
                'num_oportunidades': 150,
                'top_rotas_otimizacao': [],
                'top_transportadoras_negociacao': [],
                'metricas_modelo': {}
            }
            
            mock_predictor_instance.train.return_value = mock_metrics
            mock_predictor_instance.generate_optimization_report.return_value = mock_optimization_report
            
            # Executar pipeline
            result = run_fase1_pipeline("test_data.xlsx")
            
            # Verificar se o tempo foi medido
            assert result['success'] is True
            assert 'execution_time' in result
            assert isinstance(result['execution_time'], float)
            assert result['execution_time'] > 0
    
    def test_pipeline_output_structure(self, sample_data):
        """Testa estrutura dos outputs do pipeline"""
        with patch('notebooks.fase1_pipeline.PolarsAdapter') as mock_adapter, \
             patch('notebooks.fase1_pipeline.FretePricePredictor') as mock_predictor:
            
            # Mock funcionando
            mock_adapter_instance = Mock()
            mock_adapter.return_value = mock_adapter_instance
            
            mock_df_raw = pl.from_pandas(sample_data)
            mock_df_processed = create_mock_processed_data()
            
            mock_adapter_instance.read_excel.return_value = mock_df_raw
            mock_adapter_instance.prepare_data.return_value = mock_df_processed
            mock_adapter_instance.calculate_route_metrics.return_value = pl.DataFrame()
            mock_adapter_instance.calculate_carrier_metrics.return_value = pl.DataFrame()
            mock_adapter_instance.export_to_excel.return_value = None
            
            mock_predictor_instance = Mock()
            mock_predictor.return_value = mock_predictor_instance
            
            mock_metrics = {'train_r2': 0.75, 'test_r2': 0.70}
            mock_optimization_report = {
                'economia_potencial_total': 1000000.0,
                'num_oportunidades': 150,
                'top_rotas_otimizacao': [],
                'top_transportadoras_negociacao': [],
                'metricas_modelo': {}
            }
            
            mock_predictor_instance.train.return_value = mock_metrics
            mock_predictor_instance.generate_optimization_report.return_value = mock_optimization_report
            
            # Executar pipeline
            result = run_fase1_pipeline("test_data.xlsx")
            
            # Verificar estrutura do resultado
            assert 'success' in result
            assert 'execution_time' in result
            assert 'metrics' in result
            assert 'optimization_report' in result
            
            # Verificar se as métricas estão corretas
            assert result['metrics']['train_r2'] == 0.75
            assert result['metrics']['test_r2'] == 0.70
            
            # Verificar se o relatório está correto
            assert result['optimization_report']['economia_potencial_total'] == 1000000.0
            assert result['optimization_report']['num_oportunidades'] == 150


class TestPipelineIntegration:
    """Testes de integração do pipeline"""
    
    def test_pipeline_with_real_data_structure(self):
        """Testa se o pipeline funciona com estrutura de dados real"""
        # Verificar se os módulos podem ser importados
        try:
            from notebooks.fase1_pipeline import run_fase1_pipeline, benchmark_performance
            assert True
        except ImportError as e:
            pytest.fail(f"Falha ao importar módulos: {e}")
    
    def test_pipeline_function_signatures(self):
        """Testa assinaturas das funções do pipeline"""
        from notebooks.fase1_pipeline import run_fase1_pipeline, benchmark_performance
        
        # Verificar se as funções existem e têm os parâmetros corretos
        import inspect
        
        # Verificar run_fase1_pipeline
        sig = inspect.signature(run_fase1_pipeline)
        assert 'data_file' in sig.parameters
        assert sig.parameters['data_file'].default is None  # Valor padrão mudou para None
        
        # Verificar benchmark_performance
        sig = inspect.signature(benchmark_performance)
        assert 'data_file' in sig.parameters
        assert sig.parameters['data_file'].default is None  # Valor padrão mudou para None
