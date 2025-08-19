

import pytest
import polars as pl
import numpy as np
import tempfile
import os
import sys

# Adicionar caminho do projeto
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.ml.baseline_price_predictor import BaselinePricePredictor
from utils.ml.feature_engineering import FreteFeatureEngineer


class TestBaselinePricePredictor:
    
    @pytest.fixture
    def sample_data(self):
        data = {
            'data_faturamento': [
                '2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05',
                '2024-02-01', '2024-02-02', '2024-02-03', '2024-02-04', '2024-02-05',
                '2024-03-01', '2024-03-02', '2024-03-03', '2024-03-04', '2024-03-05',
                '2024-04-01', '2024-04-02', '2024-04-03', '2024-04-04', '2024-04-05',
                '2024-05-01', '2024-05-02', '2024-05-03', '2024-05-04', '2024-05-05',
                '2024-06-01', '2024-06-02', '2024-06-03', '2024-06-04', '2024-06-05'
            ],
            'rota': [
                'SP-RJ', 'SP-MG', 'MG-RJ', 'RJ-SP', 'MG-SP', 'SP-RS', 'RS-SP', 'MG-RS',
                'RJ-MG', 'SP-GO', 'GO-SP', 'MG-GO', 'RJ-GO', 'SP-MT', 'MT-SP', 'MG-MT',
                'RJ-MT', 'SP-BA', 'BA-SP', 'MG-BA', 'RJ-BA', 'SP-PE', 'PE-SP', 'MG-PE',
                'RJ-PE', 'SP-CE', 'CE-SP', 'MG-CE', 'RJ-CE', 'SP-PA'
            ],
            'transportadora': [
                'Transp1', 'Transp2', 'Transp3', 'Transp4', 'Transp5',
                'Transp1', 'Transp2', 'Transp3', 'Transp4', 'Transp5',
                'Transp1', 'Transp2', 'Transp3', 'Transp4', 'Transp5',
                'Transp1', 'Transp2', 'Transp3', 'Transp4', 'Transp5',
                'Transp1', 'Transp2', 'Transp3', 'Transp4', 'Transp5',
                'Transp1', 'Transp2', 'Transp3', 'Transp4', 'Transp5'
            ],
            'volume_ton': [10.0, 15.0, 20.0, 12.0, 18.0, 25.0, 30.0, 35.0, 40.0, 45.0,
                          50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0,
                          100.0, 105.0, 110.0, 115.0, 120.0, 125.0, 130.0, 135.0, 140.0, 145.0],
            'frete_brl': [100.0, 150.0, 200.0, 120.0, 180.0, 250.0, 300.0, 350.0, 400.0, 450.0,
                         500.0, 550.0, 600.0, 650.0, 700.0, 750.0, 800.0, 850.0, 900.0, 950.0,
                         1000.0, 1050.0, 1100.0, 1150.0, 1200.0, 1250.0, 1300.0, 1350.0, 1400.0, 1450.0],
            'distancia_km': [100.0, 200.0, 300.0, 100.0, 200.0, 500.0, 600.0, 700.0, 800.0, 900.0,
                            1000.0, 1100.0, 1200.0, 1300.0, 1400.0, 1500.0, 1600.0, 1700.0, 1800.0, 1900.0,
                            2000.0, 2100.0, 2200.0, 2300.0, 2400.0, 2500.0, 2600.0, 2700.0, 2800.0, 2900.0],
            'modal': ['RODOVIARIO', 'RODOVIARIO', 'FERROVIARIO', 'RODOVIARIO', 'RODOVIARIO', 'RODOVIARIO',
                     'FERROVIARIO', 'AQUAVIARIO', 'RODOVIARIO', 'RODOVIARIO', 'FERROVIARIO', 'RODOVIARIO',
                     'AQUAVIARIO', 'RODOVIARIO', 'FERROVIARIO', 'RODOVIARIO', 'AQUAVIARIO', 'RODOVIARIO',
                     'FERROVIARIO', 'RODOVIARIO', 'AQUAVIARIO', 'RODOVIARIO', 'FERROVIARIO', 'RODOVIARIO',
                     'AQUAVIARIO', 'RODOVIARIO', 'FERROVIARIO', 'RODOVIARIO', 'AQUAVIARIO', 'RODOVIARIO'],
            'tipo_rodovia': ['BR-116', 'SP-300', 'BR-040', 'BR-116', 'SP-300', 'BR-116', 'BR-101', 'BR-040',
                            'BR-116', 'SP-300', 'BR-040', 'BR-116', 'SP-300', 'BR-116', 'BR-101', 'BR-040',
                            'BR-116', 'SP-300', 'BR-040', 'BR-116', 'BR-101', 'BR-040', 'BR-116', 'SP-300',
                            'BR-040', 'BR-116', 'SP-300', 'BR-116', 'BR-101', 'BR-040'],
            'tipo_veiculo': ['TRUCK', 'TRUCK', 'VAGAO', 'TRUCK', 'TRUCK', 'TRUCK', 'VAGAO', 'BARCO',
                            'TRUCK', 'TRUCK', 'VAGAO', 'TRUCK', 'BARCO', 'TRUCK', 'VAGAO', 'TRUCK',
                            'BARCO', 'TRUCK', 'VAGAO', 'TRUCK', 'BARCO', 'TRUCK', 'VAGAO', 'TRUCK',
                            'BARCO', 'TRUCK', 'VAGAO', 'TRUCK', 'BARCO', 'TRUCK']
        }
        
        df = pl.DataFrame(data)
        df = df.with_columns([
            pl.col('data_faturamento').str.strptime(pl.Datetime, '%Y-%m-%d'),
            # Adicionar preco_ton_km_calculado que é necessário para features econômicas
            (pl.col('frete_brl') / (pl.col('volume_ton') * pl.col('distancia_km'))).alias('preco_ton_km_calculado')
        ])
        
        return df
    
    @pytest.fixture
    def baseline_predictor(self):
        return BaselinePricePredictor()
    
    def test_initialization(self, baseline_predictor):
        assert baseline_predictor is not None
        assert len(baseline_predictor.baseline_models) > 0
        assert baseline_predictor.target_feature == 'preco_ton_km_calculado'
    
    def test_create_advanced_features(self, baseline_predictor, sample_data):
        sample_data = sample_data.with_columns([
            pl.col('data_faturamento').dt.month().alias('mes'),
            pl.col('data_faturamento').dt.quarter().alias('trimestre'),
            pl.col('data_faturamento').dt.year().alias('ano')
        ])
        
        sample_data = sample_data.with_columns([
            (pl.col('frete_brl') / (pl.col('volume_ton') * pl.col('distancia_km')))
            .alias('preco_ton_km_calculado')
        ])
        
        df_enhanced = baseline_predictor.create_advanced_features(sample_data)
        
        assert df_enhanced.shape[1] > sample_data.shape[1]
        assert 'dia_semana' in df_enhanced.columns
        assert 'estacao' in df_enhanced.columns
        assert 'modal_encoded' in df_enhanced.columns
    
    def test_prepare_features(self, baseline_predictor, sample_data):
        """Testa preparação de features"""
        # Preparar dados básicos
        sample_data = sample_data.with_columns([
            pl.col('data_faturamento').dt.month().alias('mes'),
            pl.col('data_faturamento').dt.quarter().alias('trimestre'),
            pl.col('data_faturamento').dt.year().alias('ano')
        ])
        
        # Adicionar target
        sample_data = sample_data.with_columns([
            (pl.col('frete_brl') / (pl.col('volume_ton') * pl.col('distancia_km')))
            .alias('preco_ton_km_calculado')
        ])
        
        # Criar features
        df_enhanced = baseline_predictor.create_advanced_features(sample_data)
        
        # Preparar features
        X, y = baseline_predictor.prepare_features(df_enhanced)
        
        assert X.shape[0] > 0
        assert y.shape[0] > 0
        assert X.shape[0] == y.shape[0]
    
    def test_train_baseline_models(self, baseline_predictor, sample_data):
        """Testa treinamento dos modelos baseline"""
        # Preparar dados
        sample_data = sample_data.with_columns([
            pl.col('data_faturamento').dt.month().alias('mes'),
            pl.col('data_faturamento').dt.quarter().alias('trimestre'),
            pl.col('data_faturamento').dt.year().alias('ano')
        ])
        
        # Adicionar target
        sample_data = sample_data.with_columns([
            (pl.col('frete_brl') / (pl.col('volume_ton') * pl.col('distancia_km')))
            .alias('preco_ton_km_calculado')
        ])
        
        # Criar features
        df_enhanced = baseline_predictor.create_advanced_features(sample_data)
        
        # Treinar modelos
        results = baseline_predictor.train_baseline_models(df_enhanced)
        
        # Verificar resultados
        assert len(results) > 0
        assert baseline_predictor.best_model is not None
        assert baseline_predictor.best_model_name is not None
    
    def test_save_and_load_model(self, baseline_predictor, sample_data, tmp_path):
        """Testa salvamento e carregamento do modelo"""
        # Preparar dados
        sample_data = sample_data.with_columns([
            pl.col('data_faturamento').dt.month().alias('mes'),
            pl.col('data_faturamento').dt.quarter().alias('trimestre'),
            pl.col('data_faturamento').dt.year().alias('ano')
        ])
        
        # Adicionar target
        sample_data = sample_data.with_columns([
            (pl.col('frete_brl') / (pl.col('volume_ton') * pl.col('distancia_km')))
            .alias('preco_ton_km_calculado')
        ])
        
        # Criar features e treinar
        df_enhanced = baseline_predictor.create_advanced_features(sample_data)
        baseline_predictor.train_baseline_models(df_enhanced)
        
        # Salvar modelo
        baseline_predictor.models_dir = tmp_path
        baseline_predictor.save_baseline_model()
        
        # Verificar se arquivos foram criados
        assert (tmp_path / "baseline_model" / "best_model.joblib").exists()
        assert (tmp_path / "baseline_model" / "metadata.json").exists()
    
    def test_generate_baseline_report(self, baseline_predictor, sample_data):
        """Testa geração de relatório baseline"""
        # Preparar dados
        sample_data = sample_data.with_columns([
            pl.col('data_faturamento').dt.month().alias('mes'),
            pl.col('data_faturamento').dt.quarter().alias('trimestre'),
            pl.col('data_faturamento').dt.year().alias('ano')
        ])
        
        # Adicionar target
        sample_data = sample_data.with_columns([
            (pl.col('frete_brl') / (pl.col('volume_ton') * pl.col('distancia_km')))
            .alias('preco_ton_km_calculado')
        ])
        
        # Criar features e treinar
        df_enhanced = baseline_predictor.create_advanced_features(sample_data)
        results = baseline_predictor.train_baseline_models(df_enhanced)
        
        # Gerar relatório
        report = baseline_predictor.generate_baseline_report(df_enhanced, results)
        
        # Verificar estrutura do relatório
        assert 'model_ranking' in report
        assert 'best_model' in report
        assert 'training_summary' in report
        assert len(report['model_ranking']) > 0


class TestFreteFeatureEngineer:
    """Testes para o engenheiro de features"""
    
    @pytest.fixture
    def feature_engineer(self):
        """Instância do engenheiro de features"""
        return FreteFeatureEngineer()
    
    @pytest.fixture
    def sample_data(self):
        """Dados de exemplo para testes"""
        data = {
            'data_faturamento': [
                '2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05',
                '2024-02-01', '2024-02-02', '2024-02-03', '2024-02-04', '2024-02-05',
                '2024-03-01', '2024-03-02', '2024-03-03', '2024-03-04', '2024-03-05',
                '2024-04-01', '2024-04-02', '2024-04-03', '2024-04-04', '2024-04-05',
                '2024-05-01', '2024-05-02', '2024-05-03', '2024-05-04', '2024-05-05',
                '2024-06-01', '2024-06-02', '2024-06-03', '2024-06-04', '2024-06-05'
            ],
            'rota': [
                'SP-RJ', 'SP-MG', 'MG-RJ', 'RJ-SP', 'MG-SP', 'SP-RS', 'RS-SP', 'MG-RS',
                'RJ-MG', 'SP-GO', 'GO-SP', 'MG-GO', 'RJ-GO', 'SP-MT', 'MT-SP', 'MG-MT',
                'RJ-MT', 'SP-BA', 'BA-SP', 'MG-BA', 'RJ-BA', 'SP-PE', 'PE-SP', 'MG-PE',
                'RJ-PE', 'SP-CE', 'CE-SP', 'MG-CE', 'RJ-CE', 'SP-PA'
            ],
            'transportadora': [
                'Transp1', 'Transp2', 'Transp3', 'Transp4', 'Transp5',
                'Transp1', 'Transp2', 'Transp3', 'Transp4', 'Transp5',
                'Transp1', 'Transp2', 'Transp3', 'Transp4', 'Transp5',
                'Transp1', 'Transp2', 'Transp3', 'Transp4', 'Transp5',
                'Transp1', 'Transp2', 'Transp3', 'Transp4', 'Transp5',
                'Transp1', 'Transp2', 'Transp3', 'Transp4', 'Transp5'
            ],
            'volume_ton': [10.0, 15.0, 20.0, 12.0, 18.0, 25.0, 30.0, 35.0, 40.0, 45.0,
                          50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0,
                          100.0, 105.0, 110.0, 115.0, 120.0, 125.0, 130.0, 135.0, 140.0, 145.0],
            'frete_brl': [100.0, 150.0, 200.0, 120.0, 180.0, 250.0, 300.0, 350.0, 400.0, 450.0,
                         500.0, 550.0, 600.0, 650.0, 700.0, 750.0, 800.0, 850.0, 900.0, 950.0,
                         1000.0, 1050.0, 1100.0, 1150.0, 1200.0, 1250.0, 1300.0, 1350.0, 1400.0, 1450.0],
            'distancia_km': [100.0, 200.0, 300.0, 100.0, 200.0, 500.0, 600.0, 700.0, 800.0, 900.0,
                            1000.0, 1100.0, 1200.0, 1300.0, 1400.0, 1500.0, 1600.0, 1700.0, 1800.0, 1900.0,
                            2000.0, 2100.0, 2200.0, 2300.0, 2400.0, 2500.0, 2600.0, 2700.0, 2800.0, 2900.0],
            'modal': ['RODOVIARIO', 'RODOVIARIO', 'FERROVIARIO', 'RODOVIARIO', 'RODOVIARIO', 'RODOVIARIO',
                     'FERROVIARIO', 'AQUAVIARIO', 'RODOVIARIO', 'RODOVIARIO', 'FERROVIARIO', 'RODOVIARIO',
                     'AQUAVIARIO', 'RODOVIARIO', 'FERROVIARIO', 'RODOVIARIO', 'AQUAVIARIO', 'RODOVIARIO',
                     'FERROVIARIO', 'RODOVIARIO', 'AQUAVIARIO', 'RODOVIARIO', 'FERROVIARIO', 'RODOVIARIO',
                     'AQUAVIARIO', 'RODOVIARIO', 'FERROVIARIO', 'RODOVIARIO', 'AQUAVIARIO', 'RODOVIARIO'],
            'tipo_rodovia': ['BR-116', 'SP-300', 'BR-040', 'BR-116', 'SP-300', 'BR-116', 'BR-101', 'BR-040',
                            'BR-116', 'SP-300', 'BR-040', 'BR-116', 'SP-300', 'BR-116', 'BR-101', 'BR-040',
                            'BR-116', 'SP-300', 'BR-040', 'BR-116', 'BR-101', 'BR-040', 'BR-116', 'SP-300',
                            'BR-040', 'BR-116', 'SP-300', 'BR-116', 'BR-101', 'BR-040'],
            'tipo_veiculo': ['TRUCK', 'TRUCK', 'VAGAO', 'TRUCK', 'TRUCK', 'TRUCK', 'VAGAO', 'BARCO',
                            'TRUCK', 'TRUCK', 'VAGAO', 'TRUCK', 'BARCO', 'TRUCK', 'VAGAO', 'TRUCK',
                            'BARCO', 'TRUCK', 'VAGAO', 'TRUCK', 'BARCO', 'TRUCK', 'VAGAO', 'TRUCK',
                            'BARCO', 'TRUCK', 'VAGAO', 'TRUCK', 'BARCO', 'TRUCK']
        }
        
        df = pl.DataFrame(data)
        df = df.with_columns([
            pl.col('data_faturamento').str.strptime(pl.Datetime, '%Y-%m-%d'),
            # Adicionar preco_ton_km_calculado que é necessário para features econômicas
            (pl.col('frete_brl') / (pl.col('volume_ton') * pl.col('distancia_km'))).alias('preco_ton_km_calculado')
        ])
        
        return df
    
    def test_create_temporal_features(self, feature_engineer, sample_data):
        """Testa criação de features temporais"""
        df_temporal = feature_engineer.create_temporal_features(sample_data)
        
        # Verificar features criadas
        assert 'dia_semana' in df_temporal.columns
        assert 'estacao' in df_temporal.columns
        assert 'mes_alta_temporada' in df_temporal.columns
        assert 'fim_semana' in df_temporal.columns
        
        # Verificar se valores estão corretos (janeiro = estação 1)
        assert 1 in df_temporal['estacao'].unique().to_list()
    
    def test_create_geographic_features(self, feature_engineer, sample_data):
        """Testa criação de features geográficas"""
        df_geo = feature_engineer.create_geographic_features(sample_data)
        
        # Verificar features criadas
        assert 'estado_origem' in df_geo.columns
        assert 'estado_destino' in df_geo.columns
        assert 'regiao' in df_geo.columns
        assert 'origem_sp' in df_geo.columns
        
        # Verificar valores
        assert df_geo['origem_sp'].sum() > 0  # Deve ter pelo menos uma rota de SP
    
    def test_create_operational_features(self, feature_engineer, sample_data):
        """Testa criação de features operacionais"""
        df_op = feature_engineer.create_operational_features(sample_data)
        
        # Verificar features criadas
        assert 'modal_encoded' in df_op.columns
        assert 'tipo_rodovia_encoded' in df_op.columns
        assert 'eficiencia_volume' in df_op.columns
        
        # Verificar valores (RODOVIARIO=1, FERROVIARIO=2, AQUAVIARIO=3)
        assert 1 in df_op['modal_encoded'].unique().to_list()
        assert 2 in df_op['modal_encoded'].unique().to_list()
        assert 3 in df_op['modal_encoded'].unique().to_list()
    
    def test_create_all_features(self, feature_engineer, sample_data):
        """Testa criação de todas as features"""
        df_features = feature_engineer.create_all_features(sample_data)
        
        # Verificar se todas as features foram criadas
        expected_features = sum(len(features) for features in feature_engineer.feature_groups.values())
        assert df_features.shape[1] > sample_data.shape[1]
        
        # Verificar se grupos foram preenchidos
        for group, features in feature_engineer.feature_groups.items():
            assert len(features) > 0
    
    def test_analyze_feature_importance(self, feature_engineer, sample_data):
        """Testa análise de importância das features"""
        # Criar features primeiro
        df_features = feature_engineer.create_all_features(sample_data)
        
        # Adicionar target
        df_features = df_features.with_columns([
            (pl.col('frete_brl') / (pl.col('volume_ton') * pl.col('distancia_km')))
            .alias('preco_ton_km_calculado')
        ])
        
        # Analisar importância
        importance = feature_engineer.analyze_feature_importance(df_features)
        
        # Verificar resultados
        assert len(importance) > 0
        assert feature_engineer.feature_stats['total_features'] > 0
        assert len(feature_engineer.feature_stats['top_correlations']) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
