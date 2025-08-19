"""
Testes unitários para o modelo de seleção de transportadora
"""

import pytest
import polars as pl
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
from utils.ml.transportadora_selector import TransportadoraSelector


def prepare_test_data_for_selector(df):
    """Helper para preparar dados de teste para o seletor"""
    return df.rename({
        '02.01.00 - Volume (ton)': 'volume_ton',
        '02.01.02 - DISTANCIA (KM)': 'distancia_km',
        '00.nm_modal': 'modal',
        'nm_tipo_rodovia': 'tipo_rodovia',
        'nm_veiculo': 'tipo_veiculo',
        '02.03.02 - Preço_Frete Geral (BRL / TON / KM)': 'preco_ton_km',
        '01.Rota_MuniOrigem_MuniDestino': 'rota',
        'nm_transportadora_aux': 'transportadora',
        '02.01.01 - Frete Geral (BRL)': 'frete_brl'
    }).with_columns([
        pl.col('00.dt_doc_faturamento').dt.month().alias('mes'),
        pl.col('00.dt_doc_faturamento').dt.quarter().alias('trimestre'),
        pl.col('00.dt_doc_faturamento').dt.year().alias('ano'),
        pl.col('frete_brl').alias('preco_total'),
        pl.col('volume_ton').alias('volume_historico'),
        pl.col('preco_ton_km').alias('preco_ton_km'),
        pl.col('distancia_km').alias('distancia_km')
    ])


class TestTransportadoraSelector:
    
    def test_init(self, temp_model_dir):
        """Testa inicialização do seletor"""
        model_path = os.path.join(temp_model_dir, "test_selector.joblib")
        selector = TransportadoraSelector(model_path)
        
        assert selector.model is not None
        assert selector.scaler is not None
        assert selector.label_encoders == {}
        assert selector.model_path == Path(model_path)
        assert len(selector.numeric_features) == 12
        assert len(selector.categorical_features) == 5
        assert len(selector.target_features) == 3
    
    def test_criterios_avaliacao(self):
        """Testa configuração dos critérios de avaliação"""
        selector = TransportadoraSelector()
        
        expected_criterios = ['preco', 'performance', 'confiabilidade', 'capacidade', 'custo_beneficio']
        assert all(criterio in selector.criterios_avaliacao for criterio in expected_criterios)
        
        # Verificar se os pesos somam 1.0
        total_weight = sum(selector.criterios_avaliacao.values())
        assert abs(total_weight - 1.0) < 1e-6
    
    def test_calculate_performance_scores(self, sample_polars_df):
        """Testa cálculo de scores de performance"""
        selector = TransportadoraSelector()
        df_processed = prepare_test_data_for_selector(sample_polars_df)
        
        # Adicionar preco_ton_km se não existir
        if 'preco_ton_km' not in df_processed.columns:
            df_processed = df_processed.with_columns([
                (pl.col('frete_brl') / (pl.col('volume_ton') * pl.col('distancia_km'))).alias('preco_ton_km')
            ])
        
        performance_scores = selector.calculate_performance_scores(df_processed)
        
        assert isinstance(performance_scores, pl.DataFrame)
        assert 'performance_score' in performance_scores.columns
        assert 'score_preco' in performance_scores.columns
        assert 'score_volume' in performance_scores.columns
        
        # Verificar se os scores estão normalizados (0-1)
        assert performance_scores['performance_score'].min() >= 0
        assert performance_scores['performance_score'].max() <= 1
    
    def test_calculate_reliability_scores(self, sample_polars_df):
        """Testa cálculo de scores de confiabilidade"""
        selector = TransportadoraSelector()
        df_processed = prepare_test_data_for_selector(sample_polars_df)
        
        # Adicionar preco_ton_km se não existir
        if 'preco_ton_km' not in df_processed.columns:
            df_processed = df_processed.with_columns([
                (pl.col('frete_brl') / (pl.col('volume_ton') * pl.col('distancia_km'))).alias('preco_ton_km')
            ])
        
        reliability_scores = selector.calculate_reliability_scores(df_processed)
        
        assert isinstance(reliability_scores, pl.DataFrame)
        assert 'confiabilidade_score' in reliability_scores.columns
        assert 'cv_preco' in reliability_scores.columns
        
        # Verificar se os scores estão normalizados (0-1)
        assert reliability_scores['confiabilidade_score'].min() >= 0
        assert reliability_scores['confiabilidade_score'].max() <= 1
    
    def test_calculate_capacity_scores(self, sample_polars_df):
        """Testa cálculo de scores de capacidade"""
        selector = TransportadoraSelector()
        df_processed = prepare_test_data_for_selector(sample_polars_df)
        
        capacity_scores = selector.calculate_capacity_scores(df_processed)
        
        assert isinstance(capacity_scores, pl.DataFrame)
        assert 'capacidade_score' in capacity_scores.columns
        assert 'score_volume_total' in capacity_scores.columns
        assert 'score_distancia' in capacity_scores.columns
        
        # Verificar se os scores estão normalizados (0-1)
        assert capacity_scores['capacidade_score'].min() >= 0
        assert capacity_scores['capacidade_score'].max() <= 1
    
    def test_calculate_cost_benefit_scores(self, sample_polars_df):
        """Testa cálculo de scores de custo-benefício"""
        selector = TransportadoraSelector()
        df_processed = prepare_test_data_for_selector(sample_polars_df)
        
        # Adicionar preco_ton_km se não existir
        if 'preco_ton_km' not in df_processed.columns:
            df_processed = df_processed.with_columns([
                (pl.col('frete_brl') / (pl.col('volume_ton') * pl.col('distancia_km'))).alias('preco_ton_km')
            ])
        
        cost_benefit_scores = selector.calculate_cost_benefit_scores(df_processed)
        
        assert isinstance(cost_benefit_scores, pl.DataFrame)
        assert 'custo_beneficio_score' in cost_benefit_scores.columns
        assert 'custo_total_km' in cost_benefit_scores.columns
        
        # Verificar se os scores estão normalizados (0-1)
        assert cost_benefit_scores['custo_beneficio_score'].min() >= 0
        assert cost_benefit_scores['custo_beneficio_score'].max() <= 1
    
    def test_generate_training_data(self, sample_polars_df):
        """Testa geração de dados de treinamento"""
        selector = TransportadoraSelector()
        df_processed = prepare_test_data_for_selector(sample_polars_df)
        
        # Adicionar preco_ton_km se não existir
        if 'preco_ton_km' not in df_processed.columns:
            df_processed = df_processed.with_columns([
                (pl.col('frete_brl') / (pl.col('volume_ton') * pl.col('distancia_km'))).alias('preco_ton_km')
            ])
        
        df_training = selector.generate_training_data(df_processed)
        
        assert isinstance(df_training, pl.DataFrame)
        assert 'transportadora_recomendada' in df_training.columns
        assert 'score_recomendacao' in df_training.columns
        assert 'categoria_prioridade' in df_training.columns
        
        # Verificar se as categorias estão corretas
        categorias = df_training['categoria_prioridade'].unique().to_list()
        assert all(cat in ['ALTA', 'MÉDIA', 'BAIXA'] for cat in categorias)
    
    def test_train_success(self, sample_polars_df):
        """Testa treinamento bem-sucedido"""
        selector = TransportadoraSelector()
        df_processed = prepare_test_data_for_selector(sample_polars_df)
        
        # Adicionar preco_ton_km se não existir
        if 'preco_ton_km' not in df_processed.columns:
            df_processed = df_processed.with_columns([
                (pl.col('frete_brl') / (pl.col('volume_ton') * pl.col('distancia_km'))).alias('preco_ton_km')
            ])
        
        metrics = selector.train(df_processed)
        
        assert hasattr(selector.model, 'feature_importances_')
        assert hasattr(selector.scaler, 'mean_')
        
        required_metrics = [
            'train_accuracy', 'test_accuracy', 'cv_accuracy_mean', 'cv_accuracy_std'
        ]
        
        for metric in required_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))
            assert metrics[metric] >= 0
    
    def test_predict_transportadora(self, sample_polars_df):
        """Testa previsões de transportadora"""
        selector = TransportadoraSelector()
        df_processed = prepare_test_data_for_selector(sample_polars_df)
        
        # Adicionar preco_ton_km se não existir
        if 'preco_ton_km' not in df_processed.columns:
            df_processed = df_processed.with_columns([
                (pl.col('frete_brl') / (pl.col('volume_ton') * pl.col('distancia_km'))).alias('preco_ton_km')
            ])
        
        # Treinar modelo primeiro
        selector.train(df_processed)
        
        # Fazer previsões
        df_with_predictions = selector.predict_transportadora(df_processed)
        
        assert isinstance(df_with_predictions, pl.DataFrame)
        assert 'transportadora_recomendada_ml' in df_with_predictions.columns
        assert 'score_confianca_ml' in df_with_predictions.columns
        
        # Verificar se os scores de confiança estão entre 0 e 1
        confidence_scores = df_with_predictions['score_confianca_ml'].to_list()
        assert all(0 <= score <= 1 for score in confidence_scores)
    
    def test_get_feature_importance(self, sample_polars_df):
        """Testa obtenção de importância das features"""
        selector = TransportadoraSelector()
        df_processed = prepare_test_data_for_selector(sample_polars_df)
        
        # Adicionar preco_ton_km se não existir
        if 'preco_ton_km' not in df_processed.columns:
            df_processed = df_processed.with_columns([
                (pl.col('frete_brl') / (pl.col('volume_ton') * pl.col('distancia_km'))).alias('preco_ton_km')
            ])
        
        # Treinar modelo primeiro
        selector.train(df_processed)
        
        importance = selector.get_feature_importance()
        
        assert isinstance(importance, dict)
        expected_features = selector.numeric_features + selector.categorical_features
        
        for feature in expected_features:
            assert feature in importance
            assert isinstance(importance[feature], (int, float))
            assert importance[feature] >= 0
    
    def test_save_and_load_model(self, sample_polars_df, temp_model_dir):
        """Testa salvamento e carregamento do modelo"""
        selector = TransportadoraSelector()
        df_processed = prepare_test_data_for_selector(sample_polars_df)
        
        # Adicionar preco_ton_km se não existir
        if 'preco_ton_km' not in df_processed.columns:
            df_processed = df_processed.with_columns([
                (pl.col('frete_brl') / (pl.col('volume_ton') * pl.col('distancia_km'))).alias('preco_ton_km')
            ])
        
        # Treinar modelo
        selector.train(df_processed)
        
        # Salvar modelo
        selector.save_model()
        
        # Verificar se os arquivos foram criados
        models_dir = "models"
        assert os.path.exists(models_dir)
        
        # Verificar arquivos principais
        joblib_path = os.path.join(models_dir, "transportadora_selector.joblib")
        pickle_path = os.path.join(models_dir, "transportadora_selector.pkl")
        metadata_path = os.path.join(models_dir, "transportadora_selector_metadata.json")
        
        assert os.path.exists(joblib_path)
        assert os.path.exists(pickle_path)
        assert os.path.exists(metadata_path)
        
        # Testar carregamento
        new_selector = TransportadoraSelector()
        new_selector.load_model()
        
        assert hasattr(new_selector.model, 'feature_importances_')
        assert hasattr(new_selector.scaler, 'mean_')
    
    def test_generate_selection_report(self, sample_polars_df):
        """Testa geração de relatório de seleção"""
        selector = TransportadoraSelector()
        df_processed = prepare_test_data_for_selector(sample_polars_df)
        
        # Adicionar preco_ton_km se não existir
        if 'preco_ton_km' not in df_processed.columns:
            df_processed = df_processed.with_columns([
                (pl.col('frete_brl') / (pl.col('volume_ton') * pl.col('distancia_km'))).alias('preco_ton_km')
            ])
        
        # Treinar modelo primeiro
        selector.train(df_processed)
        
        # Gerar relatório
        report = selector.generate_selection_report(df_processed)
        
        assert isinstance(report, dict)
        assert 'resumo_geral' in report
        assert 'analise_por_transportadora' in report
        assert 'analise_por_rota' in report
        assert 'metricas_modelo' in report
        
        # Verificar estrutura do resumo
        resumo = report['resumo_geral']
        assert 'total_recomendacoes' in resumo
        assert 'score_confianca_medio' in resumo
        assert 'transportadora_mais_recomendada' in resumo
        
        # Verificar métricas do modelo
        metricas = report['metricas_modelo']
        assert 'feature_importance' in metricas
        assert 'criterios_avaliacao' in metricas
    
    def test_handle_missing_data(self, sample_polars_df):
        """Testa tratamento de dados faltantes"""
        selector = TransportadoraSelector()
        df_processed = prepare_test_data_for_selector(sample_polars_df)
        
        # Adicionar preco_ton_km se não existir
        if 'preco_ton_km' not in df_processed.columns:
            df_processed = df_processed.with_columns([
                (pl.col('frete_brl') / (pl.col('volume_ton') * pl.col('distancia_km'))).alias('preco_ton_km')
            ])
        
        # Adicionar dados faltantes
        df_with_missing = df_processed.with_columns([
            pl.lit(None).alias('modal'),
            pl.lit(None).alias('tipo_rodovia'),
            pl.lit(None).alias('tipo_veiculo')
        ])
        
        # Deve funcionar sem erro
        try:
            df_training = selector.generate_training_data(df_with_missing)
            assert isinstance(df_training, pl.DataFrame)
        except Exception as e:
            pytest.fail(f"Falhou ao processar dados com valores faltantes: {e}")
    
    def test_model_performance_metrics(self, sample_polars_df):
        """Testa métricas de performance do modelo"""
        selector = TransportadoraSelector()
        df_processed = prepare_test_data_for_selector(sample_polars_df)
        
        # Adicionar preco_ton_km se não existir
        if 'preco_ton_km' not in df_processed.columns:
            df_processed = df_processed.with_columns([
                (pl.col('frete_brl') / (pl.col('volume_ton') * pl.col('distancia_km'))).alias('preco_ton_km')
            ])
        
        metrics = selector.train(df_processed)
        
        # Verificar métricas de acurácia
        assert 0 <= metrics['train_accuracy'] <= 1
        assert 0 <= metrics['test_accuracy'] <= 1
        assert 0 <= metrics['cv_accuracy_mean'] <= 1
        assert metrics['cv_accuracy_std'] >= 0
        
        # Verificar relatórios de classificação
        assert 'train_classification_report' in metrics
        assert 'test_classification_report' in metrics
