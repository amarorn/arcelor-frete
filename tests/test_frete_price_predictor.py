import pytest
import polars as pl
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
from utils.ml.frete_price_predictor import FretePricePredictor


def prepare_test_data(df):
    return df.rename({
        '02.01.00 - Volume (ton)': 'volume_ton',
        '02.01.02 - DISTANCIA (KM)': 'distancia_km',
        '00.nm_modal': 'modal',
        'nm_tipo_rodovia': 'tipo_rodovia',
        'nm_veiculo': 'tipo_veiculo',
        '02.03.02 - Preço_Frete Geral (BRL / TON / KM)': 'preco_ton_km_calculado',
        '01.Rota_MuniOrigem_MuniDestino': 'rota',
        'nm_transportadora_aux': 'transportadora'
    }).with_columns([
        pl.col('00.dt_doc_faturamento').dt.month().alias('mes'),
        pl.col('00.dt_doc_faturamento').dt.quarter().alias('trimestre'),
        pl.col('00.dt_doc_faturamento').dt.year().alias('ano')
    ])


class TestFretePricePredictor:
    
    def test_init(self, temp_model_dir):
        model_path = os.path.join(temp_model_dir, "test_model.joblib")
        predictor = FretePricePredictor(model_path)
        
        assert predictor.model is not None
        assert predictor.scaler is not None
        assert predictor.label_encoders == {}
        assert predictor.model_path == Path(model_path)
        assert os.path.exists(temp_model_dir)
    
    def test_feature_configuration(self, temp_model_dir):
        predictor = FretePricePredictor()
        
        expected_numeric = ['volume_ton', 'distancia_km', 'mes', 'trimestre', 'ano']
        expected_categorical = ['modal', 'tipo_rodovia', 'tipo_veiculo']
        
        assert predictor.numeric_features == expected_numeric
        assert predictor.categorical_features == expected_categorical
        assert predictor.target_feature == 'preco_ton_km_calculado'
    
    def test_prepare_features_success(self, sample_polars_df):
        predictor = FretePricePredictor()
        df_processed = prepare_test_data(sample_polars_df)
        
        X, y = predictor.prepare_features(df_processed)
        
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert X.shape[0] == y.shape[0]
        assert X.shape[1] == len(predictor.numeric_features) + len(predictor.categorical_features)
        assert not np.isnan(X).any()
        assert not np.isnan(y).any()
    
    def test_prepare_features_with_nan_values(self, sample_polars_df):
        predictor = FretePricePredictor()
        df_processed = prepare_test_data(sample_polars_df)
        
        # Usar valor numérico inválido em vez de None para testar filtro
        df_with_invalid = df_processed.with_columns([
            pl.lit(-999.0).alias('preco_ton_km_calculado')
        ])
        
        X, y = predictor.prepare_features(df_with_invalid)
        
        # Verificar se os dados foram processados corretamente
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
    
    def test_train_success(self, sample_polars_df):
        predictor = FretePricePredictor()
        df_processed = prepare_test_data(sample_polars_df)
        
        metrics = predictor.train(df_processed)
        
        assert hasattr(predictor.model, 'feature_importances_')
        assert hasattr(predictor.scaler, 'mean_')
        
        required_metrics = [
            'train_mae', 'train_rmse', 'train_r2',
            'test_mae', 'test_rmse', 'test_r2',
            'cv_r2_mean', 'cv_r2_std'
        ]
        
        for metric in required_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))
    
    def test_train_with_insufficient_data(self):
        predictor = FretePricePredictor()
        
        small_df = pl.DataFrame({
            'volume_ton': [10, 20, 15, 25, 30, 35, 40, 45, 50, 55],
            'distancia_km': [100, 200, 150, 250, 300, 350, 400, 450, 500, 550],
            'modal': ['RODOVIARIO'] * 10,
            'tipo_rodovia': ['PEDAGIO', 'LIVRE'] * 5,
            'tipo_veiculo': ['TRUCK', 'CARRETA'] * 5,
            'mes': list(range(1, 11)),
            'trimestre': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4],
            'ano': [2024] * 10,
            'preco_ton_km_calculado': [0.5, 0.6, 0.55, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        })
        
        metrics = predictor.train(small_df)
        assert 'train_r2' in metrics
    
    def test_predict_success(self, sample_polars_df):
        predictor = FretePricePredictor()
        df_processed = prepare_test_data(sample_polars_df)
        
        predictor.train(df_processed)
        predictions = predictor.predict(df_processed)
        
        assert isinstance(predictions, np.ndarray)
        assert predictions.shape[0] == df_processed.shape[0]
        assert not np.isnan(predictions).any()
    
    def test_predict_without_training(self, sample_polars_df):
        predictor = FretePricePredictor()
        df_processed = sample_polars_df
        
        with pytest.raises(Exception):
            predictor.predict(df_processed)
    
    def test_get_feature_importance(self, sample_polars_df):
        predictor = FretePricePredictor()
        df_processed = prepare_test_data(sample_polars_df)
        
        predictor.train(df_processed)
        importance = predictor.get_feature_importance()
        
        assert isinstance(importance, dict)
        assert len(importance) == len(predictor.numeric_features) + len(predictor.categorical_features)
        
        all_features = predictor.numeric_features + predictor.categorical_features
        for feature in all_features:
            assert feature in importance
            assert isinstance(importance[feature], (int, float))
            assert importance[feature] >= 0
    
    def test_save_and_load_model(self, sample_polars_df, temp_model_dir):
        """Testa salvamento e carregamento do modelo"""
        # Usar caminho temporário para teste
        model_path = os.path.join(temp_model_dir, "test_model.joblib")
        predictor = FretePricePredictor(model_path)
        df_processed = prepare_test_data(sample_polars_df)
        
        predictor.train(df_processed)
        predictor.save_model()
        
        # Verificar se os arquivos foram criados na pasta models/
        models_dir = "models"
        assert os.path.exists(models_dir)
        
        # Verificar se pelo menos o arquivo pickle foi criado
        pickle_path = os.path.join(models_dir, "frete_predictor.pkl")
        assert os.path.exists(pickle_path)
        assert os.path.getsize(pickle_path) > 0
        
        # Verificar se o arquivo joblib foi criado
        joblib_path = os.path.join(models_dir, "frete_predictor.joblib")
        assert os.path.exists(joblib_path)
        assert os.path.getsize(joblib_path) > 0
        
        # Testar carregamento do modelo salvo
        new_predictor = FretePricePredictor()
        new_predictor.load_model()
        
        assert hasattr(new_predictor.model, 'feature_importances_')
        assert hasattr(new_predictor.scaler, 'mean_')
        
        predictions = new_predictor.predict(df_processed)
        assert isinstance(predictions, np.ndarray)
    
    def test_load_nonexistent_model(self, temp_model_dir):
        model_path = os.path.join(temp_model_dir, "nonexistent.joblib")
        predictor = FretePricePredictor(model_path)
        
        predictor.load_model()
    
    def test_analyze_predictions(self, sample_polars_df):
        predictor = FretePricePredictor()
        df_processed = prepare_test_data(sample_polars_df)
        
        predictor.train(df_processed)
        predictions = predictor.predict(df_processed)
        
        oportunidades = predictor.analyze_predictions(df_processed, predictions)
        
        assert isinstance(oportunidades, pl.DataFrame)
        assert 'preco_previsto' in oportunidades.columns
        assert 'diferenca_preco' in oportunidades.columns
        assert 'percentual_diferenca' in oportunidades.columns
        
        if oportunidades.shape[0] > 0:
            assert oportunidades.filter(pl.col('percentual_diferenca') <= 10).shape[0] == 0
    
    def test_generate_optimization_report(self, sample_polars_df):
        predictor = FretePricePredictor()
        df_processed = prepare_test_data(sample_polars_df)
        
        predictor.train(df_processed)
        report = predictor.generate_optimization_report(df_processed)
        
        assert isinstance(report, dict)
        assert 'economia_potencial_total' in report
        assert 'num_oportunidades' in report
        assert 'top_rotas_otimizacao' in report
        assert 'top_transportadoras_negociacao' in report
        assert 'metricas_modelo' in report
        
        assert isinstance(report['economia_potencial_total'], (int, float))
        assert isinstance(report['num_oportunidades'], int)
        assert isinstance(report['top_rotas_otimizacao'], list)
        assert isinstance(report['top_transportadoras_negociacao'], list)
        assert isinstance(report['metricas_modelo'], dict)
    
    def test_model_performance_metrics(self, sample_polars_df):
        predictor = FretePricePredictor()
        df_processed = prepare_test_data(sample_polars_df)
        
        metrics = predictor.train(df_processed)
        
        assert 0 <= metrics['train_r2'] <= 1
        assert 0 <= metrics['test_r2'] <= 1
        assert metrics['train_mae'] >= 0
        assert metrics['test_mae'] >= 0
        assert metrics['train_rmse'] >= 0
        assert metrics['test_rmse'] >= 0
        assert metrics['cv_r2_std'] >= 0
    
    def test_categorical_encoding_consistency(self, sample_polars_df):
        predictor = FretePricePredictor()
        df_processed = prepare_test_data(sample_polars_df)
        
        predictor.train(df_processed)
        
        predictions1 = predictor.predict(df_processed)
        predictions2 = predictor.predict(df_processed)
        
        np.testing.assert_array_almost_equal(predictions1, predictions2)
    
    def test_handle_missing_categorical_values(self, sample_polars_df):
        predictor = FretePricePredictor()
        df_processed = prepare_test_data(sample_polars_df)
        
        df_with_none = df_processed.with_columns([
            pl.lit(None).alias('modal'),
            pl.lit(None).alias('tipo_rodovia'),
            pl.lit(None).alias('tipo_veiculo')
        ])
        
        X, y = predictor.prepare_features(df_with_none)
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
