"""
Modelo de Machine Learning para Previsão de Preços de Frete
Fase 1: Modelo baseline usando Random Forest
"""

import polars as pl
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import logging
from pathlib import Path
from typing import Tuple, Dict, Any
import os
import pickle
import pandas as pd

logger = logging.getLogger(__name__)


class FretePricePredictor:
    """
    Modelo de ML para prever preços de frete baseado em características da rota
    """
    
    def __init__(self, model_path: str = "models/frete_predictor.joblib"):
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.model_path = Path(model_path)
        self.model_path.parent.mkdir(exist_ok=True)
        
        # Features para o modelo
        self.numeric_features = [
            'volume_ton', 'distancia_km', 'mes', 'trimestre', 'ano'
        ]
        
        self.categorical_features = [
            'modal', 'tipo_rodovia', 'tipo_veiculo'
        ]
        
        self.target_feature = 'preco_ton_km_calculado'
    
    def prepare_features(self, df: pl.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepara features para treinamento do modelo
        """
        logger.info("Preparando features para o modelo...")
        
        # Converter para pandas para compatibilidade com sklearn
        df_pandas = df.to_pandas()
        
        # Features numéricas
        X_numeric = df_pandas[self.numeric_features].values
        
        # Features categóricas (encoding)
        X_categorical = np.zeros((len(df_pandas), len(self.categorical_features)))
        
        for i, feature in enumerate(self.categorical_features):
            if feature not in self.label_encoders:
                self.label_encoders[feature] = LabelEncoder()
                X_categorical[:, i] = self.label_encoders[feature].fit_transform(
                    df_pandas[feature].fillna('UNKNOWN')
                )
            else:
                # Para dados de teste, usar transform existente
                X_categorical[:, i] = self.label_encoders[feature].transform(
                    df_pandas[feature].fillna('UNKNOWN')
                )
        
        # Combinar features
        X = np.hstack([X_numeric, X_categorical])
        
        # Target
        y = df_pandas[self.target_feature].values
        
        # Remover linhas com valores NaN
        mask = ~np.isnan(y)
        X = X[mask]
        y = y[mask]
        
        logger.info(f"Features preparadas. X shape: {X.shape}, y shape: {y.shape}")
        return X, y
    
    def train(self, df: pl.DataFrame) -> Dict[str, float]:
        """
        Treina o modelo de previsão de preços
        """
        logger.info("Iniciando treinamento do modelo...")
        
        # Preparar features
        X, y = self.prepare_features(df)
        
        # Split treino/teste
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Normalizar features numéricas
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Treinar modelo
        self.model.fit(X_train_scaled, y_train)
        
        # Fazer previsões
        y_pred_train = self.model.predict(X_train_scaled)
        y_pred_test = self.model.predict(X_test_scaled)
        
        # Calcular métricas
        metrics = {
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'train_r2': r2_score(y_train, y_pred_train),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'test_r2': r2_score(y_test, y_pred_test)
        }
        
        # Cross-validation
        cv_scores = cross_val_score(
            self.model, X_train_scaled, y_train, 
            cv=5, scoring='r2'
        )
        metrics['cv_r2_mean'] = cv_scores.mean()
        metrics['cv_r2_std'] = cv_scores.std()
        
        logger.info("Modelo treinado com sucesso!")
        logger.info(f"Métricas de treino - MAE: {metrics['train_mae']:.4f}, R²: {metrics['train_r2']:.4f}")
        logger.info(f"Métricas de teste - MAE: {metrics['test_mae']:.4f}, R²: {metrics['test_r2']:.4f}")
        
        return metrics
    
    def predict(self, df: pl.DataFrame) -> np.ndarray:
        """
        Faz previsões para novos dados
        """
        X, _ = self.prepare_features(df)
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Retorna importância das features
        """
        feature_names = self.numeric_features + self.categorical_features
        importance = self.model.feature_importances_
        
        return dict(zip(feature_names, importance))
    
    def save_model(self) -> None:
        """
        Salva o modelo treinado em múltiplos formatos para compatibilidade
        """
        
        # Criar diretório se não existir
        try:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            models_dir = os.path.join(project_root, "models")
        except NameError:
            models_dir = "../../models"
        
        os.makedirs(models_dir, exist_ok=True)
        
        # 1. Salvar no formato joblib (padrão scikit-learn)
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.numeric_features + self.categorical_features
        }
        
        joblib_path = os.path.join(models_dir, "frete_predictor.joblib")
        joblib.dump(model_data, joblib_path)
        logger.info(f"Modelo salvo em formato joblib: {joblib_path}")
        
        # 2. Salvar no formato pickle (compatível com ML Studio)
        pickle_path = os.path.join(models_dir, "frete_predictor.pkl")
        with open(pickle_path, 'wb') as f:
            pickle.dump(model_data, f)
        logger.info(f"Modelo salvo em formato pickle: {pickle_path}")
        
        # 3. Salvar modelo ONNX (formato universal para ML)
        try:
            import onnxmltools
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType
            
            # Preparar dados de exemplo para conversão
            X_sample = np.zeros((1, len(self.numeric_features) + len(self.categorical_features)))
            X_sample_scaled = self.scaler.transform(X_sample)
            
            # Converter para ONNX
            initial_type = [('float_input', FloatTensorType([None, X_sample_scaled.shape[1]]))]
            onx = convert_sklearn(self.model, initial_types=initial_type)
            
            onnx_path = os.path.join(models_dir, "frete_predictor.onnx")
            with open(onnx_path, "wb") as f:
                f.write(onx.SerializeToString())
            logger.info(f"Modelo salvo em formato ONNX: {onnx_path}")
            
        except ImportError:
            logger.info("ONNX não disponível. Instale: pip install onnxmltools skl2onnx")
        
        # 4. Salvar modelo PMML (formato XML para ML Studio)
        try:
            from sklearn2pmml import sklearn2pmml
            from sklearn2pmml.pipeline import PMMLPipeline
            
            # Criar pipeline PMML
            pipeline = PMMLPipeline([
                ("scaler", self.scaler),
                ("regressor", self.model)
            ])
            
            pmml_path = os.path.join(models_dir, "frete_predictor.pmml")
            sklearn2pmml(pipeline, pmml_path)
            logger.info(f"Modelo salvo em formato PMML: {pmml_path}")
            
        except ImportError:
            logger.info("PMML não disponível. Instale: pip install sklearn2pmml")
        
        # 5. Salvar metadados do modelo para ML Studio
        metadata = {
            'model_type': 'RandomForestRegressor',
            'features': {
                'numeric': self.numeric_features,
                'categorical': self.categorical_features
            },
            'target': self.target_feature,
            'model_params': self.model.get_params(),
            'feature_importance': self.get_feature_importance(),
            'training_date': str(pd.Timestamp.now()),
            'version': '1.0'
        }
        
        metadata_path = os.path.join(models_dir, "model_metadata.json")
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info(f"Metadados salvos em: {metadata_path}")
        
        # 6. Salvar modelo em formato nativo do ML Studio (Azure)
        try:
            # Salvar como arquivo de modelo do Azure ML
            azure_model_path = os.path.join(models_dir, "azure_ml_model")
            os.makedirs(azure_model_path, exist_ok=True)
            
            # Salvar modelo principal
            azure_model_file = os.path.join(azure_model_path, "model.pkl")
            with open(azure_model_file, 'wb') as f:
                pickle.dump(self.model, f)
            
            # Salvar scaler
            azure_scaler_file = os.path.join(azure_model_path, "scaler.pkl")
            with open(azure_scaler_file, 'wb') as f:
                pickle.dump(self.scaler, f)
            
            # Salvar encoders
            azure_encoders_file = os.path.join(azure_model_path, "encoders.pkl")
            with open(azure_encoders_file, 'wb') as f:
                pickle.dump(self.label_encoders, f)
            
            # Salvar configuração
            config = {
                'numeric_features': self.numeric_features,
                'categorical_features': self.categorical_features,
                'target_feature': self.target_feature
            }
            
            config_file = os.path.join(azure_model_path, "config.json")
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"Modelo salvo para Azure ML em: {azure_model_path}")
            
        except Exception as e:
            logger.warning(f"Erro ao salvar para Azure ML: {e}")
        
        logger.info("Modelo salvo em múltiplos formatos para compatibilidade com diferentes plataformas!")
    
    def load_model(self) -> None:
        """
        Carrega modelo salvo
        """
        if self.model_path.exists():
            model_data = joblib.load(self.model_path)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.label_encoders = model_data['label_encoders']
            logger.info(f"Modelo carregado de: {self.model_path}")
        else:
            logger.warning(f"Modelo não encontrado em: {self.model_path}")
    
    def analyze_predictions(self, df: pl.DataFrame, predictions: np.ndarray) -> pl.DataFrame:
        """
        Analisa previsões e identifica oportunidades de otimização
        """
        # Adicionar previsões ao DataFrame
        df_with_pred = df.with_columns([
            pl.Series("preco_previsto", predictions),
            (pl.col("preco_ton_km_calculado") - pl.Series("preco_previsto", predictions))
            .alias("diferenca_preco"),
            ((pl.col("preco_ton_km_calculado") - pl.Series("preco_previsto", predictions)) / 
             pl.col("preco_ton_km_calculado") * 100).alias("percentual_diferenca")
        ])
        
        # Identificar oportunidades de economia
        oportunidades = df_with_pred.filter(
            pl.col("percentual_diferenca") > 10  # Preço 10% acima do previsto
        ).sort("percentual_diferenca", descending=True)
        
        return oportunidades
    
    def generate_optimization_report(self, df: pl.DataFrame) -> Dict[str, Any]:
        """
        Gera relatório de otimização de custos
        """
        predictions = self.predict(df)
        oportunidades = self.analyze_predictions(df, predictions)
        
        # Calcular economia potencial
        economia_potencial = oportunidades.select([
            (pl.col("diferenca_preco") * pl.col("volume_ton") * pl.col("distancia_km"))
            .alias("economia_por_entrega")
        ]).sum().item()
        
        # Top rotas para otimização
        top_rotas = oportunidades.group_by("rota").agg([
            pl.col("percentual_diferenca").mean().alias("sobrepreco_medio"),
            pl.col("volume_ton").sum().alias("volume_total"),
            pl.len().alias("num_entregas")
        ]).sort("sobrepreco_medio", descending=True).head(10)
        
        # Top transportadoras para negociação
        top_transportadoras = oportunidades.group_by("transportadora").agg([
            pl.col("percentual_diferenca").mean().alias("sobrepreco_medio"),
            pl.col("volume_ton").sum().alias("volume_total"),
            pl.len().alias("num_entregas")
        ]).sort("sobrepreco_medio", descending=True).head(10)
        
        return {
            'economia_potencial_total': economia_potencial,
            'num_oportunidades': len(oportunidades),
            'top_rotas_otimizacao': top_rotas.to_pandas().to_dict('records'),
            'top_transportadoras_negociacao': top_transportadoras.to_pandas().to_dict('records'),
            'metricas_modelo': {
                'feature_importance': self.get_feature_importance()
            }
        }
