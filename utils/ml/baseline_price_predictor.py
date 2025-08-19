
# Modelo de ML que roda vários algoritmos de previsão de preços e escolhe o melhor
# Compara 8 algoritmos diferentes: linear, random forest, gradient boosting, etc.
# Calcula várias métricas pra ver qual é o mais preciso

import polars as pl
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import joblib
import logging
from pathlib import Path
from typing import Tuple, Dict, Any, List
import os
import json
from datetime import datetime

from .feature_engineering import FreteFeatureEngineer

logger = logging.getLogger(__name__)

# Função auxiliar pra converter strings em números (encode) de jeito manual
# porque o LabelEncoder do sklearn às vezes da problema com o pipeline
def encode_categorical(X):
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    encoded = np.zeros_like(X, dtype=int)
    for i in range(X.shape[1]):
        unique_values = np.unique(X[:, i])
        value_to_int = {val: idx for idx, val in enumerate(unique_values)}
        encoded[:, i] = np.array([value_to_int.get(val, 0) for val in X[:, i]])
    
    return encoded


class BaselinePricePredictor:
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # Lista das features que vão ser usadas pros modelos
        # Números: volume, distância, mês, etc.
        self.numeric_features = [
            'volume_ton', 'distancia_km', 'mes', 'trimestre', 'ano'
        ]
        
        # Textos: modal (rodoviário/ferroviário), tipo de rodovia, etc.
        self.categorical_features = [
            'modal', 'tipo_rodovia', 'tipo_veiculo', 'rota', 'transportadora'
        ]
        
        # O que queremos prever: preço por tonelada por km
        self.target_feature = 'preco_ton_km_calculado'
        
        # Aqui tá a galera toda dos algoritmos que vamos testar
        # Cada um tem sua personalidade e jeito de aprender
        self.baseline_models = {
            'linear_regression': LinearRegression(),                    # O mais simples, linha reta
            'ridge': Ridge(alpha=1.0),                                # Linear com regularização
            'lasso': Lasso(alpha=0.1),                               # Linear que elimina features
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),  # Várias árvores votando
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),  # Aprende com erros
            'decision_tree': DecisionTreeRegressor(random_state=42),   # Uma árvore só
            'knn': KNeighborsRegressor(n_neighbors=5),                # Olha os 5 vizinhos mais próximos
            'svr': SVR(kernel='rbf', C=1.0)                           # Support Vector, o mais inteligente
        }
        
        # Melhor modelo
        self.best_model = None
        self.best_model_name = None
        self.best_score = -np.inf
        
        # Preprocessadores
        self.numeric_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler())
        ])
        
        # Preprocessador customizado para features categóricas
        from sklearn.preprocessing import FunctionTransformer
        
        self.categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='UNKNOWN')),
            ('encoder', FunctionTransformer(encode_categorical, validate=False))
        ])
        
        # Feature engineering
        self.feature_engineering_pipeline = None
        
    def create_advanced_features(self, df: pl.DataFrame) -> pl.DataFrame:
        # Chama o módulo de feature engineering pra criar um monte de features novas
        # A partir dos dados básicos (volume, distância) ele cria +60 features:
        # - Temporais: dia da semana, estação, feriados
        # - Geográficas: regiões, estados, complexidade da rota
        # - Operacionais: eficiência, tipo de modal codificado
        # - Econômicas: preços calculados, categorias
        # - Interações: cruzamento entre features
        # - Derivadas: log, sqrt, ratios
        logger.info("Criando features avançadas...")
        
        feature_engineer = FreteFeatureEngineer()
        df_enhanced = feature_engineer.create_all_features(df)
        
        # Analisa qual feature tem mais correlação com o preço
        feature_importance = feature_engineer.analyze_feature_importance(df_enhanced)
        
        self.feature_summary = feature_engineer.get_feature_summary()
        
        logger.info(f"Features criadas. Shape final: {df_enhanced.shape}")
        logger.info(f"Total de features criadas: {self.feature_summary['feature_stats']['total_features']}")
        
        return df_enhanced
    
    def prepare_features(self, df: pl.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        logger.info("Preparando features para o modelo...")
        
        df_pandas = df.to_pandas()
        
        numeric_cols = [col for col in df_pandas.columns if df_pandas[col].dtype in ['int64', 'float64']]
        numeric_cols = [col for col in numeric_cols if col != self.target_feature]
        
        categorical_cols = [col for col in df_pandas.columns if df_pandas[col].dtype == 'object']
        
        X = df_pandas[numeric_cols + categorical_cols]
        y = df_pandas[self.target_feature]
        
        mask = ~y.isna()
        X = X[mask]
        y = y[mask]
        
        self.feature_engineering_pipeline = ColumnTransformer(
            transformers=[
                ('num', self.numeric_transformer, numeric_cols),
                ('cat', self.categorical_transformer, categorical_cols)
            ],
            remainder='drop'
        )
        
        X_processed = self.feature_engineering_pipeline.fit_transform(X)
        
        logger.info(f"Features preparadas. X shape: {X_processed.shape}, y shape: {y.shape}")
        return X_processed, y.values
    
    def train_baseline_models(self, df: pl.DataFrame) -> Dict[str, Dict[str, float]]:
        # Aqui é onde a mágica acontece! Treina todos os 8 modelos e compara eles
        # Pra cada modelo calcula várias métricas de qualidade:
        # - MAE (Mean Absolute Error): erro médio absoluto em R$/ton/km
        # - RMSE (Root Mean Square Error): penaliza erros grandes
        # - R² (R-squared): % da variação que o modelo explica (0-1, quanto maior melhor)
        # - Cross-validation: testa em várias fatias dos dados pra ver se é consistente
        logger.info("Treinando modelos baseline...")
        
        X, y = self.prepare_features(df)
        
        # Divide 80% pra treinar e 20% pra testar (dados que o modelo nunca viu)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=None
        )
        
        results = {}
        
        for name, model in self.baseline_models.items():
            logger.info(f"Treinando {name}...")
            
            try:
                model.fit(X_train, y_train)
                
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)
                
                # Calcula as métricas de qualidade do modelo
                metrics = {
                    'train_mae': mean_absolute_error(y_train, y_pred_train),      # Erro médio no treino
                    'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),  # RMSE no treino
                    'train_r2': r2_score(y_train, y_pred_train),                 # R² no treino
                    'test_mae': mean_absolute_error(y_test, y_pred_test),        # Erro médio no teste
                    'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),    # RMSE no teste
                    'test_r2': r2_score(y_test, y_pred_test)                     # R² no teste (mais importante!)
                }
                
                # Cross-validation: testa em 5 fatias diferentes pros dados
                # pra ver se o modelo é consistente ou se deu sorte
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
                metrics['cv_r2_mean'] = cv_scores.mean()
                metrics['cv_r2_std'] = cv_scores.std()
                
                results[name] = metrics
                
                # Guarda o melhor modelo baseado no R² de teste
                if metrics['test_r2'] > self.best_score:
                    self.best_score = metrics['test_r2']
                    self.best_model = model
                    self.best_model_name = name
                
                logger.info(f"{name} - R² teste: {metrics['test_r2']:.4f}")
                
            except Exception as e:
                logger.error(f"Erro ao treinar {name}: {e}")
                results[name] = {'error': str(e)}
        
        return results
    
    def hyperparameter_tuning(self, df: pl.DataFrame) -> Dict[str, Any]:
        if self.best_model is None:
            logger.warning("Nenhum modelo treinado para tuning")
            return {}
        
        logger.info(f"Otimizando hiperparâmetros de {self.best_model_name}...")
        
        # Preparar features
        X, y = self.prepare_features(df)
        
        param_grids = {
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0]
            },
            'ridge': {
                'alpha': [0.1, 1.0, 10.0, 100.0]
            },
            'lasso': {
                'alpha': [0.001, 0.01, 0.1, 1.0]
            }
        }
        
        if self.best_model_name in param_grids:
            grid_search = GridSearchCV(
                self.best_model,
                param_grids[self.best_model_name],
                cv=5,
                scoring='r2',
                n_jobs=-1
            )
            
            grid_search.fit(X, y)
            
            self.best_model = grid_search.best_estimator_
            
            return {
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'cv_results': grid_search.cv_results_
            }
        
        return {}
    
    def save_baseline_model(self) -> None:
        if self.best_model is None:
            logger.warning("Nenhum modelo para salvar")
            return
        
        baseline_dir = self.models_dir / "baseline_model"
        baseline_dir.mkdir(exist_ok=True)
        
        model_path = baseline_dir / "best_model.joblib"
        joblib.dump(self.best_model, model_path)
        
        preprocessor_path = baseline_dir / "feature_engineering_pipeline.joblib"
        joblib.dump(self.feature_engineering_pipeline, preprocessor_path)
        
        metadata = {
            'model_name': self.best_model_name,
            'best_score': self.best_score,
            'training_date': datetime.now().isoformat(),
            'features': {
                'numeric': self.numeric_features,
                'categorical': self.categorical_features
            },
            'target': self.target_feature
        }
        
        metadata_path = baseline_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Modelo baseline salvo em: {baseline_dir}")
    
    def predict_baseline(self, df: pl.DataFrame) -> np.ndarray:
        if self.best_model is None:
            raise ValueError("Modelo não treinado")
        
        X, _ = self.prepare_features(df)
        
        return self.best_model.predict(X)
    
    def generate_baseline_report(self, df: pl.DataFrame, results: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        logger.info("Gerando relatório baseline...")
        
        model_ranking = []
        for name, metrics in results.items():
            if 'error' not in metrics:
                model_ranking.append({
                    'model': name,
                    'test_r2': metrics['test_r2'],
                    'test_mae': metrics['test_mae'],
                    'test_rmse': metrics['test_rmse'],
                    'cv_r2_mean': metrics['cv_r2_mean']
                })
        
        model_ranking.sort(key=lambda x: x['test_r2'], reverse=True)
        
        if self.best_model is not None:
            predictions = self.predict_baseline(df)
            
            actual_prices = df.select(self.target_feature).to_series().to_numpy()
            mask = ~np.isnan(actual_prices)
            
            if np.any(mask):
                mae = mean_absolute_error(actual_prices[mask], predictions[mask])
                rmse = np.sqrt(mean_squared_error(actual_prices[mask], predictions[mask]))
                r2 = r2_score(actual_prices[mask], predictions[mask])
            else:
                mae = rmse = r2 = np.nan
        else:
            mae = rmse = r2 = np.nan
        
        return {
            'model_ranking': model_ranking,
            'best_model': {
                'name': self.best_model_name,
                'score': self.best_score,
                'final_metrics': {
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2
                }
            },
            'training_summary': {
                'total_models': len(results),
                'successful_models': len([m for m in results.values() if 'error' not in m]),
                'failed_models': len([m for m in results.values() if 'error' in m])
            }
        }
