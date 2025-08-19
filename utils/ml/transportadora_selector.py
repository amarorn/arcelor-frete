# Sistema inteligente de recomendação de transportadoras
# Não só olha preço, mas também performance, confiabilidade, capacidade
# Calcula scores pra cada transportadora e recomenda a melhor pra cada situação

import polars as pl
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.multioutput import MultiOutputClassifier
import joblib
import logging
from pathlib import Path
from typing import Tuple, Dict, Any, List
import os
import pickle
import pandas as pd

logger = logging.getLogger(__name__)


class TransportadoraSelector:
    
    def __init__(self, model_path: str = "models/transportadora_selector.joblib"):
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.model_path = Path(model_path)
        self.model_path.parent.mkdir(exist_ok=True)
        
        # Features para o modelo
        self.numeric_features = [
            'volume_ton', 'distancia_km', 'preco_ton_km', 'preco_total',
            'mes', 'trimestre', 'ano', 'volume_historico', 'performance_score',
            'confiabilidade_score', 'capacidade_score', 'custo_beneficio_score'
        ]
        
        self.categorical_features = [
            'modal', 'tipo_rodovia', 'tipo_veiculo', 'regiao_origem', 'regiao_destino'
        ]
        
        self.target_features = [
            'transportadora_recomendada', 'score_recomendacao', 'categoria_prioridade'
        ]
        
        # Critérios de avaliação: como cada aspecto influencia na nota final
        # Total = 100% distribuído entre 5 critérios principais
        self.criterios_avaliacao = {
            'preco': 0.25,           # 25% - Preço continua importante, mas não é tudo
            'performance': 0.20,      # 20% - Histórico de desempenho (volume, frequência)
            'confiabilidade': 0.25,   # 25% - Consistência nos preços (não varia muito)
            'capacidade': 0.15,       # 15% - Capacidade operacional (consegue atender?)
            'custo_beneficio': 0.15   # 15% - Relação geral custo x benefício
        }
    
    def prepare_features(self, df: pl.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepara features para treinamento do modelo
        """
        logger.info("Preparando features para seleção de transportadora...")
        
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
                X_categorical[:, i] = self.label_encoders[feature].transform(
                    df_pandas[feature].fillna('UNKNOWN')
                )
        
        # Combinar features
        X = np.hstack([X_numeric, X_categorical])
        
        # Target (transportadora recomendada) - encoded
        if 'transportadora_recomendada' not in self.label_encoders:
            self.label_encoders['transportadora_recomendada'] = LabelEncoder()
            y = self.label_encoders['transportadora_recomendada'].fit_transform(
                df_pandas['transportadora_recomendada'].fillna('UNKNOWN')
            )
        else:
            y = self.label_encoders['transportadora_recomendada'].transform(
                df_pandas['transportadora_recomendada'].fillna('UNKNOWN')
            )
        
        # Remover linhas com valores NaN (apenas para features numéricas)
        mask = ~pd.isna(df_pandas['transportadora_recomendada'])
        X = X[mask]
        y = y[mask]
        
        logger.info(f"Features preparadas. X shape: {X.shape}, y shape: {y.shape}")
        return X, y
    
    def calculate_performance_scores(self, df: pl.DataFrame) -> pl.DataFrame:
        # Score de performance: como cada transportadora tá se saindo no geral
        # Olha 4 aspectos: preço médio, volume movimentado, frequência, estabilidade nos preços
        # Cada aspecto ganha um score 0-1, depois faz média ponderada
        logger.info("Calculando scores de performance...")
        
        # Agrega dados por transportadora
        performance_scores = df.group_by('transportadora').agg([
            pl.col('preco_ton_km').mean().alias('preco_medio'),           # Preço médio praticado
            pl.col('volume_ton').sum().alias('volume_total'),            # Total de volume movimentado
            pl.len().alias('num_entregas'),                              # Número de entregas feitas
            pl.col('preco_ton_km').std().alias('variabilidade_preco')    # Desvio padrão dos preços
        ])
        
        # Normaliza scores (0-1): transforma valores absolutos em scores relativos
        performance_scores = performance_scores.with_columns([
            # Score de preço: quanto menor o preço, maior o score (invertido)
            ((pl.col('preco_medio').max() - pl.col('preco_medio')) / 
             (pl.col('preco_medio').max() - pl.col('preco_medio').min())).alias('score_preco'),
            # Score de volume: quanto maior o volume, maior o score (direto)
            (pl.col('volume_total') / pl.col('volume_total').max()).alias('score_volume'),
            # Score de frequência: quanto mais entregas, maior o score (direto)
            (pl.col('num_entregas') / pl.col('num_entregas').max()).alias('score_frequencia'),
            # Score de estabilidade: quanto menor a variação, maior o score (invertido)
            ((pl.col('variabilidade_preco').max() - pl.col('variabilidade_preco')) / 
             (pl.col('variabilidade_preco').max() - pl.col('variabilidade_preco').min())).alias('score_estabilidade')
        ])
        
        # Score final de performance: média ponderada dos 4 aspectos
        performance_scores = performance_scores.with_columns([
            (pl.col('score_preco') * 0.4 +        # 40% preço (mais importante)
             pl.col('score_volume') * 0.3 +       # 30% volume
             pl.col('score_frequencia') * 0.2 +   # 20% frequência  
             pl.col('score_estabilidade') * 0.1   # 10% estabilidade
            ).alias('performance_score')
        ])
        
        return performance_scores
    
    def calculate_reliability_scores(self, df: pl.DataFrame) -> pl.DataFrame:
        # Score de confiabilidade: quão previsível é a transportadora?
        # Usa coeficiente de variação (CV): desvio_padrão / média
        # CV baixo = preços consistentes = mais confiável
        logger.info("Calculando scores de confiabilidade...")
        
        # Pega estatísticas de consistência por transportadora
        reliability_scores = df.group_by('transportadora').agg([
            pl.col('preco_ton_km').std().alias('desvio_preco'),    # Desvio padrão dos preços
            pl.col('preco_ton_km').mean().alias('preco_medio'),    # Preço médio
            pl.len().alias('num_entregas'),                        # Número de entregas
            pl.col('volume_ton').sum().alias('volume_total')       # Volume total
        ])
        
        # Calcula coeficiente de variação (CV = desvio/média)
        # CV alto = preços muito variáveis = menos confiável
        reliability_scores = reliability_scores.with_columns([
            (pl.col('desvio_preco') / pl.col('preco_medio')).alias('cv_preco')
        ])
        
        # Transforma CV em score: quanto menor o CV, maior o score de confiabilidade
        reliability_scores = reliability_scores.with_columns([
            ((pl.col('cv_preco').max() - pl.col('cv_preco')) / 
             (pl.col('cv_preco').max() - pl.col('cv_preco').min())).alias('confiabilidade_score')
        ])
        
        return reliability_scores
    
    def calculate_capacity_scores(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Calcula scores de capacidade operacional
        """
        logger.info("Calculando scores de capacidade...")
        
        capacity_scores = df.group_by('transportadora').agg([
            pl.col('volume_ton').sum().alias('volume_total'),
            pl.col('distancia_km').mean().alias('distancia_media'),
            pl.len().alias('num_entregas'),
            pl.col('volume_ton').max().alias('volume_max_entrega')
        ])
        
        # Normalizar scores usando as colunas do DataFrame agrupado
        capacity_scores = capacity_scores.with_columns([
            (pl.col('volume_total') / pl.col('volume_total').max()).alias('score_volume_total'),
            ((pl.col('distancia_media').max() - pl.col('distancia_media')) / 
             (pl.col('distancia_media').max() - pl.col('distancia_media').min())).alias('score_distancia'),
            (pl.col('num_entregas') / pl.col('num_entregas').max()).alias('score_frequencia'),
            (pl.col('volume_max_entrega') / pl.col('volume_max_entrega').max()).alias('score_capacidade_entrega')
        ])
        
        # Score composto de capacidade
        capacity_scores = capacity_scores.with_columns([
            (pl.col('score_volume_total') * 0.4 + 
             pl.col('score_distancia') * 0.2 + 
             pl.col('score_frequencia') * 0.2 + 
             pl.col('score_capacidade_entrega') * 0.2).alias('capacidade_score')
        ])
        
        return capacity_scores
    
    def calculate_cost_benefit_scores(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Calcula scores de custo-benefício
        """
        logger.info("Calculando scores de custo-benefício...")
        
        cost_benefit_scores = df.group_by('transportadora').agg([
            pl.col('preco_ton_km').mean().alias('preco_medio'),
            pl.col('volume_ton').sum().alias('volume_total'),
            pl.col('distancia_km').mean().alias('distancia_media'),
            pl.len().alias('num_entregas')
        ])
        
        # Calcular custo total por km usando as colunas do DataFrame agrupado
        cost_benefit_scores = cost_benefit_scores.with_columns([
            (pl.col('preco_medio') * pl.col('volume_total') * pl.col('distancia_media')).alias('custo_total_km')
        ])
        
        # Normalizar scores (menor custo = maior score)
        cost_benefit_scores = cost_benefit_scores.with_columns([
            ((pl.col('custo_total_km').max() - pl.col('custo_total_km')) / 
             (pl.col('custo_total_km').max() - pl.col('custo_total_km').min())).alias('custo_beneficio_score')
        ])
        
        return cost_benefit_scores
    
    def generate_training_data(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Gera dados de treinamento com todas as features calculadas
        """
        logger.info("Gerando dados de treinamento...")
        
        # Calcular todos os scores
        performance_scores = self.calculate_performance_scores(df)
        reliability_scores = self.calculate_reliability_scores(df)
        capacity_scores = self.calculate_capacity_scores(df)
        cost_benefit_scores = self.calculate_cost_benefit_scores(df)
        
        # Juntar todos os scores, renomeando colunas conflitantes
        all_scores = performance_scores.join(reliability_scores, on='transportadora', how='left', suffix='_rel')
        all_scores = all_scores.join(capacity_scores, on='transportadora', how='left', suffix='_cap')
        all_scores = all_scores.join(cost_benefit_scores, on='transportadora', how='left', suffix='_cb')
        
        # Calcular score composto final
        all_scores = all_scores.with_columns([
            (pl.col('performance_score') * self.criterios_avaliacao['performance'] +
             pl.col('confiabilidade_score') * self.criterios_avaliacao['confiabilidade'] +
             pl.col('capacidade_score') * self.criterios_avaliacao['capacidade'] +
             pl.col('custo_beneficio_score') * self.criterios_avaliacao['custo_beneficio']).alias('score_final')
        ])
        
        # Adicionar features ao DataFrame original
        df_with_scores = df.join(all_scores.select(['transportadora', 'performance_score', 
                                                   'confiabilidade_score', 'capacidade_score', 
                                                   'custo_beneficio_score', 'score_final']), 
                                on='transportadora', how='left')
        
        # Adicionar features calculadas
        df_with_scores = df_with_scores.with_columns([
            pl.col('frete_brl').alias('preco_total'),
            pl.col('volume_ton').alias('volume_historico'),
            pl.col('preco_ton_km').alias('preco_ton_km'),
            pl.col('distancia_km').alias('distancia_km'),
            pl.col('mes').alias('mes'),
            pl.col('trimestre').alias('trimestre'),
            pl.col('ano').alias('ano')
        ])
        
        # Adicionar features de região (simplificado)
        df_with_scores = df_with_scores.with_columns([
            pl.lit('REGIAO_1').alias('regiao_origem'),
            pl.lit('REGIAO_2').alias('regiao_destino')
        ])
        
        # Selecionar transportadora com maior score para cada rota e manter todas as features
        df_training = df_with_scores.with_columns([
            pl.col('transportadora').alias('transportadora_recomendada'),
            pl.col('score_final').alias('score_recomendacao'),
            pl.when(pl.col('score_final') >= 0.8).then(pl.lit('ALTA'))
            .when(pl.col('score_final') >= 0.6).then(pl.lit('MÉDIA'))
            .otherwise(pl.lit('BAIXA')).alias('categoria_prioridade')
        ])
        
        return df_training
    
    def train(self, df: pl.DataFrame) -> Dict[str, float]:
        """
        Treina o modelo de seleção de transportadora
        """
        logger.info("Iniciando treinamento do modelo de seleção...")
        
        # Gerar dados de treinamento
        df_training = self.generate_training_data(df)
        
        # Preparar features
        X, y = self.prepare_features(df_training)
        
        # Split treino/teste (remover stratify se amostras insuficientes)
        unique_classes = len(np.unique(y))
        test_size = max(0.2, min(0.5, unique_classes / len(y)))
        
        if len(y) >= 2 * unique_classes:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
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
            'train_accuracy': accuracy_score(y_train, y_pred_train),
            'test_accuracy': accuracy_score(y_test, y_pred_test),
            'train_classification_report': classification_report(y_train, y_pred_train, output_dict=True),
            'test_classification_report': classification_report(y_test, y_pred_test, output_dict=True)
        }
        
        # Cross-validation (ajustar cv para dados pequenos)
        cv_folds = min(5, len(y_train), unique_classes)
        if cv_folds >= 2:
            cv_scores = cross_val_score(
                self.model, X_train_scaled, y_train, 
                cv=cv_folds, scoring='accuracy'
            )
        else:
            cv_scores = np.array([metrics['train_accuracy']])
        metrics['cv_accuracy_mean'] = cv_scores.mean()
        metrics['cv_accuracy_std'] = cv_scores.std()
        
        logger.info("Modelo de seleção treinado com sucesso!")
        logger.info(f"Acurácia de treino: {metrics['train_accuracy']:.4f}")
        logger.info(f"Acurácia de teste: {metrics['test_accuracy']:.4f}")
        logger.info(f"CV Acurácia: {metrics['cv_accuracy_mean']:.4f} (+/- {metrics['cv_accuracy_std']*2:.4f})")
        
        return metrics
    
    def predict_transportadora(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Faz previsões de transportadora recomendada para novos dados
        """
        logger.info("Fazendo previsões de transportadora...")
        
        # Gerar dados com scores
        df_with_scores = self.generate_training_data(df)
        
        # Preparar features
        X, _ = self.prepare_features(df_with_scores)
        X_scaled = self.scaler.transform(X)
        
        # Fazer previsões
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
        # Adicionar previsões ao DataFrame
        df_with_predictions = df_with_scores.with_columns([
            pl.Series("transportadora_recomendada_ml", predictions),
            pl.Series("score_confianca_ml", np.max(probabilities, axis=1))
        ])
        
        return df_with_predictions
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Retorna importância das features
        """
        feature_names = self.numeric_features + self.categorical_features
        importance = self.model.feature_importances_
        
        return dict(zip(feature_names, importance))
    
    def save_model(self) -> None:
        """
        Salva o modelo treinado
        """
        try:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            models_dir = os.path.join(project_root, "models")
        except NameError:
            models_dir = "../../models"
        
        os.makedirs(models_dir, exist_ok=True)
        
        # Salvar modelo
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.numeric_features + self.categorical_features,
            'criterios_avaliacao': self.criterios_avaliacao
        }
        
        # Salvar em múltiplos formatos
        joblib_path = os.path.join(models_dir, "transportadora_selector.joblib")
        joblib.dump(model_data, joblib_path)
        logger.info(f"Modelo salvo em formato joblib: {joblib_path}")
        
        pickle_path = os.path.join(models_dir, "transportadora_selector.pkl")
        with open(pickle_path, 'wb') as f:
            pickle.dump(model_data, f)
        logger.info(f"Modelo salvo em formato pickle: {pickle_path}")
        
        # Salvar metadados
        metadata = {
            'model_type': 'RandomForestClassifier',
            'features': {
                'numeric': self.numeric_features,
                'categorical': self.categorical_features
            },
            'target_features': self.target_features,
            'criterios_avaliacao': self.criterios_avaliacao,
            'feature_importance': self.get_feature_importance(),
            'training_date': str(pd.Timestamp.now()),
            'version': '1.0'
        }
        
        metadata_path = os.path.join(models_dir, "transportadora_selector_metadata.json")
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info(f"Metadados salvos em: {metadata_path}")
    
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
    
    def generate_selection_report(self, df: pl.DataFrame) -> Dict[str, Any]:
        """
        Gera relatório completo de seleção de transportadoras
        """
        logger.info("Gerando relatório de seleção...")
        
        # Fazer previsões
        df_with_predictions = self.predict_transportadora(df)
        
        # Análise por transportadora
        analysis_by_carrier = df_with_predictions.group_by('transportadora_recomendada_ml').agg([
            pl.len().alias('num_recomendacoes'),
            pl.col('score_confianca_ml').mean().alias('score_confianca_medio'),
            pl.col('volume_ton').sum().alias('volume_total_recomendado'),
            pl.col('preco_ton_km').mean().alias('preco_medio_recomendado')
        ]).sort('num_recomendacoes', descending=True)
        
        # Análise por rota
        analysis_by_route = df_with_predictions.group_by('rota').agg([
            pl.col('transportadora_recomendada_ml').first().alias('transportadora_recomendada'),
            pl.col('score_confianca_ml').first().alias('score_confianca'),
            pl.col('volume_ton').sum().alias('volume_total'),
            pl.col('preco_ton_km').mean().alias('preco_medio')
        ]).sort('volume_total', descending=True)
        
        # Estatísticas gerais
        total_recommendations = len(df_with_predictions)
        avg_confidence = df_with_predictions['score_confianca_ml'].mean()
        top_carrier = analysis_by_carrier.select('transportadora_recomendada_ml').item(0, 0)
        
        return {
            'resumo_geral': {
                'total_recomendacoes': total_recommendations,
                'score_confianca_medio': float(avg_confidence),
                'transportadora_mais_recomendada': top_carrier
            },
            'analise_por_transportadora': analysis_by_carrier.to_pandas().to_dict('records'),
            'analise_por_rota': analysis_by_route.to_pandas().to_dict('records'),
            'metricas_modelo': {
                'feature_importance': self.get_feature_importance(),
                'criterios_avaliacao': self.criterios_avaliacao
            }
        }
