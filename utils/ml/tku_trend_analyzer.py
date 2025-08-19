"""
Sistema de Análise de Tendências Temporais Inteligente para TKU
Analisa tendências de preços por rota em diferentes períodos (3, 6, 12 meses)
Fornece recomendações práticas para o operador
"""

import polars as pl
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class TKUTrendAnalyzer:
    """
    Analisador inteligente de tendências de TKU por rota
    """
    
    def __init__(self):
        self.trend_models = {}
        self.analysis_cache = {}
        
        # Configurações de análise
        self.periods = {
            'short_term': 90,    # 3 meses
            'medium_term': 180,  # 6 meses
            'long_term': 365     # 12 meses
        }
        
        # Thresholds para classificação de tendências
        self.trend_thresholds = {
            'stable_threshold': 0.05,      # 5% de variação = estável
            'moderate_threshold': 0.15,    # 15% de variação = moderado
            'fast_threshold': 0.25         # 25% de variação = rápido
        }
    
    def analyze_route_tku_trends(self, df: pl.DataFrame, rota_id: str) -> Dict[str, Any]:
        """
        Análise completa de tendências de TKU para uma rota específica
        
        Args:
            df: DataFrame com dados históricos
            rota_id: ID da rota para análise
            
        Returns:
            Dicionário com análise completa da rota
        """
        logger.info(f"Iniciando análise de tendências para rota: {rota_id}")
        
        try:
            # Filtrar dados da rota específica
            df_rota = self._filter_route_data(df, rota_id)
            
            if df_rota.is_empty():
                logger.warning(f"Nenhum dado encontrado para rota: {rota_id}")
                return self._create_empty_analysis(rota_id)
            
            # Calcular TKU histórico
            df_rota = self._calculate_tku_historical(df_rota)
            
            # Análise de tendências por período
            trends_analysis = {
                '3_meses': self._analyze_short_term_trend(df_rota),
                '6_meses': self._analyze_medium_term_trend(df_rota),
                '12_meses': self._analyze_long_term_trend(df_rota)
            }
            
            # Análise de sazonalidade
            seasonality_analysis = self._analyze_seasonality(df_rota)
            
            # Análise de benchmark
            benchmark_analysis = self._analyze_route_benchmark(df, rota_id, df_rota)
            
            # Gerar recomendações
            recommendations = self._generate_recommendations(trends_analysis, benchmark_analysis)
            
            # Análise completa
            comprehensive_analysis = {
                'rota_id': rota_id,
                'data_analise': datetime.now().isoformat(),
                'periodo_analise': {
                    'inicio': df_rota['data_faturamento'].min().isoformat(),
                    'fim': df_rota['data_faturamento'].max().isoformat(),
                    'total_entregas': len(df_rota)
                },
                'tku_atual': float(df_rota['tku_historico'].mean()),
                'tku_anterior': self._get_previous_period_tku(df_rota),
                'variacao_percentual': self._calculate_percentage_change(df_rota),
                'tendencias': trends_analysis,
                'sazonalidade': seasonality_analysis,
                'benchmark': benchmark_analysis,
                'recomendacoes': recommendations,
                'status_geral': self._get_route_status(trends_analysis, benchmark_analysis)
            }
            
            # Cache da análise
            self.analysis_cache[rota_id] = comprehensive_analysis
            
            logger.info(f"Análise concluída para rota: {rota_id}")
            return comprehensive_analysis
            
        except Exception as e:
            logger.error(f"Erro na análise da rota {rota_id}: {str(e)}")
            return self._create_error_analysis(rota_id, str(e))
    
    def _filter_route_data(self, df: pl.DataFrame, rota_id: str) -> pl.DataFrame:
        """Filtra dados para uma rota específica"""
        # Tentar diferentes colunas de identificação de rota
        route_columns = ['rota_municipio', 'rota_microregiao', 'rota_mesoregiao', 'rota_uf']
        
        for col in route_columns:
            if col in df.columns:
                filtered_df = df.filter(pl.col(col) == rota_id)
                if not filtered_df.is_empty():
                    return filtered_df
        
        # Se não encontrar, retornar DataFrame vazio
        return pl.DataFrame()
    
    def _calculate_tku_historical(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calcula TKU histórico para cada entrega"""
        return df.with_columns([
            (pl.col('frete_brl') / (pl.col('volume_ton') * pl.col('distancia_km')))
            .alias('tku_historico')
        ]).filter(
            (pl.col('tku_historico').is_finite()) & 
            (pl.col('tku_historico') > 0)
        )
    
    def _analyze_short_term_trend(self, df: pl.DataFrame) -> Dict[str, Any]:
        """Análise de tendência de 3 meses"""
        return self._analyze_trend_period(df, days=90, period_name="3 meses")
    
    def _analyze_medium_term_trend(self, df: pl.DataFrame) -> Dict[str, Any]:
        """Análise de tendência de 6 meses"""
        return self._analyze_trend_period(df, days=180, period_name="6 meses")
    
    def _analyze_long_term_trend(self, df: pl.DataFrame) -> Dict[str, Any]:
        """Análise de tendência de 12 meses"""
        return self._analyze_trend_period(df, days=365, period_name="12 meses")
    
    def _analyze_trend_period(self, df: pl.DataFrame, days: int, period_name: str) -> Dict[str, Any]:
        """Análise de tendência para um período específico"""
        try:
            # Filtrar período
            cutoff_date = datetime.now() - timedelta(days=days)
            df_period = df.filter(pl.col('data_faturamento') >= pl.lit(cutoff_date))
            
            if df_period.is_empty():
                return {
                    'periodo': period_name,
                    'tendencia': 'DADOS_INSUFICIENTES',
                    'velocidade': 'N/A',
                    'confianca': 0.0,
                    'tku_atual': 0.0,
                    'tku_anterior': 0.0,
                    'variacao_percentual': 0.0,
                    'dados_disponiveis': False
                }
            
            # Calcular estatísticas
            tku_values = df_period['tku_historico'].to_numpy()
            tku_atual = float(np.mean(tku_values))
            
            # Período anterior para comparação
            previous_cutoff = cutoff_date - timedelta(days=days)
            df_previous = df.filter(
                (pl.col('data_faturamento') >= pl.lit(previous_cutoff)) &
                (pl.col('data_faturamento') < pl.lit(cutoff_date))
            )
            
            if df_previous.is_empty():
                tku_anterior = tku_atual
                variacao_percentual = 0.0
            else:
                tku_anterior = float(df_previous['tku_historico'].mean())
                variacao_percentual = ((tku_atual - tku_anterior) / tku_anterior * 100) if tku_anterior > 0 else 0.0
            
            # Análise de tendência usando regressão linear
            trend_direction, trend_speed, confidence = self._calculate_trend_metrics(df_period)
            
            return {
                'periodo': period_name,
                'tendencia': trend_direction,
                'velocidade': trend_speed,
                'confianca': confidence,
                'tku_atual': tku_atual,
                'tku_anterior': tku_anterior,
                'variacao_percentual': variacao_percentual,
                'dados_disponiveis': True,
                'num_entregas': len(df_period)
            }
            
        except Exception as e:
            logger.error(f"Erro na análise de {period_name}: {str(e)}")
            return {
                'periodo': period_name,
                'tendencia': 'ERRO_ANALISE',
                'velocidade': 'N/A',
                'confianca': 0.0,
                'tku_atual': 0.0,
                'tku_anterior': 0.0,
                'variacao_percentual': 0.0,
                'dados_disponiveis': False,
                'erro': str(e)
            }
    
    def _calculate_trend_metrics(self, df_period: pl.DataFrame) -> Tuple[str, str, float]:
        """Calcula métricas de tendência usando regressão linear"""
        try:
            # Preparar dados para regressão
            df_sorted = df_period.sort('data_faturamento')
            
            # Criar índice temporal
            X = np.arange(len(df_sorted)).reshape(-1, 1)
            y = df_sorted['tku_historico'].to_numpy()
            
            if len(y) < 2:
                return 'ESTÁVEL', 'N/A', 0.0
            
            # Ajustar regressão linear
            model = LinearRegression()
            model.fit(X, y)
            
            # Calcular coeficiente (inclinação)
            slope = model.coef_[0]
            
            # Calcular R² (confiança do modelo)
            y_pred = model.predict(X)
            r2 = r2_score(y, y_pred)
            
            # Determinar direção da tendência
            if abs(slope) < self.trend_thresholds['stable_threshold']:
                direction = 'ESTÁVEL'
            elif slope > 0:
                direction = 'SUBINDO'
            else:
                direction = 'CAINDO'
            
            # Determinar velocidade da tendência
            slope_abs = abs(slope)
            if slope_abs < self.trend_thresholds['stable_threshold']:
                speed = 'ESTÁVEL'
            elif slope_abs < self.trend_thresholds['moderate_threshold']:
                speed = 'LENTO'
            elif slope_abs < self.trend_thresholds['fast_threshold']:
                speed = 'MÉDIO'
            else:
                speed = 'RÁPIDO'
            
            # Confiança baseada no R²
            confidence = min(100.0, max(0.0, r2 * 100))
            
            return direction, speed, confidence
            
        except Exception as e:
            logger.error(f"Erro no cálculo de métricas de tendência: {str(e)}")
            return 'ERRO', 'N/A', 0.0
    
    def _analyze_seasonality(self, df: pl.DataFrame) -> Dict[str, Any]:
        """Analisa padrões sazonais no TKU"""
        try:
            # Extrair mês da data
            df_with_month = df.with_columns([
                pl.col('data_faturamento').dt.month().alias('mes_numero'),
                pl.col('data_faturamento').dt.month().alias('mes_nome')
            ])
            
            # Mapear nomes dos meses
            month_names = {
                1: 'Janeiro', 2: 'Fevereiro', 3: 'Março', 4: 'Abril',
                5: 'Maio', 6: 'Junho', 7: 'Julho', 8: 'Agosto',
                9: 'Setembro', 10: 'Outubro', 11: 'Novembro', 12: 'Dezembro'
            }
            
            df_with_month = df_with_month.with_columns([
                pl.col('mes_numero').map_elements(lambda x: month_names.get(x, 'Desconhecido')).alias('mes_nome')
            ])
            
            # Agrupar por mês
            seasonal_analysis = df_with_month.group_by('mes_numero', 'mes_nome').agg([
                pl.col('tku_historico').mean().alias('tku_medio_mes'),
                pl.col('tku_historico').std().alias('tku_variabilidade_mes'),
                pl.len().alias('num_entregas_mes')
            ]).sort('mes_numero')
            
            # Calcular TKU médio geral
            tku_medio_geral = df['tku_historico'].mean()
            
            # Classificar temporadas
            seasonal_analysis = seasonal_analysis.with_columns([
                pl.when(pl.col('tku_medio_mes') > tku_medio_geral * 1.1).then(pl.lit('ALTA'))
                .when(pl.col('tku_medio_mes') < tku_medio_geral * 0.9).then(pl.lit('BAIXA'))
                .otherwise(pl.lit('NORMAL')).alias('temporada')
            ])
            
            # Identificar meses críticos
            meses_criticos = seasonal_analysis.filter(pl.col('temporada') == 'ALTA')
            meses_otimos = seasonal_analysis.filter(pl.col('temporada') == 'BAIXA')
            
            return {
                'tku_medio_geral': float(tku_medio_geral),
                'analise_mensal': seasonal_analysis.to_pandas().to_dict('records'),
                'meses_alta_temporada': meses_criticos['mes_nome'].to_list(),
                'meses_baixa_temporada': meses_otimos['mes_nome'].to_list(),
                'variacao_sazonal': float(seasonal_analysis['tku_medio_mes'].max() - seasonal_analysis['tku_medio_mes'].min()),
                'padrao_sazonal_detectado': len(meses_criticos) > 0 or len(meses_otimos) > 0
            }
            
        except Exception as e:
            logger.error(f"Erro na análise de sazonalidade: {str(e)}")
            return {
                'erro': str(e),
                'tku_medio_geral': 0.0,
                'analise_mensal': [],
                'meses_alta_temporada': [],
                'meses_baixa_temporada': [],
                'variacao_sazonal': 0.0,
                'padrao_sazonal_detectado': False
            }
    
    def _analyze_route_benchmark(self, df_full: pl.DataFrame, rota_id: str, df_rota: pl.DataFrame) -> Dict[str, Any]:
        """Analisa posicionamento da rota em relação ao benchmark do mercado"""
        try:
            # Dados da rota específica
            tku_rota = float(df_rota['tku_historico'].mean())
            
            # Identificar micro-região da rota
            if 'microregiao_origem' in df_rota.columns:
                microregiao = df_rota['microregiao_origem'].first()
            elif 'rota_microregiao' in df_rota.columns:
                microregiao = df_rota['rota_microregiao'].first()
            else:
                microregiao = "GERAL"
            
            # Benchmark por micro-região
            if microregiao != "GERAL":
                benchmark_data = df_full.filter(pl.col('microregiao_origem') == microregiao)
            else:
                benchmark_data = df_full
            
            if benchmark_data.is_empty():
                benchmark_data = df_full  # Fallback para dados gerais
            
            # Calcular TKU do benchmark
            benchmark_tku = float(benchmark_data['frete_brl'].sum() / 
                                (benchmark_data['volume_ton'].sum() * benchmark_data['distancia_km'].sum()))
            
            # Calcular percentis para posicionamento
            tku_values = benchmark_data['frete_brl'] / (benchmark_data['volume_ton'] * benchmark_data['distancia_km'])
            tku_values = tku_values.filter(tku_values.is_finite() & (tku_values > 0))
            
            if tku_values.is_empty():
                percentis = {'25': tku_rota, '50': tku_rota, '75': tku_rota}
            else:
                percentis = {
                    '25': float(tku_values.quantile(0.25)),
                    '50': float(tku_values.quantile(0.50)),
                    '75': float(tku_values.quantile(0.75))
                }
            
            # Posicionamento da rota
            if tku_rota <= percentis['25']:
                posicionamento = 'EXCELENTE (Top 25%)'
                recomendacao = 'Manter estratégia atual'
                acao_necessaria = 'Nenhuma'
            elif tku_rota <= percentis['50']:
                posicionamento = 'BOM (Top 50%)'
                recomendacao = 'Oportunidade de melhoria'
                acao_necessaria = 'Monitorar tendências'
            elif tku_rota <= percentis['75']:
                posicionamento = 'REGULAR (Top 75%)'
                recomendacao = 'Ação necessária para redução'
                acao_necessaria = 'Negociar com transportadora'
            else:
                posicionamento = 'CRÍTICO (Top 100%)'
                recomendacao = 'Ação urgente necessária'
                acao_necessaria = 'Revisar estratégia completa'
            
            # Economia potencial
            economia_potencial = max(0, tku_rota - percentis['25'])
            
            return {
                'tku_rota': tku_rota,
                'tku_benchmark': benchmark_tku,
                'tku_mediana_microregiao': percentis['50'],
                'tku_quartil_25': percentis['25'],
                'tku_quartil_75': percentis['75'],
                'posicionamento': posicionamento,
                'recomendacao': recomendacao,
                'acao_necessaria': acao_necessaria,
                'economia_potencial': economia_potencial,
                'microregiao_analisada': microregiao
            }
            
        except Exception as e:
            logger.error(f"Erro na análise de benchmark: {str(e)}")
            return {
                'erro': str(e),
                'tku_rota': 0.0,
                'tku_benchmark': 0.0,
                'posicionamento': 'ERRO_ANALISE',
                'recomendacao': 'Verificar dados',
                'acao_necessaria': 'N/A',
                'economia_potencial': 0.0
            }
    
    def _generate_recommendations(self, trends_analysis: Dict, benchmark_analysis: Dict) -> List[Dict[str, Any]]:
        """Gera recomendações baseadas na análise"""
        recommendations = []
        
        try:
            # Recomendações baseadas em tendências
            short_term = trends_analysis.get('3_meses', {})
            medium_term = trends_analysis.get('6_meses', {})
            
            # Tendência de alta - ação urgente
            if short_term.get('tendencia') == 'SUBINDO' and short_term.get('velocidade') in ['MÉDIO', 'RÁPIDO']:
                recommendations.append({
                    'tipo': 'URGENTE',
                    'prioridade': 1,
                    'acao': 'Negociar com transportadora atual',
                    'justificativa': f'TKU subindo {short_term.get("velocidade")} nos últimos 3 meses',
                    'economia_esperada': '5-15%',
                    'prazo': 'Imediato (próxima entrega)',
                    'dificuldade': 'BAIXA'
                })
            
            # Tendência de alta a médio prazo
            if medium_term.get('tendencia') == 'SUBINDO':
                recommendations.append({
                    'tipo': 'ESTRATÉGICO',
                    'prioridade': 2,
                    'acao': 'Avaliar alternativas de transportadora',
                    'justificativa': 'Tendência de alta sustentada nos últimos 6 meses',
                    'economia_esperada': '10-25%',
                    'prazo': '1-2 meses',
                    'dificuldade': 'MÉDIA'
                })
            
            # Recomendações baseadas no benchmark
            if benchmark_analysis.get('posicionamento', '').startswith('CRÍTICO'):
                recommendations.append({
                    'tipo': 'CRÍTICO',
                    'prioridade': 1,
                    'acao': 'Revisar estratégia de logística completa',
                    'justificativa': f'TKU {benchmark_analysis.get("economia_potencial", 0):.4f} acima do benchmark',
                    'economia_esperada': '20-40%',
                    'prazo': '1 mês',
                    'dificuldade': 'ALTA'
                })
            
            # Recomendações baseadas em sazonalidade (se disponível)
            if 'sazonalidade' in trends_analysis:
                seasonality = trends_analysis['sazonalidade']
                if seasonality.get('padrao_sazonal_detectado'):
                    recommendations.append({
                        'tipo': 'ESTRATÉGICO',
                        'prioridade': 3,
                        'acao': 'Planejar entregas para baixa temporada',
                        'justificativa': f'Padrão sazonal detectado: {len(seasonality.get("meses_baixa_temporada", []))} meses de baixa',
                        'economia_esperada': '10-20%',
                        'prazo': '3-6 meses',
                        'dificuldade': 'BAIXA'
                    })
            
            # Ordenar por prioridade
            recommendations.sort(key=lambda x: x['prioridade'])
            
            # Se não houver recomendações específicas, gerar uma recomendação básica
            if not recommendations:
                # Verificar se há dados suficientes para análise
                if short_term.get('dados_disponiveis') and medium_term.get('dados_disponiveis'):
                    if short_term.get('tendencia') == 'ESTÁVEL' and medium_term.get('tendencia') == 'ESTÁVEL':
                        recommendations.append({
                            'tipo': 'INFORMATIVO',
                            'prioridade': 4,
                            'acao': 'Manter monitoramento da rota',
                            'justificativa': 'Tendências estáveis nos últimos 3 e 6 meses',
                            'economia_esperada': '0-5%',
                            'prazo': 'Contínuo',
                            'dificuldade': 'BAIXA'
                        })
                    else:
                        recommendations.append({
                            'tipo': 'MONITORAMENTO',
                            'prioridade': 3,
                            'acao': 'Acompanhar evolução das tendências',
                            'justificativa': f'Tendência 3M: {short_term.get("tendencia")}, 6M: {medium_term.get("tendencia")}',
                            'economia_esperada': '2-8%',
                            'prazo': '1 mês',
                            'dificuldade': 'BAIXA'
                        })
                else:
                    recommendations.append({
                        'tipo': 'INFORMATIVO',
                        'prioridade': 5,
                        'acao': 'Aguardar mais dados para análise',
                        'justificativa': 'Dados insuficientes para gerar recomendações específicas',
                        'economia_esperada': 'N/A',
                        'prazo': 'Quando houver dados suficientes',
                        'dificuldade': 'BAIXA'
                    })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Erro na geração de recomendações: {str(e)}")
            return [{
                'tipo': 'ERRO',
                'acao': 'Verificar análise',
                'justificativa': f'Erro na geração: {str(e)}',
                'economia_esperada': 'N/A',
                'prazo': 'N/A',
                'dificuldade': 'N/A'
            }]
    
    def _get_previous_period_tku(self, df: pl.DataFrame) -> float:
        """Obtém TKU do período anterior (3 meses atrás)"""
        try:
            cutoff_date = datetime.now() - timedelta(days=90)
            previous_start = cutoff_date - timedelta(days=90)
            
            df_previous = df.filter(
                (pl.col('data_faturamento') >= pl.lit(previous_start)) &
                (pl.col('data_faturamento') < pl.lit(cutoff_date))
            )
            
            if df_previous.is_empty():
                return float(df['tku_historico'].mean())
            
            return float(df_previous['tku_historico'].mean())
            
        except Exception as e:
            logger.error(f"Erro ao calcular TKU anterior: {str(e)}")
            return float(df['tku_historico'].mean())
    
    def _calculate_percentage_change(self, df: pl.DataFrame) -> float:
        """Calcula variação percentual do TKU"""
        try:
            current_tku = float(df['tku_historico'].mean())
            previous_tku = self._get_previous_period_tku(df)
            
            if previous_tku > 0:
                return ((current_tku - previous_tku) / previous_tku) * 100
            return 0.0
            
        except Exception as e:
            logger.error(f"Erro ao calcular variação percentual: {str(e)}")
            return 0.0
    
    def _get_route_status(self, trends_analysis: Dict, benchmark_analysis: Dict) -> str:
        """Determina status geral da rota"""
        try:
            short_term = trends_analysis.get('3_meses', {})
            benchmark_pos = benchmark_analysis.get('posicionamento', '')
            
            # Status crítico
            if (short_term.get('tendencia') == 'SUBINDO' and 
                short_term.get('velocidade') == 'RÁPIDO'):
                return 'CRÍTICO - Ação urgente necessária'
            
            if benchmark_pos.startswith('CRÍTICO'):
                return 'CRÍTICO - Muito acima do benchmark'
            
            # Status de atenção
            if short_term.get('tendencia') == 'SUBINDO':
                return 'ATENÇÃO - Monitorar tendência'
            
            if benchmark_pos.startswith('REGULAR'):
                return 'ATENÇÃO - Oportunidade de melhoria'
            
            # Status estável
            if short_term.get('tendencia') == 'ESTÁVEL':
                return 'ESTÁVEL - Manter estratégia atual'
            
            # Status positivo
            if short_term.get('tendencia') == 'CAINDO':
                return 'POSITIVO - Tendência favorável'
            
            return 'ANALISANDO - Dados insuficientes'
            
        except Exception as e:
            logger.error(f"Erro ao determinar status da rota: {str(e)}")
            return 'ERRO - Verificar análise'
    
    def _create_empty_analysis(self, rota_id: str) -> Dict[str, Any]:
        """Cria análise vazia quando não há dados"""
        return {
            'rota_id': rota_id,
            'erro': 'Nenhum dado encontrado para esta rota',
            'status_geral': 'SEM DADOS',
            'recomendacoes': [{
                'tipo': 'INFORMATIVO',
                'acao': 'Verificar dados da rota',
                'justificativa': 'Rota não encontrada ou sem histórico',
                'economia_esperada': 'N/A',
                'prazo': 'N/A'
            }]
        }
    
    def _create_error_analysis(self, rota_id: str, error_msg: str) -> Dict[str, Any]:
        """Cria análise de erro"""
        return {
            'rota_id': rota_id,
            'erro': error_msg,
            'status_geral': 'ERRO NA ANÁLISE',
            'recomendacoes': [{
                'tipo': 'ERRO',
                'acao': 'Verificar dados e tentar novamente',
                'justificativa': f'Erro durante análise: {error_msg}',
                'economia_esperada': 'N/A',
                'prazo': 'N/A'
            }]
        }
    
    def get_cached_analysis(self, rota_id: str) -> Optional[Dict[str, Any]]:
        """Retorna análise em cache se disponível"""
        return self.analysis_cache.get(rota_id)
    
    def clear_cache(self) -> None:
        """Limpa cache de análises"""
        self.analysis_cache.clear()
        logger.info("Cache de análises limpo")
    
    def get_analysis_summary(self, rota_id: str) -> Dict[str, Any]:
        """Retorna resumo da análise para uma rota"""
        analysis = self.get_cached_analysis(rota_id)
        if not analysis:
            return {'erro': 'Análise não encontrada. Execute analyze_route_tku_trends primeiro.'}
        
        return {
            'rota_id': analysis['rota_id'],
            'status_geral': analysis['status_geral'],
            'tku_atual': analysis['tku_atual'],
            'variacao_percentual': analysis['variacao_percentual'],
            'tendencia_3m': analysis['tendencias']['3_meses']['tendencia'],
            'posicionamento_mercado': analysis['benchmark']['posicionamento'],
            'recomendacao_principal': analysis['recomendacoes'][0]['acao'] if analysis['recomendacoes'] else 'N/A'
        }
