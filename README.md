# Análise de Rotas e transportadoras - ArcelorMittal

## 🚀 **FASE 2 IMPLEMENTADA: Seleção Inteligente de Transportadoras**

### **📊 Arquitetura Atual (Fase 2)**
```
├── notebooks/
│   ├── fase2_pipeline.py           # 🆕 Pipeline principal da Fase 2
│   ├── fase1_pipeline.py           # Pipeline da Fase 1 (Frete + ML)
│   ├── 00_main_pipeline.py        # Pipeline original (Bronze → Silver → Gold)
│   ├── bronze/
│   │   └── 01_api_ingestion.py    # Ingestão bruta da API
│   ├── silver/
│   │   └── 02_data_cleansing.py   # Limpeza, deduplicação, padronização
│   └── gold/
│       └── 03_data_aggregation.py # Agregações de negócio
├── utils/
│   ├── adapters/
│   │   ├── polars_adapter.py      # Adaptador Polars (10-100x mais rápido)
│   │   └── excel_adapter.py       # Adaptador Excel original
│   └── ml/
│       ├── frete_price_predictor.py # Modelo ML para preços (Fase 1)
│       └── transportadora_selector.py # 🆕 Seletor inteligente de transportadoras (Fase 2)
├── config/
│   └── config_api_real.json       # Credenciais / parâmetros da API
├── outputs/                        # Resultados das Fases 1 e 2
├── models/                         # Modelos ML treinados
└── README.md
```

### **🎯 O que foi implementado na Fase 2:**

#### **1. 🤖 Modelo de Seleção Inteligente de Transportadoras:**
- **Random Forest Classifier** para recomendação de transportadoras
- **Múltiplos critérios de avaliação** com pesos configuráveis:
  - **Preço** (25%): Análise de custos e competitividade
  - **Performance** (20%): Histórico de entregas e volume
  - **Confiabilidade** (25%): Consistência de preços e estabilidade
  - **Capacidade** (15%): Capacidade operacional e cobertura
  - **Custo-Benefício** (15%): Relação entre custo e valor agregado

#### **2. 📊 Sistema de Scoring Multi-dimensional:**
- **Performance Score**: Baseado em histórico de preços, volume e frequência
- **Confiabilidade Score**: Análise de consistência e variabilidade de preços
- **Capacidade Score**: Avaliação de capacidade operacional e cobertura geográfica
- **Custo-Benefício Score**: Análise de relação custo vs. valor agregado

#### **3. 🔗 Pipeline Integrado Fase 1 + Fase 2:**
- **Combinação automática** de previsões de frete com seleção de transportadora
- **Análise integrada** que considera tanto preço quanto qualidade do serviço
- **Relatórios unificados** com economia potencial e recomendações

#### **4. 📈 Relatórios Avançados:**
- **Análise por transportadora**: Performance e confiabilidade individual
- **Análise por rota**: Recomendações específicas para cada trajeto
- **Métricas de confiança**: Score de confiança para cada recomendação
- **Economia potencial integrada**: Considerando preço + qualidade

### **🎯 O que foi implementado na Fase 1:**

#### **1. 🚀 Migração para Polars:**
- **Substituição do pandas** por Polars para processamento
- **10-100x melhor performance** para datasets grandes
- **Processamento paralelo** nativo
- **API similar ao pandas** para migração fácil

#### **2. 🤖 Primeiro Modelo de Machine Learning:**
- **Random Forest** para previsão de preços de frete
- **Features principais**: volume, distância, modal, tipo de rodovia, sazonalidade
- **Métricas de performance**: R², MAE, RMSE
- **Cross-validation** para robustez

#### **3. 🆕 Modelo Baseline Avançado (NOVO):**
- **Múltiplos algoritmos** para comparação de performance
- **Feature Engineering Avançado** com 6 grupos de features
- **8 modelos baseline**: Linear, Ridge, Lasso, Random Forest, Gradient Boosting, Decision Tree, KNN, SVR
- **Otimização automática** de hiperparâmetros
- **Análise de importância** das features criadas

#### **4. 📊 Análise de Otimização:**
- **Identificação automática** de oportunidades de economia
- **Ranking de rotas** para otimização
- **Ranking de transportadoras** para negociação
- **Cálculo de economia potencial** em reais

### **⚡ Performance Esperada:**

#### **Fase 1 (Frete):**
- **Leitura de dados**: 5-10x mais rápido
- **Processamento**: 10-50x mais rápido
- **Análises**: 20-100x mais rápido
- **Precisão do modelo**: R² > 0.8 esperado

#### **Modelo Baseline (NOVO):**
- **Feature Engineering**: 50+ features criadas automaticamente
- **Múltiplos algoritmos**: 8 modelos treinados e comparados
- **Otimização automática**: Hiperparâmetros otimizados via Grid Search
- **Performance esperada**: R² > 0.85 com features avançadas

#### **Fase 2 (Transportadora):**
- **Seleção inteligente**: Acurácia > 0.85 esperada
- **Análise multi-critério**: 5 dimensões de avaliação
- **Recomendações personalizadas**: Por rota e características
- **Score de confiança**: Métrica de qualidade das recomendações

### **🔧 Como executar as Fases:**

#### **Fase 1 - Modelo de Frete:**
```bash
# Ativar ambiente virtual
source venv/bin/activate

# Executar pipeline da Fase 1
python notebooks/fase1_pipeline.py
```

#### **Modelo Baseline (NOVO):**
```bash
# Executar pipeline baseline com feature engineering
python notebooks/baseline_pipeline.py

# Apenas análise de features
python notebooks/baseline_pipeline.py --analysis
```

#### **Fase 2 - Seleção de Transportadora:**
```bash
# Executar pipeline da Fase 2 (inclui Fase 1)
python notebooks/fase2_pipeline.py
```

#### **Pipeline Completo (Fase 1 + Fase 2):**
```bash
# Executar pipeline integrado
python notebooks/fase2_pipeline.py
```

### **🧪 Como executar os testes:**

```bash
# Executar todos os testes
python run_tests.py

# Ou executar testes específicos
python -m pytest tests/ -v

# Testes com cobertura
python -m pytest tests/ --cov=utils --cov=notebooks --cov-report=html

# Testes por categoria
python -m pytest tests/ -m unit      # Testes unitários
python -m pytest tests/ -m integration  # Testes de integração

# Testes específicos da Fase 2
python -m pytest tests/test_transportadora_selector.py -v
```

### **📁 Outputs gerados:**

#### **Fase 1:**
- `outputs/fase1_metricas_rotas.xlsx` - Métricas por rota
- `outputs/fase1_metricas_transportadoras.xlsx` - Métricas por transportadora  
- `outputs/fase1_relatorio_otimizacao.json` - Relatório completo de otimização
- `models/frete_predictor.joblib` - Modelo ML treinado

#### **Modelo Baseline (NOVO):**
- `outputs/baseline_relatorio.json` - Relatório completo dos modelos baseline
- `outputs/baseline_dados_com_features.xlsx` - Dados com todas as features criadas
- `outputs/baseline_ranking_modelos.xlsx` - Ranking de performance dos modelos
- `models/baseline_model/` - Modelo otimizado e preprocessador

#### **Fase 2:**
- `outputs/fase2_metricas_rotas.xlsx` - Métricas por rota (Fase 2)
- `outputs/fase2_metricas_transportadoras.xlsx` - Métricas por transportadora (Fase 2)
- `outputs/fase2_dados_integrados.xlsx` - Dados com previsões integradas
- `outputs/fase2_relatorio_frete.json` - Relatório de frete (Fase 1)
- `outputs/fase2_relatorio_transportadora.json` - Relatório de seleção (Fase 2)
- `outputs/fase2_relatorio_integrado.json` - Relatório completo integrado
- `models/transportadora_selector.joblib` - Modelo de seleção treinado

---

## 📐 **Estrutura Original (Medallion Architecture)**

O projeto segue o padrão **Bronze → Silver → Gold**:

1. **Bronze** – captura de dados brutos sem transformação.
2. **Silver** – dados limpos e conformes.
3. **Gold** – tabelas agregadas prontas para BI.

Execute localmente:
```bash
python notebooks/00_main_pipeline.py --config config/config_api_real.json
```

Em Databricks, vincule cada notebook como tarefa em um Job ou utilize Databricks Asset Bundles.
