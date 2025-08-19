# 🧠 TKUTrendAnalyzer - Sistema de Análise de Tendências Temporais Inteligente

## 📋 Visão Geral

O **TKUTrendAnalyzer** é um sistema inteligente que analisa tendências de preços de frete (TKU - Tonelada Quilômetro Útil) por rota específica, fornecendo insights valiosos para o operador logístico.

## 🎯 O que o Sistema Faz

### **1. Análise de Tendências Temporais**
- **3 meses**: Tendência imediata e ações urgentes
- **6 meses**: Tendência de médio prazo e estratégias
- **12 meses**: Tendência de longo prazo e planejamento

### **2. Análise de Sazonalidade**
- Identifica padrões mensais de preços
- Detecta alta e baixa temporada
- Sugere otimizações baseadas em sazonalidade

### **3. Benchmark de Mercado**
- Compara rota específica com micro-região
- Posiciona a rota no mercado (Excelente, Bom, Regular, Crítico)
- Calcula economia potencial

### **4. Recomendações Inteligentes**
- Ações baseadas em tendências reais
- Priorização por urgência e impacto
- Estimativas de economia esperada

## 🚀 Como Usar

### **1. Importação e Inicialização**

```python
from utils.ml.tku_trend_analyzer import TKUTrendAnalyzer

# Inicializar analisador
analyzer = TKUTrendAnalyzer()
```

### **2. Análise Completa de uma Rota**

```python
# Analisar tendências de TKU para uma rota específica
analysis = analyzer.analyze_route_tku_trends(df_dados, 'SABARÁ')

# Resultado: análise completa com todas as informações
print(f"Status da rota: {analysis['status_geral']}")
print(f"TKU atual: R$ {analysis['tku_atual']:.4f}")
print(f"Variação 3 meses: {analysis['variacao_percentual']:+.1f}%")
```

### **3. Acesso a Análises Específicas**

```python
# Análise de tendências
tendencias = analysis['tendencias']
tendencia_3m = tendencias['3_meses']
print(f"Tendência 3 meses: {tendencia_3m['tendencia']}")
print(f"Velocidade: {tendencia_3m['velocidade']}")
print(f"Confiança: {tendencia_3m['confianca']:.1f}%")

# Análise de sazonalidade
sazonalidade = analysis['sazonalidade']
if sazonalidade['padrao_sazonal_detectado']:
    print(f"Meses de alta: {sazonalidade['meses_alta_temporada']}")
    print(f"Meses de baixa: {sazonalidade['meses_baixa_temporada']}")

# Benchmark de mercado
benchmark = analysis['benchmark']
print(f"Posicionamento: {benchmark['posicionamento']}")
print(f"Economia potencial: R$ {benchmark['economia_potencial']:.4f}")
```

### **4. Recomendações Automáticas**

```python
# Acessar recomendações
recomendacoes = analysis['recomendacoes']

for i, rec in enumerate(recomendacoes, 1):
    print(f"{i}. {rec['acao']}")
    print(f"   • Tipo: {rec['tipo']}")
    print(f"   • Prioridade: {rec['prioridade']}")
    print(f"   • Economia esperada: {rec['economia_esperada']}")
    print(f"   • Prazo: {rec['prazo']}")
    print(f"   • Dificuldade: {rec['dificuldade']}")
```

### **5. Resumo Executivo**

```python
# Obter resumo executivo
summary = analyzer.get_analysis_summary('SABARÁ')

print(f"🛣️  Rota: {summary['rota_id']}")
print(f"🎯 Status: {summary['status_geral']}")
print(f"💰 TKU Atual: R$ {summary['tku_atual']:.4f}")
print(f"📊 Variação: {summary['variacao_percentual']:+.1f}%")
print(f"📈 Tendência 3M: {summary['tendencia_3m']}")
print(f"🏆 Posicionamento: {summary['posicionamento_mercado']}")
print(f"💡 Ação Principal: {summary['recomendacao_principal']}")
```

## 📊 Estrutura dos Dados de Entrada

O DataFrame deve conter as seguintes colunas:

```python
required_columns = [
    'data_faturamento',      # Data da entrega
    'rota_municipio',        # Identificador da rota
    'rota_microregiao',      # Micro-região da rota
    'rota_mesoregiao',       # Meso-região da rota
    'microregiao_origem',    # Micro-região de origem
    'volume_ton',            # Volume em toneladas
    'distancia_km',          # Distância em quilômetros
    'frete_brl',             # Valor do frete em reais
    'modal',                 # Modal de transporte
    'tipo_rodovia',          # Tipo de rodovia
    'tipo_veiculo'           # Tipo de veículo
]
```

## 🔍 Exemplos de Uso Prático

### **Exemplo 1: Análise de Rota Crítica**

```python
# Analisar rota com tendência de alta
analysis = analyzer.analyze_route_tku_trends(df, 'SABARÁ')

if analysis['status_geral'].startswith('CRÍTICO'):
    print("🚨 ATENÇÃO: Rota crítica detectada!")
    
    # Verificar tendências
    tendencia_3m = analysis['tendencias']['3_meses']
    if tendencia_3m['tendencia'] == 'SUBINDO':
        print(f"📈 TKU subindo {tendencia_3m['velocidade']} nos últimos 3 meses")
    
    # Verificar recomendações urgentes
    recomendacoes_urgentes = [r for r in analysis['recomendacoes'] 
                             if r['tipo'] == 'URGENTE']
    
    for rec in recomendacoes_urgentes:
        print(f"⚡ Ação urgente: {rec['acao']}")
        print(f"   Prazo: {rec['prazo']}")
        print(f"   Economia: {rec['economia_esperada']}")
```

### **Exemplo 2: Análise de Sazonalidade**

```python
# Analisar padrões sazonais
analysis = analyzer.analyze_route_tku_trends(df, 'SABARÁ')
sazonalidade = analysis['sazonalidade']

if sazonalidade['padrao_sazonal_detectado']:
    print("🌍 Padrão sazonal detectado!")
    
    # Planejar para baixa temporada
    if sazonalidade['meses_baixa_temporada']:
        print(f"❄️  Meses de baixa temporada: {sazonalidade['meses_baixa_temporada']}")
        print("💡 Recomendação: Antecipar entregas para estes meses")
    
    # Evitar alta temporada
    if sazonalidade['meses_alta_temporada']:
        print(f"🔥 Meses de alta temporada: {sazonalidade['meses_alta_temporada']}")
        print("💡 Recomendação: Negociar preços ou evitar estes meses")
```

### **Exemplo 3: Comparação com Benchmark**

```python
# Analisar posicionamento no mercado
analysis = analyzer.analyze_route_tku_trends(df, 'SABARÁ')
benchmark = analysis['benchmark']

print(f"🏆 Posicionamento: {benchmark['posicionamento']}")

if benchmark['posicionamento'].startswith('CRÍTICO'):
    print("🚨 Rota muito acima do benchmark!")
    print(f"💰 Economia potencial: R$ {benchmark['economia_potencial']:.4f}/TON.KM")
    print(f"🎯 Ação necessária: {benchmark['acao_necessaria']}")
    
elif benchmark['posicionamento'].startswith('EXCELENTE'):
    print("✅ Rota com excelente posicionamento!")
    print("💡 Manter estratégia atual")
```

## ⚙️ Configurações Avançadas

### **1. Ajustar Thresholds de Tendência**

```python
analyzer = TKUTrendAnalyzer()

# Personalizar thresholds
analyzer.trend_thresholds['stable_threshold'] = 0.03      # 3% = estável
analyzer.trend_thresholds['moderate_threshold'] = 0.10    # 10% = moderado
analyzer.trend_thresholds['fast_threshold'] = 0.20        # 20% = rápido
```

### **2. Ajustar Períodos de Análise**

```python
# Personalizar períodos (em dias)
analyzer.periods['short_term'] = 60    # 2 meses
analyzer.periods['medium_term'] = 120  # 4 meses
analyzer.periods['long_term'] = 240    # 8 meses
```

## 🧪 Testes

### **Executar Testes Unitários**

```bash
# Testar apenas o TKUTrendAnalyzer
python -m pytest tests/test_tku_trend_analyzer.py -v

# Testar com cobertura
python -m pytest tests/test_tku_trend_analyzer.py --cov=utils.ml.tku_trend_analyzer --cov-report=html
```

### **Executar Demonstração**

```bash
# Executar script de exemplo
python exemplo_tku_trend_analyzer.py
```

## 📈 Interpretação dos Resultados

### **Status da Rota**

| Status | Significado | Ação Recomendada |
|--------|-------------|------------------|
| **CRÍTICO** | Ação urgente necessária | Revisar estratégia imediatamente |
| **ATENÇÃO** | Monitorar tendências | Implementar ações preventivas |
| **ESTÁVEL** | Manter estratégia atual | Continuar monitoramento |
| **POSITIVO** | Tendência favorável | Aproveitar oportunidades |

### **Velocidade da Tendência**

| Velocidade | Significado | Urgência |
|------------|-------------|----------|
| **ESTÁVEL** | Variação < 5% | Baixa |
| **LENTO** | Variação 5-15% | Média |
| **MÉDIO** | Variação 15-25% | Alta |
| **RÁPIDO** | Variação > 25% | Crítica |

### **Posicionamento no Mercado**

| Posicionamento | Significado | Economia Potencial |
|----------------|-------------|-------------------|
| **EXCELENTE (Top 25%)** | Muito competitivo | Baixa |
| **BOM (Top 50%)** | Competitivo | Média |
| **REGULAR (Top 75%)** | Melhorável | Alta |
| **CRÍTICO (Top 100%)** | Muito acima | Muito alta |

## 🔧 Funcionalidades Avançadas

### **1. Cache de Análises**

```python
# Verificar se análise está em cache
cached_analysis = analyzer.get_cached_analysis('SABARÁ')
if cached_analysis:
    print("✅ Análise encontrada em cache")
else:
    print("🔄 Executando nova análise...")

# Limpar cache
analyzer.clear_cache()
```

### **2. Análise de Múltiplas Rotas**

```python
# Analisar várias rotas
rotas = ['SABARÁ', 'CONTAGEM', 'SANTA LUZIA', 'BETIM']
analises = {}

for rota in rotas:
    analises[rota] = analyzer.analyze_route_tku_trends(df, rota)

# Comparar rotas
for rota, analise in analises.items():
    if 'erro' not in analise:
        print(f"{rota}: {analise['status_geral']} - TKU: R$ {analise['tku_atual']:.4f}")
```

## 🚨 Tratamento de Erros

O sistema trata automaticamente os seguintes cenários:

- **Dados insuficientes**: Retorna análise com status "SEM DADOS"
- **Rotas inexistentes**: Retorna análise vazia
- **Dados inválidos**: Filtra automaticamente valores problemáticos
- **Erros de cálculo**: Retorna análise com status "ERRO NA ANÁLISE"

## 📊 Exemplo de Saída Completa

```json
{
  "rota_id": "SABARÁ",
  "status_geral": "ATENÇÃO - Monitorar tendência",
  "tku_atual": 0.8542,
  "variacao_percentual": 12.5,
  "tendencias": {
    "3_meses": {
      "tendencia": "SUBINDO",
      "velocidade": "MÉDIO",
      "confianca": 78.5,
      "tku_atual": 0.8542,
      "variacao_percentual": 12.5
    },
    "6_meses": {
      "tendencia": "SUBINDO",
      "velocidade": "LENTO",
      "confianca": 82.1,
      "tku_atual": 0.8234,
      "variacao_percentual": 8.7
    },
    "12_meses": {
      "tendencia": "ESTÁVEL",
      "velocidade": "ESTÁVEL",
      "confianca": 65.3,
      "tku_atual": 0.7891,
      "variacao_percentual": 2.1
    }
  },
  "sazonalidade": {
    "padrao_sazonal_detectado": true,
    "meses_alta_temporada": ["Dezembro", "Janeiro"],
    "meses_baixa_temporada": ["Junho", "Julho"]
  },
  "benchmark": {
    "posicionamento": "REGULAR (Top 75%)",
    "economia_potencial": 0.1234,
    "acao_necessaria": "Negociar com transportadora"
  },
  "recomendacoes": [
    {
      "tipo": "URGENTE",
      "prioridade": 1,
      "acao": "Negociar com transportadora atual",
      "economia_esperada": "5-15%",
      "prazo": "Imediato (próxima entrega)"
    }
  ]
}
```

## 🎯 Casos de Uso Típicos

### **1. Monitoramento Semanal**
- Analisar tendências de todas as rotas críticas
- Identificar mudanças súbitas de preço
- Ajustar estratégias conforme necessário

### **2. Planejamento Mensal**
- Analisar padrões sazonais
- Planejar entregas para baixa temporada
- Negociar contratos com transportadoras

### **3. Revisão Trimestral**
- Avaliar performance das rotas
- Comparar com benchmarks de mercado
- Implementar melhorias estruturais

### **4. Análise de Crise**
- Identificar rotas com tendências críticas
- Implementar ações emergenciais
- Revisar estratégias de logística

## 🔮 Próximos Passos

### **Funcionalidades Planejadas**
- [ ] Integração com Prophet para previsões mais precisas
- [ ] Alertas automáticos por email/Slack
- [ ] Dashboard web interativo
- [ ] Análise de correlação entre rotas
- [ ] Recomendações baseadas em ML

### **Melhorias Técnicas**
- [ ] Otimização de performance para grandes datasets
- [ ] Cache persistente em banco de dados
- [ ] API REST para integração com outros sistemas
- [ ] Suporte a múltiplos formatos de dados

---

## 📞 Suporte

Para dúvidas ou problemas:
1. Verifique os logs de execução
2. Execute os testes unitários
3. Verifique a estrutura dos dados de entrada
4. Consulte esta documentação

**🎉 O TKUTrendAnalyzer está pronto para revolucionar sua análise de tendências de frete!**
