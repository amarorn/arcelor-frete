# Análise de Oportunidades de Redução de Preços

## Visão Geral

Esta funcionalidade analisa oportunidades de redução de preços de frete baseada no preço médio por TKU (Tonelada-Quilômetro) por micro-região. O sistema identifica rotas onde há potencial para redução de custos comparando o preço atual com o preço médio da micro-região.

## Funcionalidades

### 1. Cálculo de Preços por Micro-região
- **MicroRegionPriceCalculator**: Calcula o preço médio por TKU para cada micro-região
- Mapeamento automático de localizações para micro-regiões
- Estatísticas por micro-região (média, desvio padrão, quantidade de rotas)

### 2. Análise de Oportunidades
- **OpportunityAnalyzer**: Identifica oportunidades de redução de preços
- Threshold configurável para determinar ações
- Classificação automática: Redução, Aumento ou Manter

### 3. Geração de Relatórios
- Relatório em Excel com análise completa
- Resumo estatístico das oportunidades
- Formatação similar à tabela de referência

## Estrutura do Código

```
utils/ml/opportunity_analyzer.py
├── MicroRegionPriceCalculator
│   ├── _extract_microregion()
│   └── calculate_average_price_by_microregion()
└── OpportunityAnalyzer
    ├── _calculate_opportunity()
    ├── _determine_action()
    └── analyze_reduction_opportunities()
```

## Como Usar

### 1. Execução Rápida
```bash
python analise_oportunidades_reducao.py
```

### 2. Uso Programático
```python
from utils.ml import OpportunityAnalyzer

# Inicializar analisador
analyzer = OpportunityAnalyzer(
    threshold_opportunity=0.02,  # 2 centavos por TON/KM
    min_volume_threshold=100.0   # Volume mínimo
)

# Executar análise
resultado = analyzer.analyze_reduction_opportunities(df)
```

### 3. Configuração Personalizada
```python
# Threshold personalizado para oportunidades
analyzer = OpportunityAnalyzer(threshold_opportunity=0.05)

# Volume mínimo personalizado
analyzer = OpportunityAnalyzer(min_volume_threshold=500.0)
```

## Estrutura dos Dados de Entrada

O DataFrame deve conter as seguintes colunas:
- `centro_origem`: Localização de origem
- `volume_ton`: Volume em toneladas
- `distancia_km`: Distância em quilômetros
- `custo_sup_tku`: Custo de suprimento em TKU

## Estrutura dos Dados de Saída

O relatório contém:
- **Centro Origem**: Localização analisada
- **Volume (TON)**: Volume em toneladas
- **Distância (KM)**: Distância em quilômetros
- **Custo Sup (TKU)**: Custo atual
- **04.01 - Média MicroRegião - Preço SUP (BRL/TON/KM)**: Preço médio da micro-região
- **Oport. (BRL/TON/KM)**: Oportunidade calculada
- **Ação**: Recomendação (Redução/Aumento/Manter)

## Mapeamento de Micro-regiões

### João Monlevade
- SABARÁ, CONTAGEM, SANTA LUZIA, CONFINS
- NOVA LIMA, PEDRO LEOPOLDO, BELO HORIZONTE
- BETIM, SÃO JOAQUIM DE BICAS, VESPASIANO
- USINA MONLEVADE

### Belo Horizonte
- BETIM, SÃO JOAQUIM DE BICAS, VESPASIANO

### Itabira
- Localizações específicas de Itabira

## Configurações

### Threshold de Oportunidade
- **Padrão**: 0.02 BRL/TON/KM (2 centavos)
- **Redução**: Oportunidade > threshold
- **Aumento**: Oportunidade < -threshold
- **Manter**: Entre -threshold e +threshold

### Volume Mínimo
- **Padrão**: 100.0 toneladas
- Rotas com volume abaixo do threshold são ignoradas

## Exemplos de Uso

### Cenário 1: Análise Padrão
```python
analyzer = OpportunityAnalyzer()
resultado = analyzer.analyze_reduction_opportunities(dados)
```

### Cenário 2: Threshold Conservador
```python
analyzer = OpportunityAnalyzer(threshold_opportunity=0.05)
resultado = analyzer.analyze_reduction_opportunities(dados)
```

### Cenário 3: Volume Mínimo Alto
```python
analyzer = OpportunityAnalyzer(min_volume_threshold=1000.0)
resultado = analyzer.analyze_reduction_opportunities(dados)
```

## Testes

Execute os testes com:
```bash
python -m pytest tests/test_opportunity_analyzer.py -v
```

## Saídas

### 1. Console
- Logs detalhados do processo
- Resumo da análise
- Top 5 oportunidades de redução

### 2. Arquivo Excel
- **Sheet 1**: Análise completa das oportunidades
- **Sheet 2**: Resumo estatístico

### 3. Localização dos Arquivos
- **Relatório**: `outputs/analise_oportunidades_reducao.xlsx`
- **Logs**: Console/terminal

## Dependências

- pandas
- numpy
- openpyxl (para Excel)
- logging (built-in)

## Contribuição

Para adicionar novas funcionalidades:
1. Implemente na classe apropriada
2. Adicione testes unitários
3. Atualize a documentação
4. Execute todos os testes

## Suporte

Em caso de dúvidas ou problemas:
1. Verifique os logs de execução
2. Execute os testes unitários
3. Verifique a estrutura dos dados de entrada
4. Consulte a documentação das classes
