# 📊 **EXPLICAÇÃO EXECUTIVA: COMO CADA COLUNA É CALCULADA**

## 🎯 **RESUMO EXECUTIVO**

O sistema analisa **86 rotas de frete** automaticamente e calcula **8 colunas** que mostram exatamente onde estão as oportunidades de economia. Cada coluna é calculada sem intervenção manual, transformando dados brutos em insights acionáveis.

---

## 📋 **COLUNAS DO RELATÓRIO - EXPLICAÇÃO SIMPLES**

### **🏗️ COLUNAS DE IDENTIFICAÇÃO**

#### **1️⃣ CENTRO_ORIGEM**

- **O que é:** De onde sai o frete
- **Como é calculado:** Nome do centro de origem
- **Exemplo:** JOÃO MONLEVADE
- **Complexidade:** Baixa - apenas nome da origem

#### **2️⃣ SUB_CATEGORIA**

- **O que é:** Região de destino
- **Como é calculado:** Micro-região ou município de destino
- **Exemplo:** SABARÁ, CONTAGEM, ITAÚNA
- **Complexidade:** Baixa - apenas nome do destino

#### **3️⃣ ROTA_ESPECIFICA**

- **O que é:** Rota completa (origem → destino)
- **Como é calculado:** Centro Origem + "→" + Sub-categoria
- **Exemplo:** JOÃO MONLEVADE-SABARÁ
- **Complexidade:** Baixa - concatenação de nomes

---

### **💰 COLUNAS DE VOLUME E CUSTO**

#### **4️⃣ VOLUME_TON**

- **O que é:** Quantas toneladas foram transportadas
- **Como é calculado:** Soma de todas as viagens da rota no período
- **Fórmula:** `Soma(Volume de cada viagem)`
- **Exemplo prático:**
  ```
  Viagem 1: 5,000 ton
  Viagem 2: 4,500 ton
  Viagem 3: 6,200 ton
  Viagem 4: 4,800 ton
  Viagem 5: 3,900 ton
  Viagem 6: 3,720.5 ton
  TOTAL: 32,120.5 ton
  ```
- **Complexidade:** Média - soma simples de valores

#### **5️⃣ FR_GERAL_BRL_TON**

- **O que é:** Quanto custa transportar 1 tonelada
- **Como é calculado:** Custo total ÷ Volume total
- **Fórmula:** `Custo Total ÷ Volume Total`
- **Exemplo prático:**
  ```
  Custo total: R$ 1,234,567.89
  Volume total: 32,120.5 ton
  Preço por ton: 1,234,567.89 ÷ 32,120.5 = R$ 38.43
  ```
- **Complexidade:** Média - divisão simples

#### **6️⃣ FR_GERAL_BRL_TON_KM**

- **O que é:** Quanto custa transportar 1 tonelada por 1 km
- **Como é calculado:** Custo total ÷ (Volume total × Distância média)
- **Fórmula:** `Custo Total ÷ (Volume Total × Distância Média)`
- **Exemplo prático:**
  ```
  Custo total: R$ 1,234,567.89
  Volume total: 32,120.5 ton
  Distância média: 45 km
  Preço por ton/km: 1,234,567.89 ÷ (32,120.5 × 45) = R$ 0.8547
  ```
- **Complexidade:** Média - divisão com multiplicação

---

### **🎯 COLUNAS DE OPORTUNIDADE**

#### **7️⃣ OPORT_BRL_TON_KM**

- **O que é:** Quanto você pode economizar ou ganhar por ton/km
- **Como é calculado:** Preço atual da rota - Preço médio de referência
- **Fórmula:** `Preço Atual da Rota - Preço de Referência (Benchmark)`
- **Exemplo prático:**
  ```
  Preço atual da rota: R$ 0.8547 por ton/km
  Preço de referência (média): R$ 0.5349 por ton/km
  Oportunidade: 0.8547 - 0.5349 = R$ 0.3198 por ton/km
  ```
- **Interpretação:**
  - **Valor positivo:** Rota está mais cara → **OPORTUNIDADE DE REDUÇÃO**
  - **Valor negativo:** Rota está mais barata → **OPORTUNIDADE DE AUMENTO**
- **Complexidade:** Média - subtração simples

#### **8️⃣ AÇÃO**

- **O que é:** O que fazer com essa rota
- **Como é calculado:** Classificação automática baseada na oportunidade
- **Regras automáticas:**
  - **Redução:** Se oportunidade > R$ 0.01 por ton/km
  - **Aumento:** Se oportunidade < -R$ 0.01 por ton/km
  - **Manter:** Se oportunidade entre -R$ 0.01 e +R$ 0.01
- **Complexidade:** Baixa - regras condicionais simples

---

## 🏗️ **COMO O SISTEMA FUNCIONA - PASSO A PASSO**

### **📊 ETAPA 1: COLETA DE DADOS**

```
Período: Maio, Junho e Julho 2025
Total: 18,521 viagens de frete
Filtro: Apenas viagens válidas (volume > 0, distância > 0)
Resultado: 14,663 viagens válidas
```

### **🔢 ETAPA 2: CÁLCULOS BÁSICOS**

```
Para cada viagem:
- Preço por ton/km = Custo ÷ (Volume × Distância)
```

### **📈 ETAPA 3: AGREGAÇÃO POR ROTA**

```
Agrupa todas as viagens de cada rota:
- Soma os volumes
- Soma os custos  
- Calcula distância média
- Calcula preço médio por ton/km
```

### **🎯 ETAPA 4: CÁLCULO DO BENCHMARK**

```
Filtra rotas com volume ≥ 200 ton (47 rotas)
Calcula preço médio por micro-região
Este é o "preço de referência" para comparação
```

### **💡 ETAPA 5: IDENTIFICAÇÃO DE OPORTUNIDADES**

```
Para cada rota:
- Compara preço atual com preço de referência
- Calcula a diferença (oportunidade)
- Classifica automaticamente a ação
```

---

## 📊 **EXEMPLO COMPLETO: ROTA SABARÁ**

| Coluna                        | Valor                   | Explicação                  | Complexidade |
| ----------------------------- | ----------------------- | ----------------------------- | ------------ |
| **CENTRO_ORIGEM**       | JOÃO MONLEVADE         | De onde sai o frete           | Baixa        |
| **SUB_CATEGORIA**       | SABARÁ                 | Para onde vai o frete         | Baixa        |
| **ROTA_ESPECIFICA**     | JOÃO MONLEVADE-SABARÁ | Rota completa                 | Baixa        |
| **VOLUME_TON**          | 32,120.5                | Soma de todas as viagens      | Média       |
| **FR_GERAL_BRL_TON**    | 38.43                   | R$ 38.43 por tonelada         | Média       |
| **FR_GERAL_BRL_TON_KM** | 0.8547                  | R$ 0.8547 por ton/km          | Média       |
| **OPORT_BRL_TON_KM**    | 0.3198                  | Economia potencial por ton/km | Média       |
| **AÇÃO**              | Redução               | O que fazer com essa rota     | Baixa        |

---

## 💰 **IMPACTO FINANCEIRO - EXEMPLO PRÁTICO**

### **🎯 ECONOMIA POTENCIAL:**

```
Rota: JOÃO MONLEVADE-SABARÁ
Oportunidade: R$ 0.3198 por ton/km
Volume anual: 128,482 ton (4 trimestres)
Economia potencial: 128,482 × 0.3198 = R$ 41,087 por ano
```

### **📈 RESULTADO FINAL:**

- **Total de rotas analisadas:** 86
- **Oportunidades de redução:** 10 rotas
- **Oportunidades de aumento:** 76 rotas
- **Economia total identificada:** Calculada automaticamente

---

## 🔧 **CONFIGURAÇÕES TÉCNICAS**

### **⚙️ PARÂMETROS DO SISTEMA:**

- **Volume mínimo para benchmark:** 200 toneladas
- **Período de análise:** Trimestral (Maio-Junho-Julho 2025)
- **Filtro de qualidade:** Apenas rotas com volume significativo
- **Agrupamento:** Por micro-região de origem

### **📊 MÉTRICAS MONITORADAS:**

- **Preço por tonelada-quilômetro**
- **Volume total por rota**
- **Custo total por rota**
- **Distância média por rota**
- **Oportunidade de redução/aumento**

---

## 🎯 **VANTAGENS DO SISTEMA**

### **✅ BENEFÍCIOS IMEDIATOS:**

- **Cálculos automáticos:** Sem intervenção manual
- **Decisões baseadas em dados:** Ao invés de intuição
- **Identificação automática:** De oportunidades de economia
- **Classificação automática:** De ações recomendadas

### **🚀 IMPACTO FINANCEIRO:**

- **Redução de custos:** Identificada automaticamente
- **Otimização de preços:** Baseada no mercado real
- **Negociação estratégica:** Com transportadoras
- **ROI imediato:** Na implementação

---

## 📋 **RESUMO EXECUTIVO**

### **🎯 O QUE O SISTEMA FAZ:**

1. **Analisa** todas as rotas de frete automaticamente
2. **Calcula** custos por tonelada e por quilômetro
3. **Compara** com preços de referência da região
4. **Identifica** onde há oportunidades de economia
5. **Recomenda** ações específicas para cada rota

### **💡 COMO FUNCIONA:**

- **8 colunas** calculadas automaticamente
- **Fórmulas simples** de matemática básica
- **Comparações automáticas** com benchmarks
- **Classificações automáticas** de ações

### **💰 RESULTADO:**

- **Economias identificadas** automaticamente
- **Decisões estratégicas** baseadas em dados
- **ROI imediato** na implementação
- **Processo 100% automatizado**

---

## 🚀 **PRÓXIMOS PASSOS**

### **1️⃣ IMPLEMENTAÇÃO IMEDIATA:**

- **Aprovar sistema** para uso operacional
- **Treinar equipe** de logística
- **Implementar monitoramento** contínuo

### **2️⃣ EXPANSÃO:**

- **Incluir mais centros** de origem
- **Análise mensal** ao invés de trimestral
- **Dashboard executivo** com KPIs

### **3️⃣ OTIMIZAÇÃO:**

- **Ajustar parâmetros** baseado nos resultados
- **Integrar com sistema** de negociação
- **Automatizar ações** baseadas nas oportunidades

---

## 📞 **CONTATO E SUPORTE**

**Equipe de Desenvolvimento:** Disponível para demonstrações
**Documentação Técnica:** Completa e atualizada
**Suporte Operacional:** 24/7 para questões críticas

---

*Documento gerado automaticamente pelo Sistema de Análise Hierárquica*
*Data: Agosto 2025*
*Versão: 2.0 - Explicação Executiva das Colunas*
