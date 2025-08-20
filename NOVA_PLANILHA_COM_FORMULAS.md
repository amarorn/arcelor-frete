# 📊 **NOVA PLANILHA COM FÓRMULAS - ANÁLISE HIERÁRQUICA**

## 🎯 **RESUMO EXECUTIVO**

Foi criada uma **nova planilha completa** que inclui **todas as fórmulas de cálculo** utilizadas no sistema de análise hierárquica. A planilha contém **duas abas** com informações detalhadas para executivos e equipes técnicas.

---

## 📁 **ARQUIVO GERADO**

### **🏷️ NOME DO ARQUIVO:**
```
dashboard_hierarquico_maio_junho_julho_2025_COM_FORMULAS.xlsx
```

### **📊 LOCALIZAÇÃO:**
```
outputs/dashboard_hierarquico_maio_junho_julho_2025_COM_FORMULAS.xlsx
```

---

## 🔍 **ESTRUTURA DA PLANILHA**

### **📋 ABA 1: "Analise_Hierarquica"**
**Conteúdo:** Dados completos das 86 rotas analisadas + fórmulas de cálculo

**Colunas incluídas:**
1. **Colunas de dados originais (8 colunas):**
   - Centro_Origem
   - Sub_Categoria
   - Rota_Especifica
   - Volume_TON
   - Fr_Geral_BRL_TON
   - Fr_Geral_BRL_TON_KM
   - Oport_BRL_TON_KM
   - Acao

2. **Colunas de fórmulas (5 colunas):**
   - Formula_Volume_TON
   - Formula_Fr_Geral_BRL_TON
   - Formula_Fr_Geral_BRL_TON_KM
   - Formula_Oport_BRL_TON_KM
   - Formula_Acao

3. **Colunas de explicações (5 colunas):**
   - Explicacao_Volume_TON
   - Explicacao_Fr_Geral_BRL_TON
   - Explicacao_Fr_Geral_BRL_TON_KM
   - Explicacao_Oport_BRL_TON_KM
   - Explicacao_Acao

4. **Colunas de referência (3 colunas):**
   - Preco_Referencia_Microregiao
   - Volume_Minimo_Benchmark
   - Periodo_Analise

---

### **📋 ABA 2: "FORMULAS_CALCULO"**
**Conteúdo:** Tabela completa com todas as fórmulas e explicações técnicas

**Colunas da aba de fórmulas:**
1. **Coluna** - Nome da coluna
2. **Descricao** - O que representa
3. **Formula_Excel** - Fórmula em formato Excel
4. **Formula_Matematica** - Fórmula matemática
5. **Exemplo_Calculo** - Exemplo prático com números
6. **Unidade** - Unidade de medida
7. **Complexidade** - Nível de complexidade (Baixa/Média)

---

## 🧮 **FÓRMULAS INCLUÍDAS**

### **1️⃣ VOLUME_TON**
- **Fórmula Excel:** `=SOMA(Volume de todas as viagens da rota)`
- **Fórmula Matemática:** `Σ(Volume_viagem_i)`
- **Exemplo:** `5,000 + 4,500 + 6,200 + 4,800 + 3,900 + 3,720.5 = 32,120.5 ton`

### **2️⃣ FR_GERAL_BRL_TON**
- **Fórmula Excel:** `=Custo_Total_Rota / Volume_TON`
- **Fórmula Matemática:** `Custo_Total / Volume_Total`
- **Exemplo:** `1,234,567.89 ÷ 32,120.5 = R$ 38.43`

### **3️⃣ FR_GERAL_BRL_TON_KM**
- **Fórmula Excel:** `=Custo_Total_Rota / (Volume_TON × Distancia_Media)`
- **Fórmula Matemática:** `Custo_Total / (Volume_Total × Distância_Média)`
- **Exemplo:** `1,234,567.89 ÷ (32,120.5 × 45) = R$ 0.8547`

### **4️⃣ OPORT_BRL_TON_KM**
- **Fórmula Excel:** `=Fr_Geral_BRL_TON_KM - Preco_Medio_Microregiao`
- **Fórmula Matemática:** `Preço_Atual - Preço_Referência`
- **Exemplo:** `0.8547 - 0.5349 = R$ 0.3198`

### **5️⃣ AÇÃO**
- **Fórmula Excel:** `=SE(Oport_BRL_TON_KM > 0.01; "Redução"; SE(Oport_BRL_TON_KM < -0.01; "Aumento"; "Manter"))`
- **Fórmula Matemática:** `Classificação condicional`
- **Exemplo:** `0.3198 > 0.01 → "Redução"`

---

## 💡 **BENEFÍCIOS DA NOVA PLANILHA**

### **✅ PARA EXECUTIVOS:**
- **Transparência total** dos cálculos
- **Auditoria facilitada** dos resultados
- **Compreensão clara** de como as oportunidades são identificadas
- **Decisões baseadas** em metodologia transparente

### **🔧 PARA EQUIPES TÉCNICAS:**
- **Fórmulas prontas** para implementação em outros sistemas
- **Documentação completa** de todos os cálculos
- **Base para desenvolvimento** de novas funcionalidades
- **Padrão de qualidade** para futuras análises

### **📊 PARA AUDITORIA:**
- **Rastreabilidade completa** dos cálculos
- **Validação independente** dos resultados
- **Documentação para compliance** e auditorias
- **Histórico de metodologia** utilizada

---

## 🚀 **COMO UTILIZAR A PLANILHA**

### **📋 PASSO 1: ABERTURA**
1. Abrir o arquivo `dashboard_hierarquico_maio_junho_julho_2025_COM_FORMULAS.xlsx`
2. Verificar as duas abas disponíveis

### **📊 PASSO 2: ANÁLISE DOS DADOS**
1. **Aba "Analise_Hierarquica":** Dados completos + fórmulas inline
2. **Aba "FORMULAS_CALCULO":** Referência técnica completa

### **🧮 PASSO 3: VALIDAÇÃO DOS CÁLCULOS**
1. Verificar as fórmulas em cada coluna
2. Validar os exemplos de cálculo
3. Confirmar a lógica de classificação das ações

### **📈 PASSO 4: IMPLEMENTAÇÃO**
1. Usar as fórmulas como base para outros sistemas
2. Adaptar para diferentes períodos de análise
3. Implementar em dashboards executivos

---

## 📊 **RESULTADOS INCLUÍDOS**

### **🎯 ANÁLISE COMPLETA:**
- **Total de rotas:** 86 rotas analisadas
- **Período:** Maio, Junho e Julho 2025
- **Volume mínimo:** 200 toneladas para benchmark
- **Rotas para benchmark:** 47 rotas qualificadas

### **💡 OPORTUNIDADES IDENTIFICADAS:**
- **Oportunidades de redução:** 10 rotas
- **Oportunidades de aumento:** 76 rotas
- **Rotas para manter:** 0 rotas

### **🏆 TOP 5 OPORTUNIDADES DE REDUÇÃO:**
1. **JOÃO MONLEVADE-SABARÁ:** R$ 0.3255/ton/km
2. **JOÃO MONLEVADE-SANTA LUZIA:** R$ 0.2540/ton/km
3. **JOÃO MONLEVADE-SETE LAGOAS:** R$ 0.2473/ton/km
4. **JOÃO MONLEVADE-CONFINS:** R$ 0.2206/ton/km
5. **JOÃO MONLEVADE-GOVERNADOR VALADARES:** R$ 0.2045/ton/km

---

## 🔧 **CONFIGURAÇÕES TÉCNICAS**

### **⚙️ PARÂMETROS UTILIZADOS:**
- **Volume mínimo para benchmark:** 200 toneladas
- **Período de análise:** Trimestral (Maio-Junho-Julho 2025)
- **Filtro de qualidade:** Apenas rotas com volume significativo
- **Agrupamento:** Por micro-região de origem

### **📊 MÉTRICAS CALCULADAS:**
- **Preço por tonelada-quilômetro**
- **Volume total por rota**
- **Custo total por rota**
- **Distância média por rota**
- **Oportunidade de redução/aumento**

---

## 📞 **PRÓXIMOS PASSOS**

### **1️⃣ IMPLEMENTAÇÃO IMEDIATA:**
- **Revisar planilha** com equipe executiva
- **Validar fórmulas** com equipe técnica
- **Implementar monitoramento** contínuo

### **2️⃣ EXPANSÃO:**
- **Adaptar fórmulas** para outros períodos
- **Incluir mais centros** de origem
- **Criar dashboard** executivo interativo

### **3️⃣ OTIMIZAÇÃO:**
- **Ajustar parâmetros** baseado nos resultados
- **Integrar com sistemas** existentes
- **Automatizar ações** baseadas nas oportunidades

---

## 🎯 **CONCLUSÃO**

A nova planilha com fórmulas representa um **marco na transparência** do sistema de análise hierárquica. Agora executivos e equipes técnicas têm acesso completo a:

- **Todas as fórmulas** utilizadas nos cálculos
- **Exemplos práticos** com números reais
- **Explicações técnicas** de cada coluna
- **Metodologia completa** de análise

**Esta é a ferramenta definitiva para compreensão, validação e implementação do sistema de benchmark inteligente para fretes!** 🎯✨

---

*Documento gerado automaticamente pelo Sistema de Análise Hierárquica*
*Data: Agosto 2025*
*Versão: 2.0 - Nova Planilha com Fórmulas*
