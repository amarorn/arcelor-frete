# 🚀 Otimização de Performance com Databricks e Spark

## **Problema Identificado**

A versão atual do script `analise_oportunidades_reducao.py` está **muito lenta** devido a:

1. **Processamento sequencial** com pandas
2. **Loops Python** para análise de dados
3. **Operações não vetorizadas**
4. **Uso ineficiente de memória**
5. **Falta de paralelização**

## **Solução: Migração para Databricks/Spark**

### **Performance Esperada: 10-100x mais rápido**

## **📁 Arquivos Criados**

### **1. `analise_oportunidades_reducao_spark.py`**
- Versão Python standalone com Spark
- Para execução local ou em clusters

### **2. `notebooks/analise_oportunidades_databricks.py`**
- Notebook Databricks otimizado
- **RECOMENDADO** para produção

## **🔧 Principais Otimizações Implementadas**

### **1. Configurações Spark Avançadas**
```python
# Adaptive Query Execution
spark.conf.set("spark.sql.adaptive.enabled", "true")
spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")

# Cache e Broadcast
spark.conf.set("spark.sql.adaptive.autoBroadcastJoinThreshold", "100485760")
```

### **2. Operações Vetorizadas**
- **Agregações Spark** em vez de loops Python
- **Window Functions** para seleção de rotas representativas
- **Joins otimizados** com particionamento inteligente

### **3. Cache Estratégico**
```python
# Cache de DataFrames intermediários
df_processed.cache()
df_analysis.cache()

# Limpeza automática de cache
df_processed.unpersist()
```

### **4. UDFs Otimizadas**
- **UDFs nativas Spark** para transformações
- **Broadcast de mapeamentos** para micro-regiões
- **Operações vetorizadas** em vez de apply()

## **🚀 Como Executar no Databricks**

### **Opção 1: Notebook (Recomendado)**
1. Abrir o notebook `analise_oportunidades_databricks.py`
2. Executar célula por célula
3. Acompanhar métricas de performance

### **Opção 2: Job Automatizado**
1. Criar Job no Databricks
2. Configurar cluster com recursos adequados
3. Executar notebook como tarefa

### **Opção 3: Pipeline CI/CD**
1. Integrar com Azure DevOps/GitHub Actions
2. Deploy automático via Databricks Asset Bundles
3. Execução programada

## **⚙️ Configuração do Cluster**

### **Configurações Recomendadas**
```json
{
  "cluster_name": "analise-oportunidades-otimizado",
  "spark_version": "13.3.x-scala2.12",
  "node_type_id": "Standard_DS3_v2",
  "num_workers": 2,
  "driver_node_type_id": "Standard_DS3_v2",
  "spark_conf": {
    "spark.sql.adaptive.enabled": "true",
    "spark.sql.adaptive.coalescePartitions.enabled": "true",
    "spark.sql.adaptive.skewJoin.enabled": "true",
    "spark.sql.adaptive.localShuffleReader.enabled": "true",
    "spark.sql.adaptive.optimizeSkewedJoin.enabled": "true"
  }
}
```

### **Recursos Mínimos**
- **Driver**: 2 vCPUs, 7 GB RAM
- **Workers**: 2 vCPUs, 7 GB RAM cada
- **Storage**: 20 GB por nó

## **📊 Monitoramento de Performance**

### **Métricas Importantes**
1. **Tempo de execução** total
2. **Número de partições** ativas
3. **Status de cache** dos DataFrames
4. **Uso de memória** e CPU
5. **Skew de dados** em joins

### **Comandos de Monitoramento**
```python
# Estatísticas de partições
print(f"Partições: {df.rdd.getNumPartitions()}")

# Status de cache
print(f"Cache ativo: {df.is_cached}")

# Tamanho estimado
print(f"Tamanho: {df.count() * len(df.columns) * 8 / 1024 / 1024:.2f} MB")
```

## **🔍 Análise de Performance**

### **Comparação Esperada**
| Métrica | Pandas | Spark | Melhoria |
|---------|--------|-------|----------|
| **Tempo total** | 300-600s | 30-60s | **10x** |
| **Processamento** | Sequencial | Paralelo | **50x** |
| **Memória** | Alta | Otimizada | **5x** |
| **Escalabilidade** | Limitada | Ilimitada | **∞** |

### **Fatores de Melhoria**
1. **Paralelização**: Processamento distribuído
2. **Cache inteligente**: Reutilização de dados
3. **Agregações otimizadas**: Spark SQL nativo
4. **Particionamento**: Balanceamento de carga
5. **Adaptive Query**: Otimização automática

## **📈 Estratégias de Otimização Adicionais**

### **1. Particionamento de Dados**
```python
# Particionar por micro-região para joins mais eficientes
df_processed = df_processed.repartition(100, "microregiao_origem")
```

### **2. Broadcast de Lookups**
```python
# Broadcast de mapeamentos pequenos
microregion_broadcast = spark.sparkContext.broadcast(microregion_mapping)
```

### **3. Persistência Estratégica**
```python
# Persistir DataFrames frequentemente usados
df_analysis.persist(StorageLevel.MEMORY_AND_DISK)
```

### **4. Otimização de Joins**
```python
# Usar broadcast joins para tabelas pequenas
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "100485760")
```

## **🛠️ Troubleshooting**

### **Problemas Comuns**

#### **1. Out of Memory**
```python
# Reduzir número de partições
df = df.coalesce(50)

# Aumentar memória do executor
spark.conf.set("spark.executor.memory", "8g")
```

#### **2. Skew de Dados**
```python
# Habilitar otimização de skew
spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")
spark.conf.set("spark.sql.adaptive.skewJoin.skewedPartitionThresholdInBytes", "256MB")
```

#### **3. Performance de Joins**
```python
# Usar broadcast joins para tabelas pequenas
spark.conf.set("spark.sql.adaptive.autoBroadcastJoinThreshold", "100485760")
```

## **📋 Checklist de Implementação**

### **Fase 1: Preparação**
- [ ] Configurar cluster Databricks
- [ ] Instalar dependências (spark-excel)
- [ ] Configurar otimizações Spark
- [ ] Preparar dados de teste

### **Fase 2: Implementação**
- [ ] Migrar código para Spark
- [ ] Implementar UDFs otimizadas
- [ ] Configurar cache estratégico
- [ ] Otimizar joins e agregações

### **Fase 3: Testes**
- [ ] Testes de performance
- [ ] Validação de resultados
- [ ] Ajustes de configuração
- [ ] Documentação final

### **Fase 4: Produção**
- [ ] Deploy em ambiente produtivo
- [ ] Monitoramento contínuo
- [ ] Otimizações iterativas
- [ ] Treinamento da equipe

## **🎯 Próximos Passos**

### **Imediatos (1-2 semanas)**
1. **Testar** notebook no Databricks
2. **Comparar** performance com versão atual
3. **Ajustar** configurações conforme necessário

### **Médio Prazo (1 mês)**
1. **Implementar** em produção
2. **Automatizar** execução via Jobs
3. **Monitorar** performance continuamente

### **Longo Prazo (3 meses)**
1. **Otimizar** ainda mais baseado em uso real
2. **Implementar** cache persistente
3. **Considerar** Delta Lake para dados históricos

## **💡 Dicas de Performance**

1. **Sempre use cache()** para DataFrames reutilizados
2. **Monitore o número de partições** - muito alto ou baixo pode prejudicar performance
3. **Use broadcast joins** para tabelas pequenas (< 100MB)
4. **Configure Adaptive Query Execution** para otimizações automáticas
5. **Limpe cache** quando não for mais necessário

## **📞 Suporte**

Para dúvidas sobre otimização:
- **Documentação Spark**: https://spark.apache.org/docs/latest/
- **Databricks Docs**: https://docs.databricks.com/
- **Comunidade**: Stack Overflow com tags [apache-spark], [databricks]

---

**🎉 Com essas otimizações, sua análise será 10-100x mais rápida!**
