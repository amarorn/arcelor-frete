# Databricks notebook source
# MAGIC %md
# MAGIC # Teste de Conexão com Databricks
# MAGIC 
# MAGIC Este notebook testa a conexão com o cluster do Databricks e executa uma análise simples dos dados.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Verificar Informações do Cluster

# COMMAND ----------

# Verificar informações do cluster atual
print(f"Cluster ID: {spark.conf.get('spark.databricks.cluster.id')}")
print(f"Spark Version: {spark.version}")
print(f"Python Version: {spark.conf.get('spark.databricks.python.version')}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Testar Leitura de Dados

# COMMAND ----------

# Testar se conseguimos ler dados (exemplo com dados de teste)
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit

# Criar dados de teste
test_data = [
    ("JOÃO MONLEVADE-SABARÁ", 800.0, 35.0, 3500.0),
    ("JOÃO MONLEVADE-CONTAGEM", 1200.0, 45.0, 6000.0),
    ("ITABIRA-BELO HORIZONTE", 900.0, 120.0, 12000.0),
    ("ITABIRA-CONTAGEM", 1100.0, 110.0, 11000.0)
]

# Criar DataFrame
df_test = spark.createDataFrame(
    test_data, 
    ["rota", "volume_ton", "distancia_km", "frete_brl"]
)

# Mostrar dados
print("Dados de teste criados:")
df_test.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Testar Cálculos Básicos

# COMMAND ----------

# Calcular preço por TKU
df_calculado = df_test.withColumn(
    "preco_ton_km", 
    col("frete_brl") / (col("volume_ton") * col("distancia_km"))
)

print("Dados com preço por TKU calculado:")
df_calculado.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Testar Agregações

# COMMAND ----------

# Calcular estatísticas por rota
df_stats = df_calculado.groupBy("rota").agg(
    col("volume_ton").alias("volume_total"),
    col("frete_brl").alias("frete_total"),
    col("distancia_km").alias("distancia"),
    col("preco_ton_km").alias("preco_medio_tku")
)

print("Estatísticas por rota:")
df_stats.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Verificar Performance

# COMMAND ----------

import time

# Testar performance com operações básicas
start_time = time.time()

# Operação de teste
df_performance = df_test.crossJoin(df_test.limit(1))
count_result = df_performance.count()

end_time = time.time()
execution_time = end_time - start_time

print(f"Operação de teste executada em {execution_time:.2f} segundos")
print(f"Resultado: {count_result} registros")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ Conexão Testada com Sucesso!
# MAGIC 
# MAGIC Se você conseguiu executar este notebook, a conexão com o Databricks está funcionando perfeitamente!
# MAGIC 
# MAGIC **Próximos passos:**
# MAGIC 1. Migrar nossa análise de oportunidades para o Databricks
# MAGIC 2. Usar o poder do Spark para processar grandes volumes de dados
# MAGIC 3. Implementar análises distribuídas e paralelas
