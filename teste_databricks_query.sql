-- =====================================================
-- SCRIPT DE TESTE PARA DATABRICKS
-- Teste o arquivo sample_data_databricks_compatible.parquet
-- =====================================================

-- 1. TESTE BÁSICO DE LEITURA
-- Verificar se consegue ler o arquivo sem erros
SELECT 
    COUNT(*) as total_registros,
    COUNT(DISTINCT rota_municipio) as total_rotas,
    COUNT(DISTINCT data_faturamento) as total_datas
FROM sample_data_databricks_compatible;

-- 2. TESTE DE TIPOS DE DADOS
-- Verificar se todas as colunas estão com tipos corretos
DESCRIBE sample_data_databricks_compatible;

-- 3. TESTE DE OPERAÇÕES NUMÉRICAS
-- Verificar se consegue fazer cálculos com TKU
SELECT 
    rota_municipio,
    AVG(tku_calculado) as tku_medio,
    SUM(volume_ton) as volume_total,
    SUM(frete_brl) as frete_total,
    COUNT(*) as num_entregas
FROM sample_data_databricks_compatible
WHERE tku_calculado IS NOT NULL
GROUP BY rota_municipio
ORDER BY volume_total DESC
LIMIT 10;

-- 4. TESTE DE FILTROS DE DATA
-- Verificar se consegue filtrar por data (agora como string)
SELECT 
    data_faturamento,
    COUNT(*) as entregas_dia,
    AVG(tku_calculado) as tku_medio_dia
FROM sample_data_databricks_compatible
WHERE data_faturamento >= '2024-01-01' 
  AND data_faturamento <= '2024-12-31'
GROUP BY data_faturamento
ORDER BY data_faturamento
LIMIT 20;

-- 5. TESTE DE AGRUPAMENTO POR ROTA
-- Verificar performance de agrupamentos
SELECT 
    rota_microregiao,
    COUNT(DISTINCT rota_municipio) as num_municipios,
    SUM(volume_ton) as volume_total,
    AVG(tku_calculado) as tku_medio
FROM sample_data_databricks_compatible
WHERE tku_calculado IS NOT NULL
GROUP BY rota_microregiao
ORDER BY volume_total DESC
LIMIT 15;

-- 6. TESTE DE JOIN (se houver outras tabelas)
-- Exemplo de como seria um JOIN com outras tabelas
/*
SELECT 
    f.rota_municipio,
    f.volume_ton,
    f.tku_calculado,
    t.nome_transportadora
FROM sample_data_databricks_compatible f
LEFT JOIN transportadoras t ON f.id_transportadora = t.id
WHERE f.tku_calculado > 1.0
ORDER BY f.tku_calculado DESC
LIMIT 20;
*/

-- 7. TESTE DE PERFORMANCE
-- Verificar tempo de execução de queries complexas
-- (Execute esta query e verifique o tempo de resposta)
SELECT 
    rota_municipio,
    data_faturamento,
    volume_ton,
    frete_brl,
    distancia_km,
    tku_calculado,
    CASE 
        WHEN tku_calculado <= 0.5 THEN 'BAIXO'
        WHEN tku_calculado <= 1.0 THEN 'MÉDIO'
        ELSE 'ALTO'
    END as categoria_tku
FROM sample_data_databricks_compatible
WHERE data_faturamento >= '2024-06-01'
  AND tku_calculado IS NOT NULL
ORDER BY tku_calculado DESC
LIMIT 100;

-- 8. TESTE DE ESTATÍSTICAS
-- Verificar estatísticas gerais dos dados
SELECT 
    'Volume' as metrica,
    MIN(volume_ton) as valor_min,
    MAX(volume_ton) as valor_max,
    AVG(volume_ton) as valor_medio,
    STDDEV(volume_ton) as desvio_padrao
FROM sample_data_databricks_compatible
WHERE volume_ton IS NOT NULL

UNION ALL

SELECT 
    'TKU' as metrica,
    MIN(tku_calculado) as valor_min,
    MAX(tku_calculado) as valor_max,
    AVG(tku_calculado) as valor_medio,
    STDDEV(tku_calculado) as desvio_padrao
FROM sample_data_databricks_compatible
WHERE tku_calculado IS NOT NULL

UNION ALL

SELECT 
    'Distância' as metrica,
    MIN(distancia_km) as valor_min,
    MAX(distancia_km) as valor_max,
    AVG(distancia_km) as valor_medio,
    STDDEV(distancia_km) as desvio_padrao
FROM sample_data_databricks_compatible
WHERE distancia_km IS NOT NULL;

-- =====================================================
-- INSTRUÇÕES PARA EXECUTAR NO DATABRICKS:
-- =====================================================
-- 1. Faça upload do arquivo sample_data_databricks_compatible.parquet
-- 2. Execute cada query individualmente para testar
-- 3. Verifique se não há erros de tipo de dados
-- 4. Monitore o tempo de execução das queries
-- 5. Se tudo funcionar, o arquivo está 100% compatível!
-- =====================================================
