#!/usr/bin/env python3
"""
Script para configurar ambiente Databricks para análise de oportunidades
Instala dependências e configura otimizações
"""

import subprocess
import sys
import os

def install_databricks_cli():
    """Instala Databricks CLI se não estiver instalado"""
    try:
        import databricks
        print("✅ Databricks CLI já está instalado")
        return True
    except ImportError:
        print("📦 Instalando Databricks CLI...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "databricks-cli"])
            print("✅ Databricks CLI instalado com sucesso")
            return True
        except subprocess.CalledProcessError:
            print("❌ Erro ao instalar Databricks CLI")
            return False

def configure_databricks():
    """Configura Databricks com as credenciais"""
    print("🔧 Configurando Databricks...")
    
    # Verificar se já está configurado
    try:
        result = subprocess.run(["databricks", "workspace", "ls"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Databricks já está configurado")
            return True
    except FileNotFoundError:
        print("❌ Databricks CLI não encontrado")
        return False
    
    # Solicitar configuração
    print("\n📋 Configure suas credenciais do Databricks:")
    host = input("Host (ex: https://adb-xxxxx.xx.azuredatabricks.net): ").strip()
    token = input("Token de acesso: ").strip()
    
    if not host or not token:
        print("❌ Host e token são obrigatórios")
        return False
    
    try:
        # Configurar Databricks
        subprocess.check_call([
            "databricks", "configure", "--token",
            "--host", host,
            "--token", token
        ])
        print("✅ Databricks configurado com sucesso")
        return True
    except subprocess.CalledProcessError:
        print("❌ Erro ao configurar Databricks")
        return False

def create_cluster_config():
    """Cria arquivo de configuração do cluster"""
    print("⚙️ Criando configuração do cluster...")
    
    cluster_config = {
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
            "spark.sql.adaptive.optimizeSkewedJoin.enabled": "true",
            "spark.sql.adaptive.advisoryPartitionSizeInBytes": "128m",
            "spark.sql.adaptive.minNumPostShufflePartitions": "1",
            "spark.sql.adaptive.maxShuffledHashJoinLocalMapThreshold": "0",
            "spark.sql.adaptive.autoBroadcastJoinThreshold": "100485760",
            "spark.sql.adaptive.skewJoin.skewedPartitionThresholdInBytes": "256MB",
            "spark.sql.adaptive.skewJoin.skewedPartitionFactor": "10"
        },
        "autotermination_minutes": 60,
        "enable_elastic_disk": True,
        "disk_spec": {
            "disk_type": {
                "ebs_volume_type": "GENERAL_PURPOSE_SSD",
                "disk_count": 1,
                "disk_size": 100
            }
        }
    }
    
    # Salvar configuração
    import json
    config_path = "databricks_cluster_config.json"
    with open(config_path, 'w') as f:
        json.dump(cluster_config, f, indent=2)
    
    print(f"✅ Configuração do cluster salva em: {config_path}")
    return config_path

def create_job_config():
    """Cria arquivo de configuração do job"""
    print("📋 Criando configuração do job...")
    
    job_config = {
        "name": "Analise Oportunidades Reducao",
        "email_notifications": {
            "on_success": [],
            "on_failure": [],
            "no_alert_for_skipped_runs": False
        },
        "timeout_seconds": 3600,
        "max_concurrent_runs": 1,
        "tasks": [
            {
                "task_key": "analise_oportunidades",
                "notebook_task": {
                    "notebook_path": "/Repos/analise_oportunidades_databricks",
                    "base_parameters": {
                        "data_file": "/dbfs/FileStore/sample_data.xlsx",
                        "output_path": "/dbfs/FileStore/outputs/"
                    }
                },
                "existing_cluster_id": "{{cluster_id}}",
                "timeout_seconds": 3600,
                "retry_on_timeout": False
            }
        ]
    }
    
    # Salvar configuração
    import json
    config_path = "databricks_job_config.json"
    with open(config_path, 'w') as f:
        json.dump(job_config, f, indent=2)
    
    print(f"✅ Configuração do job salva em: {config_path}")
    return config_path

def create_requirements_file():
    """Cria arquivo de requirements para Databricks"""
    print("📦 Criando arquivo de requirements...")
    
    requirements = [
        "pyspark>=3.4.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "openpyxl>=3.1.0",
        "xlsxwriter>=3.2.0"
    ]
    
    # Salvar requirements
    requirements_path = "requirements_databricks.txt"
    with open(requirements_path, 'w') as f:
        for req in requirements:
            f.write(f"{req}\n")
    
    print(f"✅ Requirements salvo em: {requirements_path}")
    return requirements_path

def create_init_script():
    """Cria script de inicialização para o cluster"""
    print("🚀 Criando script de inicialização...")
    
    init_script = """#!/bin/bash
# Script de inicialização para cluster Databricks
# Instala dependências necessárias

echo "Instalando dependências..."

# Atualizar pip
pip install --upgrade pip

# Instalar bibliotecas para Excel
pip install openpyxl xlsxwriter

# Instalar bibliotecas de ML
pip install scikit-learn xgboost lightgbm

# Verificar instalação
echo "Verificando instalações..."
python -c "import openpyxl; print('openpyxl OK')"
python -c "import xlsxwriter; print('xlsxwriter OK')"
python -c "import sklearn; print('scikit-learn OK')"

echo "Instalação concluída!"
"""
    
    # Salvar script
    script_path = "init_script.sh"
    with open(script_path, 'w') as f:
        f.write(init_script)
    
    print(f"✅ Script de inicialização salvo em: {script_path}")
    return script_path

def create_deployment_guide():
    """Cria guia de deploy"""
    print("📚 Criando guia de deploy...")
    
    guide = """# 🚀 Guia de Deploy - Análise de Oportunidades Databricks

## Pré-requisitos
- Acesso ao Databricks
- Permissões para criar clusters e jobs
- Arquivo de dados (sample_data.xlsx) no FileStore

## Passo a Passo

### 1. Configurar Cluster
```bash
# Usar a configuração gerada
databricks clusters create --json @databricks_cluster_config.json
```

### 2. Fazer Upload do Notebook
```bash
# Fazer upload do notebook
databricks workspace import notebooks/analise_oportunidades_databricks.py \
    /Repos/analise_oportunidades_databricks \
    --language PYTHON
```

### 3. Fazer Upload dos Dados
```bash
# Fazer upload do arquivo de dados
databricks fs cp sample_data.xlsx dbfs:/FileStore/
```

### 4. Criar Job
```bash
# Criar job usando a configuração
databricks jobs create --json @databricks_job_config.json
```

### 5. Executar Job
```bash
# Executar job
databricks jobs run-now --job-id <JOB_ID>
```

## Monitoramento
- Acompanhar execução no UI do Databricks
- Verificar logs de execução
- Monitorar métricas de performance

## Troubleshooting
- Verificar logs do cluster
- Confirmar permissões de acesso
- Validar caminhos dos arquivos
"""
    
    # Salvar guia
    guide_path = "DEPLOYMENT_GUIDE.md"
    with open(guide_path, 'w') as f:
        f.write(guide)
    
    print(f"✅ Guia de deploy salvo em: {guide_path}")
    return guide_path

def main():
    """Função principal"""
    print("🚀 Configurando ambiente Databricks para análise de oportunidades")
    print("=" * 70)
    
    # Verificar Python
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ é necessário")
        return
    
    # Instalar Databricks CLI
    if not install_databricks_cli():
        print("❌ Falha na instalação do Databricks CLI")
        return
    
    # Configurar Databricks
    if not configure_databricks():
        print("❌ Falha na configuração do Databricks")
        return
    
    # Criar arquivos de configuração
    cluster_config = create_cluster_config()
    job_config = create_job_config()
    requirements = create_requirements_file()
    init_script = create_init_script()
    deployment_guide = create_deployment_guide()
    
    print("\n" + "=" * 70)
    print("✅ Configuração concluída com sucesso!")
    print("\n📁 Arquivos criados:")
    print(f"   🏗️  Cluster: {cluster_config}")
    print(f"   📋 Job: {job_config}")
    print(f"   📦 Requirements: {requirements}")
    print(f"   🚀 Init Script: {init_script}")
    print(f"   📚 Guia: {deployment_guide}")
    
    print("\n🎯 Próximos passos:")
    print("1. Revisar configurações geradas")
    print("2. Ajustar parâmetros conforme necessário")
    print("3. Seguir o guia de deploy")
    print("4. Testar no Databricks")
    
    print("\n💡 Dica: Use o comando 'databricks --help' para ver todas as opções disponíveis")

if __name__ == "__main__":
    main()
