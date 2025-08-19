#!/usr/bin/env python3
"""
Script para executar testes unitários do projeto ArcelorMittal
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Executa um comando e exibe o resultado"""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("⚠️  Warnings/Errors:")
            print(result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Erro ao executar: {e}")
        print(f"Output: {e.stdout}")
        print(f"Error: {e.stderr}")
        return False

def main():
    """Função principal"""
    print("🧪 EXECUTANDO TESTES UNITÁRIOS - PROJETO ARCELORMITTAL")
    print("=" * 60)
    
    # Verificar se estamos no diretório correto
    if not Path("tests").exists():
        print("❌ Diretório 'tests' não encontrado. Execute este script na raiz do projeto.")
        sys.exit(1)
    
    # Verificar se o ambiente virtual está ativado
    if not os.environ.get('VIRTUAL_ENV'):
        print("⚠️  Ambiente virtual não detectado. Ativando...")
        if Path("venv/bin/activate").exists():
            os.system("source venv/bin/activate")
        else:
            print("❌ Ambiente virtual não encontrado. Crie um com: python3 -m venv venv")
            sys.exit(1)
    
    # Instalar dependências de teste se necessário
    print("\n📦 Verificando dependências de teste...")
    try:
        import pytest
        print("✅ pytest já instalado")
    except ImportError:
        print("📥 Instalando pytest...")
        subprocess.run("pip install pytest pytest-cov pytest-mock", shell=True, check=True)
    
    # Executar testes com cobertura
    print("\n🧪 Executando testes com cobertura...")
    success = run_command(
        "python -m pytest tests/ -v --cov=utils --cov=notebooks --cov-report=term-missing",
        "Testes com cobertura"
    )
    
    if not success:
        print("\n❌ Alguns testes falharam!")
        sys.exit(1)
    
    # Executar testes específicos por categoria
    print("\n🔍 Executando testes por categoria...")
    
    # Testes unitários
    run_command(
        "python -m pytest tests/ -m unit -v",
        "Testes Unitários"
    )
    
    # Testes de integração
    run_command(
        "python -m pytest tests/ -m integration -v",
        "Testes de Integração"
    )
    
    # Gerar relatório de cobertura em HTML
    print("\n📊 Gerando relatório de cobertura em HTML...")
    run_command(
        "python -m pytest tests/ --cov=utils --cov=notebooks --cov-report=html",
        "Relatório HTML de cobertura"
    )
    
    print("\n🎉 Todos os testes foram executados!")
    print("📁 Relatório de cobertura disponível em: htmlcov/index.html")
    print("📊 Para ver detalhes da cobertura, abra: htmlcov/index.html no navegador")

if __name__ == "__main__":
    main()
