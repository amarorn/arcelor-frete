#!/usr/bin/env python3
"""
Script para gerar resumo de oportunidades de redução de custos
Baseado no dashboard do operador com análise temporal (3, 6 e 12 meses)
"""

import os
import sys
import logging
from pathlib import Path

# Adicionar o diretório atual ao path para importar a classe
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from analise_tendencia_tku import AnalisadorTendenciaTKU

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Função principal para gerar resumo de oportunidades com múltiplos períodos"""
    print("🎯 GERADOR DE RESUMO DE OPORTUNIDADES - ANÁLISE TEMPORAL")
    print("=" * 60)
    
    # Inicializar analisador
    analisador = AnalisadorTendenciaTKU()
    
    # Carregar dados
    print("📂 Carregando dados...")
    if not analisador.carregar_dados():
        print("❌ Erro ao carregar dados. Verifique o arquivo sample_data.xlsx")
        return
    
    # Gerar resumos para múltiplos períodos
    print("📊 Gerando resumos de oportunidades para múltiplos períodos...")
    resultados = analisador.gerar_resumo_oportunidades_multiplos_periodos()
    
    if not resultados:
        print("❌ Erro ao gerar resumos de oportunidades")
        return
    
    # Mostrar resumo geral
    print(f"\n✅ RESUMOS GERADOS COM SUCESSO!")
    print(f"📊 Períodos analisados: {len(resultados)}")
    
    # Analisar cada período
    for periodo, df_periodo in resultados.items():
        print(f"\n{'='*50}")
        print(f"📅 PERÍODO: {periodo}")
        print(f"{'='*50}")
        
        print(f"📊 Total de rotas analisadas: {len(df_periodo)}")
        
        # Contar ações
        acoes = df_periodo['Ação'].value_counts()
        print(f"\n🎯 DISTRIBUIÇÃO POR AÇÃO:")
        for acao, count in acoes.items():
            print(f"   • {acao}: {count} rotas")
        
        # Top 5 maiores volumes
        print(f"\n📈 TOP 5 ROTAS POR VOLUME:")
        df_top_volume = df_periodo.head(5)
        for i, (_, row) in enumerate(df_top_volume.iterrows(), 1):
            microregiao = row.get('Micro-Região Origem', 'N/A')
            print(f"   {i}. {row['Rota']} ({microregiao}): {row['Volume (TON)']:,.1f} ton "
                  f"(R$ {row['Custo Sup (TKU)']:.4f}/ton.km)")
        
        # Top 5 maiores oportunidades de redução
        df_reducao = df_periodo[df_periodo['Ação'] == 'Redução'].sort_values('Oport. (BRL/TON/KM)', ascending=False)
        if len(df_reducao) > 0:
            print(f"\n🔥 TOP 5 OPORTUNIDADES DE REDUÇÃO:")
            for i, (_, row) in enumerate(df_reducao.head(5).iterrows(), 1):
                microregiao = row.get('Micro-Região Origem', 'N/A')
                print(f"   {i}. {row['Rota']} ({microregiao}): R$ {row['Oport. (BRL/TON/KM)']:.4f} "
                      f"(Volume: {row['Volume (TON)']:,.1f} ton, Distância: {row['Distância (KM)']:.1f} km)")
        
        # Top 5 maiores oportunidades de aumento
        df_aumento = df_periodo[df_periodo['Ação'] == 'Aumento'].sort_values('Oport. (BRL/TON/KM)', ascending=True)
        if len(df_aumento) > 0:
            print(f"\n📈 TOP 5 OPORTUNIDADES DE AUMENTO:")
            for i, (_, row) in enumerate(df_aumento.head(5).iterrows(), 1):
                microregiao = row.get('Micro-Região Origem', 'N/A')
                print(f"   {i}. {row['Rota']} ({microregiao}): R$ {row['Oport. (BRL/TON/KM)']:.4f} "
                      f"(Volume: {row['Volume (TON)']:,.1f} ton, Distância: {row['Distância (KM)']:.1f} km)")
        
        # Distribuição por impacto
        impacto_dist = df_periodo['Impacto Estratégico'].value_counts()
        print(f"\n📊 DISTRIBUIÇÃO POR IMPACTO ESTRATÉGICO:")
        for impacto, count in impacto_dist.items():
            print(f"   • {impacto}: {count} rotas")
        
        # Informações temporais
        if 'Data Início' in df_periodo.columns and 'Data Fim' in df_periodo.columns:
            data_inicio = df_periodo['Data Início'].iloc[0]
            data_fim = df_periodo['Data Fim'].iloc[0]
            print(f"\n📅 PERÍODO ANALISADO: {data_inicio} a {data_fim}")
    
    # Salvar dashboard multi-período
    print(f"\n💾 Salvando dashboard multi-período...")
    dashboard_path = analisador.salvar_dashboard_oportunidades_multiplos_periodos()
    
    if dashboard_path:
        print(f"✅ Dashboard multi-período salvo em: {dashboard_path}")
        
        # Mostrar resumo comparativo
        print(f"\n📋 RESUMO COMPARATIVO ENTRE PERÍODOS:")
        df_resumo_comparativo = analisador._criar_resumo_comparativo_periodos(resultados)
        if len(df_resumo_comparativo) > 0:
            print(df_resumo_comparativo.to_string(index=False))
        
    else:
        print("❌ Erro ao salvar dashboard")
    
    print(f"\n🎉 Processo concluído!")
    print(f"\n📊 RESUMO FINAL:")
    print(f"   • Períodos analisados: 3, 6 e 12 meses")
    print(f"   • Dashboard salvo com múltiplas abas")
    print(f"   • Aba comparativa com métricas por período")

if __name__ == "__main__":
    main()
