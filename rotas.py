from typing import Optional, List
from datetime import date, timedelta, datetime

import pandas as pd

# <-- Configuração de logs -->
import logging

logging.basicConfig(level=logging.DEBUG)


class FretePorRotaProcessor:
    def __init__(self, caminho_excel) -> None:
        logging.info("\tIniciando o processador de rotas...")
        self.df_original = pd.read_excel(caminho_excel)
        self.col_datas = '00.dt_doc_faturamento'
        self._preparar_dados()

    def _preparar_dados(self) -> None:
        """
        Padronização de formato dos dados das colunas necessárias.
        """
        # Converter todas as entradas da coluna de datas para datetime
        self.df_original[self.col_datas] = pd.to_datetime(
            self.df_original[self.col_datas], errors='coerce', dayfirst=True
        )
        self.df_original = self.df_original[self.df_original[self.col_datas].notna()]
        logging.debug(f"\tColuna '{self.col_datas}' convertida para datetime.")

        # Limpar formatos de floating point numbers
        colunas_para_converter = [
            '02.01.00 - Volume (ton)',
            '02.01.01 - Frete Geral (BRL)',
            '02.01.02 - DISTANCIA (KM)'
        ]
        for col in colunas_para_converter:
            self.df_original[col] = self.df_original[col].astype(str).str.replace(',', '.').astype(float)
            logging.debug(f"\tColuna '{col}' convertida para float.")

    def filtrar_por_datas(self) -> None:
        """
        Gera três DataFrames:
        - df_3m: Dados dos últimos 3 meses antes de 31/12/2024. 
        - df_6m: Dados dos últimos 6 meses antes de 31/12/2024.
        - df_12m: Dados dos últimos 12 meses antes de 31/12/2024.
        """
        logging.info("\tGerando DataFrames de 3, 6 e 12 meses retroativos a partir de 31/12/2024.")

        # Tomar a data mais recente como referencia
        base_date = self.df_original[self.col_datas].max()

        periodos = {
            'df_3m': base_date - timedelta(days=90),
            'df_6m': base_date - timedelta(days=180),
            'df_12m': base_date - timedelta(days=365)
        }

        for nome_df, data_inicio in periodos.items():
            df_filtrado = self.df_original[
                (self.df_original[self.col_datas] >= data_inicio) &
                (self.df_original[self.col_datas] <= base_date)
            ].copy()
            setattr(self, nome_df, df_filtrado)
            logging.debug(f"\t{nome_df} gerado com período de {data_inicio.date()} até {base_date.date()}. Registros: {len(df_filtrado)}")

    def calcular_agregados_por_rota(self, df: pd.DataFrame) -> pd.DataFrame:
        self.df_filtrado = df.copy()

        self.df_filtrado['Preco_Frete_BRL_TON'] = (
                self.df_filtrado['02.01.01 - Frete Geral (BRL)'] /
                self.df_filtrado['02.01.00 - Volume (ton)']
        )
        self.df_filtrado['Preco_Frete_BRL_TON_KM'] = (
                self.df_filtrado['02.01.01 - Frete Geral (BRL)'] /
                (self.df_filtrado['02.01.00 - Volume (ton)'] * self.df_filtrado['02.01.02 - DISTANCIA (KM)'])
        )

        def media_ponderada(x, valor, peso):
            soma_pesos = x[peso].sum()
            if soma_pesos == 0 or pd.isna(soma_pesos):
                return float('nan')
            return (x[valor] * x[peso]).sum() / soma_pesos

        resultados = []

        for rota, grupo in self.df_filtrado.groupby('01.Rota_MuniOrigem_MuniDestino'):
            volume_total = grupo['02.01.00 - Volume (ton)'].sum()
            frete_total = grupo['02.01.01 - Frete Geral (BRL)'].sum()
            distancia_total = grupo['02.01.02 - DISTANCIA (KM)'].sum()
            distancia_unica = grupo['02.01.02 - DISTANCIA (KM)'].iloc[0]
            num_entregas = grupo.shape[0]

            def classificar_faixa(distancia):
                if distancia > 2000:
                    return "> 2000"
                elif 1501 <= distancia <= 2000:
                    return "1501 a 2000"
                elif 1001 <= distancia <= 1500:
                    return "1001 a 1500"
                elif 751 <= distancia <= 1000:
                    return "751 a 1000"
                elif 501 <= distancia <= 750:
                    return "501 a 750"
                elif 401 <= distancia <= 500:
                    return "401 a 500"
                elif 301 <= distancia <= 400:
                    return "301 a 400"
                elif 201 <= distancia <= 300:
                    return "201 a 300"
                elif 151 <= distancia <= 200:
                    return "151 a 200"
                elif 101 <= distancia <= 150:
                    return "101 a 150"
                else:
                    return "<= 100"

            faixa_distancia = classificar_faixa(distancia_unica)
            meso = grupo['01.Rota_MesoOrigem_MesoDestino'].iloc[0]
            cluster_id = f"{meso} | {faixa_distancia}"

            preco_frete_ton = media_ponderada(grupo, 'Preco_Frete_BRL_TON', '02.01.00 - Volume (ton)')
            preco_frete_ton_km = media_ponderada(grupo, 'Preco_Frete_BRL_TON_KM', '02.01.00 - Volume (ton)')

            if distancia_unica * num_entregas * volume_total > 0:
                indice_rota = frete_total / (volume_total * num_entregas * distancia_unica)
            else:
                indice_rota = float('nan')

            resultados.append({
                'Meso_Origem_Destino': meso,
                'Rota_MuniOrigem_MuniDestino': rota,
                'Cluster_ID': cluster_id,
                'Faixa_Distancia': faixa_distancia,
                'Distancia_Total_KM': distancia_total,
                'Distancia_Unica_Rota_KM': distancia_unica,
                'Num_Entregas': num_entregas,
                'Volume_Total': volume_total,
                'Frete_Total_BRL': frete_total,
                'Indice_Rota': indice_rota
            })

        df_resultado = pd.DataFrame(resultados)

        # Agrupamentos agora por Cluster_ID
        medias_cluster = df_resultado.groupby('Cluster_ID').apply(
            lambda x: pd.Series({
                'Indice_Medio_Cluster': media_ponderada(x, 'Indice_Rota', 'Volume_Total')
            })
        ).reset_index()
        df_resultado = df_resultado.merge(medias_cluster, on='Cluster_ID', how='left')

        df_resultado['Indice_Relativo'] = (
                (df_resultado['Indice_Rota'] - df_resultado['Indice_Medio_Cluster']) /
                df_resultado['Indice_Medio_Cluster']
        )

        volume_cluster = df_resultado.groupby('Cluster_ID')['Volume_Total'].sum().rename('Volume_Cluster')
        df_resultado = df_resultado.merge(volume_cluster, on='Cluster_ID', how='left')
        df_resultado['Volume_Relativo_Cluster'] = df_resultado['Volume_Total'] / df_resultado['Volume_Cluster']

        desvios_por_cluster = df_resultado.groupby('Cluster_ID')['Indice_Rota'].std().rename('Desvio_Indice_Cluster')
        df_resultado = df_resultado.merge(desvios_por_cluster, on='Cluster_ID', how='left')
        df_resultado['Indice_Zscore'] = (
                (df_resultado['Indice_Rota'] - df_resultado['Indice_Medio_Cluster']) /
                df_resultado['Desvio_Indice_Cluster']
        )

        z_limiar_por_cluster = df_resultado.groupby('Cluster_ID')['Indice_Zscore'].quantile(0.9)
        v_limiar_por_cluster = df_resultado.groupby('Cluster_ID')['Volume_Relativo_Cluster'].quantile(0.25)
        df_resultado = df_resultado.merge(z_limiar_por_cluster.rename('Z_Limiar_Cluster'), on='Cluster_ID', how='left')
        df_resultado = df_resultado.merge(v_limiar_por_cluster.rename('Volume_Limiar_Cluster'), on='Cluster_ID',
                                          how='left')

        # Novos limiares para 3 níveis
        z_p90 = df_resultado.groupby('Cluster_ID')['Indice_Zscore'].quantile(0.90).rename('Z_P90')
        z_p70 = df_resultado.groupby('Cluster_ID')['Indice_Zscore'].quantile(0.70).rename('Z_P70')
        v_p75 = df_resultado.groupby('Cluster_ID')['Volume_Relativo_Cluster'].quantile(0.75).rename('V_P75')
        v_p50 = df_resultado.groupby('Cluster_ID')['Volume_Relativo_Cluster'].quantile(0.50).rename('V_P50')

        # Merge com o DataFrame
        df_resultado = df_resultado.merge(z_p90, on='Cluster_ID', how='left')
        df_resultado = df_resultado.merge(z_p70, on='Cluster_ID', how='left')
        df_resultado = df_resultado.merge(v_p75, on='Cluster_ID', how='left')
        df_resultado = df_resultado.merge(v_p50, on='Cluster_ID', how='left')

        # Classificação em 3 níveis
        def classificar_impacto(row):
            if row['Indice_Zscore'] >= row['Z_P90'] and row['Volume_Relativo_Cluster'] >= row['V_P75']:
                return 'ALTO'
            elif row['Indice_Zscore'] >= row['Z_P70'] and row['Volume_Relativo_Cluster'] >= row['V_P50']:
                return 'MÉDIO'
            else:
                return 'BAIXO'

        df_resultado['Impacto_Estrategico'] = df_resultado.apply(classificar_impacto, axis=1)

        # Ordenação por mesorregião e faixa
        ordem_faixas = [
            "<= 100", "101 a 150", "151 a 200", "201 a 300", "301 a 400",
            "401 a 500", "501 a 750", "751 a 1000", "1001 a 1500", "1501 a 2000", "> 2000"
        ]
        df_resultado['Faixa_Distancia'] = pd.Categorical(df_resultado['Faixa_Distancia'], categories=ordem_faixas,
                                                         ordered=True)

        # Contagem por meso
        ordem_meso = df_resultado['Meso_Origem_Destino'].value_counts().index.tolist()

        # Definir ordem categórica
        df_resultado['Meso_Origem_Destino'] = pd.Categorical(
            df_resultado['Meso_Origem_Destino'], categories=ordem_meso, ordered=True
        )

        def centro_faixa(faixa):
            if faixa == '> 2000':
                return 2500
            ini, fim = faixa.split(' a ') if ' a ' in faixa else (faixa[2:], faixa[2:])
            return (int(ini) + int(fim)) / 2

        df_resultado['Faixa_Centro'] = df_resultado['Faixa_Distancia'].apply(centro_faixa)
        df_resultado = df_resultado.sort_values(by=['Meso_Origem_Destino', 'Faixa_Centro'])

        # Colunas finais (meso primeiro!)
        ordem_colunas = [
            'Meso_Origem_Destino',
            'Faixa_Distancia',
            'Rota_MuniOrigem_MuniDestino',
            'Distancia_Total_KM',
            'Volume_Total',
            'Frete_Total_BRL',
            'Indice_Rota',
            'Indice_Medio_Cluster',
            'Indice_Zscore',
            'Impacto_Estrategico'
        ]
        df_resultado = df_resultado[ordem_colunas]

        return df_resultado


if __name__ == "__main__":
    processor = FretePorRotaProcessor('sample_data.xlsx')

    # Gera os três dataframes filtrados: df_3m, df_6m, df_12m
    processor.filtrar_por_datas()

    # Calcula agregados para cada período
    df_resultado_3m = processor.calcular_agregados_por_rota(processor.df_3m)
    df_resultado_6m = processor.calcular_agregados_por_rota(processor.df_6m)
    df_resultado_12m = processor.calcular_agregados_por_rota(processor.df_12m)

    # Exporta ou imprime os resultados
    print("=== Resultado últimos 3 meses ===")
    print(df_resultado_3m)
    df_resultado_3m.to_excel('resultado_frete_3m.xlsx', index=False)

    print("=== Resultado últimos 6 meses ===")
    print(df_resultado_6m)
    df_resultado_6m.to_excel('resultado_frete_6m.xlsx', index=False)

    print("=== Resultado últimos 12 meses ===")
    print(df_resultado_12m)
    df_resultado_12m.to_excel('resultado_frete_12m.xlsx', index=False)
