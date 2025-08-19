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

        # ---- Cálculo base por rota ----
        resultados = []
        for rota, grupo in self.df_filtrado.groupby('01.Rota_MuniOrigem_MuniDestino'):
            volume_total = grupo['02.01.00 - Volume (ton)'].sum()
            frete_total = grupo['02.01.01 - Frete Geral (BRL)'].sum()
            distancia_unica = grupo['02.01.02 - DISTANCIA (KM)'].iloc[0]
            num_entregas = grupo.shape[0]

            faixa = classificar_faixa(distancia_unica)
            meso = grupo['01.Rota_MesoOrigem_MesoDestino'].iloc[0]
            cluster_id = f"{meso} | {faixa}"

            distancia_total = distancia_unica * num_entregas
            indice_rota = frete_total / (volume_total * distancia_unica) if volume_total > 0 else float('nan')

            resultados.append({
                'Meso': meso,
                'Rota': rota,
                'Faixa': faixa,
                'Cluster_ID': cluster_id,
                'Volume_Total': volume_total,
                'Frete_Total': frete_total,
                'Num_Entregas': num_entregas,
                'Distancia': distancia_unica,
                'Indice_Rota': indice_rota
            })



        df_rotas = pd.DataFrame(resultados)

        # ---- Seleção com critério Vn >= V1/8 ----
        selecionadas = []
        for meso, g in df_rotas.groupby('Meso'):
            g = g.sort_values('Volume_Total', ascending=False)
            if g.empty:
                continue
            limite = g.iloc[0]['Volume_Total'] / 16
            idxs = g[g['Volume_Total'] >= limite].index.tolist()
            selecionadas.extend(idxs)

        df_rotas['Selecionada'] = df_rotas.index.isin(selecionadas)

        # ---- Cálculo do índice médio do cluster (apenas rotas selecionadas) ----
        medias = (
            df_rotas[df_rotas['Selecionada']]
            .groupby('Cluster_ID')['Indice_Rota']
            .mean()
            .reset_index()
            .rename(columns={'Indice_Rota': 'Indice_Medio_Cluster'})
        )
        df_rotas = df_rotas.merge(medias, on='Cluster_ID', how='left')

        # ---- Classificação do impacto estratégico ----

        # Nova lógica: verificar quantas rotas existem em cada Cluster (Meso + Faixa)
        contagem_rotas_por_cluster = df_rotas.groupby(['Meso', 'Faixa'])['Rota'].transform('count')

        # Percentis de volume para ranqueamento
        p50_vol = df_rotas['Volume_Total'].quantile(0.50)
        p75_vol = df_rotas['Volume_Total'].quantile(0.75)
        p90_vol = df_rotas['Volume_Total'].quantile(0.90)

        impacto = []
        for idx, row in df_rotas.iterrows():
            if contagem_rotas_por_cluster[idx] == 1:
                impacto.append("AMOSTRAGEM INSUFICIENTE")
                continue
            if not row['Selecionada']:
                impacto.append("NÃO APLICÁVEL")
                continue
            media_cluster = row['Indice_Medio_Cluster']
            # Avaliação do impacto
            if row['Indice_Rota'] <= media_cluster:
                impacto.append("OK")
            else:
                grupo_sel = df_rotas[
                    (df_rotas['Selecionada']) &
                    (df_rotas['Cluster_ID'] == row['Cluster_ID']) &
                    (df_rotas['Indice_Rota'] > df_rotas['Indice_Medio_Cluster'])
                    ]
                diffs = (grupo_sel['Indice_Rota'] - grupo_sel['Indice_Medio_Cluster']).sort_values()
                p33 = diffs.quantile(0.33)
                p67 = diffs.quantile(0.67)
                diff = row['Indice_Rota'] - media_cluster

                # Modulação por volume
                volume = row['Volume_Total']
                fator_volume = 0
                if volume > p90_vol:
                    fator_volume = 2
                elif volume > p75_vol:
                    fator_volume = 1

                if diff <= p33:
                    nivel = 0  # BAIXO
                elif diff <= p67:
                    nivel = 1  # MÉDIO
                else:
                    nivel = 2  # ALTO

                nivel_ajustado = min(nivel + fator_volume, 2)  # limitar até ALTO

                impacto.append(["BAIXO", "MÉDIO", "ALTO"][nivel_ajustado])

        df_rotas['Impacto_Estrategico'] = impacto

        # ---- Ordenação para apresentação ----
        ordem_meso = df_rotas.groupby('Meso')['Rota'].count().sort_values(ascending=False).index.tolist()
        df_rotas['Meso'] = pd.Categorical(df_rotas['Meso'], categories=ordem_meso, ordered=True)

        ordem_faixas = ["<= 100", "101 a 150", "151 a 200", "201 a 300", "301 a 400",
                        "401 a 500", "501 a 750", "751 a 1000", "1001 a 1500",
                        "1501 a 2000", "> 2000"]
        df_rotas['Faixa'] = pd.Categorical(df_rotas['Faixa'], categories=ordem_faixas, ordered=True)

        def centro_faixa(faixa):
            if faixa == "> 2000": return 2500
            if ' a ' in faixa:
                ini, fim = faixa.split(' a ')
            else:
                ini = fim = faixa.strip('<= ')
            return (int(ini) + int(fim)) / 2

        df_rotas['Faixa_Centro'] = df_rotas['Faixa'].apply(centro_faixa)
        df_rotas = df_rotas.sort_values(by=['Meso', 'Faixa_Centro', 'Rota']).drop(columns=['Faixa_Centro'])

        # ---- Colunas finais ----
        ordem_colunas = [
            'Meso', 'Faixa', 'Rota',
            'Volume_Total', 'Frete_Total', 'Num_Entregas', 'Distancia',
            'Indice_Rota', 'Indice_Medio_Cluster',
            'Selecionada', 'Impacto_Estrategico'
        ]
        df_rotas = df_rotas[ordem_colunas]

        return df_rotas


if __name__ == "__main__":
    processor = FretePorRotaProcessor('Base_ITA_Monlevade_2024_FULL.xlsx')

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