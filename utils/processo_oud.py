# domain/processor.py
from typing import Dict
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import logging
from datetime import timedelta

logging.basicConfig(level=logging.DEBUG)

class FretePorRotaService:
    def __init__(self, df: pd.DataFrame, date_col: str = '00.dt_doc_faturamento'):
        self.df_original = df.copy()
        self.col_datas = date_col

    def filtrar_por_datas(self, referencia: pd.Timestamp = None) -> Dict[str, pd.DataFrame]:
        if referencia is None:
            referencia = self.df_original[self.col_datas].max()
        logging.info("Base date usada: %s", referencia.date())

        periodos = {
            'df_3m': referencia - timedelta(days=90),
            'df_6m': referencia - timedelta(days=180),
            'df_12m': referencia - timedelta(days=365)
        }
        result = {}
        for nome, inicio in periodos.items():
            df_filtrado = self.df_original[
                (self.df_original[self.col_datas] >= inicio) &
                (self.df_original[self.col_datas] <= referencia)
            ].copy()
            logging.debug("%s: %s -> %s (%d registros)", nome, inicio.date(), referencia.date(), len(df_filtrado))
            result[nome] = df_filtrado
        return result

    def analisar_rota_temporal(self, df_original: pd.DataFrame, rota: str) -> str:
        dados_rota = df_original[df_original['01.Rota_MuniOrigem_MuniDestino'] == rota].copy()
        dados_rota = dados_rota.sort_values(self.col_datas)

        dados_rota = dados_rota[(dados_rota['02.01.00 - Volume (ton)'] > 0) &
                                (dados_rota['02.01.02 - DISTANCIA (KM)'] > 0)]
        if dados_rota.shape[0] < 2:
            return "DADOS INSUFICIENTES"

        dados_rota['Indice'] = (
            dados_rota['02.01.01 - Frete Geral (BRL)'] /
            (dados_rota['02.01.00 - Volume (ton)'] * dados_rota['02.01.02 - DISTANCIA (KM)'])
        )
        dados_rota = dados_rota[dados_rota['Indice'].notna()]
        if dados_rota.shape[0] < 2:
            return "DADOS INSUFICIENTES"

        media = dados_rota['Indice'].mean()
        ultimo = dados_rota['Indice'].iloc[-1]
        minimo = dados_rota['Indice'].min()

        X = np.arange(len(dados_rota)).reshape(-1, 1)
        y = dados_rota['Indice'].values
        modelo = LinearRegression().fit(X, y)
        tendencia = float(modelo.coef_[0])

        delta_media = (ultimo - media) / media if media > 0 else 0
        delta_min = (ultimo - minimo) / minimo if minimo > 0 else 0

        if tendencia < 0:
            return "EM REDUÇÃO (MONITORAR)"
        elif delta_media > 0.2 and delta_min > 0.3:
            return "ALTO (SUGERIR AÇÃO)"
        else:
            return "NEUTRO (ANÁLISE NECESSÁRIA)"

    def _classificar_faixa(self, distancia: float) -> str:
        if distancia > 2000:
            return "> 2000"
        if 1501 <= distancia <= 2000:
            return "1501 a 2000"
        if 1001 <= distancia <= 1500:
            return "1001 a 1500"
        if 751 <= distancia <= 1000:
            return "751 a 1000"
        if 501 <= distancia <= 750:
            return "501 a 750"
        if 401 <= distancia <= 500:
            return "401 a 500"
        if 301 <= distancia <= 400:
            return "301 a 400"
        if 201 <= distancia <= 300:
            return "201 a 300"
        if 151 <= distancia <= 200:
            return "151 a 200"
        if 101 <= distancia <= 150:
            return "101 a 150"
        return "<= 100"

    def calcular_agregados_por_rota(self, df: pd.DataFrame) -> pd.DataFrame:
        df_filtrado = df.copy()
        resultados = []
        for rota, grupo in df_filtrado.groupby('01.Rota_MuniOrigem_MuniDestino'):
            volume_total = grupo['02.01.00 - Volume (ton)'].sum()
            frete_total = grupo['02.01.01 - Frete Geral (BRL)'].sum()
            distancia_unica = grupo['02.01.02 - DISTANCIA (KM)'].iloc[0]
            num_entregas = grupo.shape[0]

            faixa = self._classificar_faixa(distancia_unica)
            meso = grupo['01.Rota_MesoOrigem_MesoDestino'].iloc[0]
            cluster_id = f"{meso} | {faixa}"

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

        selecionadas = []
        for cluster_id, g in df_rotas.groupby('Cluster_ID'):
            g = g.copy()
            if g.empty:
                continue
            rotas_selecionadas = set(g['Rota'])

            def calcular_EP_total(rotas):
                subset = g[g['Rota'].isin(rotas)]
                media_cluster = subset['Indice_Rota'].mean()
                subset_contrib = subset[subset['Indice_Rota'] > media_cluster]
                if subset_contrib.empty:
                    return 0.0
                return ((subset_contrib['Indice_Rota'] - media_cluster) *
                        subset_contrib['Distancia'] *
                        subset_contrib['Volume_Total']).sum()

            EP_total = calcular_EP_total(rotas_selecionadas)

            while len(rotas_selecionadas) > 1:
                if EP_total == 0 or abs(EP_total) < 1e-9:
                    break
                candidatos = []
                for rota in rotas_selecionadas:
                    rotas_tmp = rotas_selecionadas - {rota}
                    EP_tmp = calcular_EP_total(rotas_tmp)
                    variacao = (EP_total - EP_tmp) / EP_total if EP_total != 0 else 0
                    candidatos.append((variacao, rota, EP_tmp))

                candidatos.sort(key=lambda x: x[0])
                menor_variacao, rota_remover, novo_EP_total = candidatos[0]
                if menor_variacao >= 0.01:
                    break
                rotas_selecionadas.remove(rota_remover)
                EP_total = novo_EP_total

            selecionadas.extend(df_rotas[df_rotas['Rota'].isin(rotas_selecionadas)].index.tolist())

        df_rotas['Selecionada'] = df_rotas.index.isin(selecionadas)

        medias = (
            df_rotas[df_rotas['Selecionada']]
            .groupby('Cluster_ID')['Indice_Rota']
            .mean()
            .reset_index()
            .rename(columns={'Indice_Rota': 'Indice_Medio_Cluster'})
        )
        df_rotas = df_rotas.merge(medias, on='Cluster_ID', how='left')

        contagem_rotas_por_cluster = df_rotas.groupby(['Meso', 'Faixa'])['Rota'].transform('count')
        p50_vol = df_rotas['Volume_Total'].quantile(0.50)
        p75_vol = df_rotas['Volume_Total'].quantile(0.75)
        p90_vol = df_rotas['Volume_Total'].quantile(0.90)

        impacto, analise_temporal = [], []

        for idx, row in df_rotas.iterrows():
            if contagem_rotas_por_cluster[idx] == 1:
                if row['Volume_Total'] > p75_vol:
                    sugestao = self.analisar_rota_temporal(self.df_original, row['Rota'])
                    analise_temporal.append(sugestao)
                else:
                    analise_temporal.append("-")
                impacto.append("AMOSTRAGEM INSUFICIENTE")
                continue

            if not row['Selecionada']:
                impacto.append("NÃO APLICÁVEL")
                analise_temporal.append("-")
                continue

            media_cluster = row['Indice_Medio_Cluster']
            if row['Indice_Rota'] <= media_cluster:
                impacto.append("OK")
                analise_temporal.append("-")
            else:
                grupo_sel = df_rotas[
                    (df_rotas['Selecionada']) &
                    (df_rotas['Cluster_ID'] == row['Cluster_ID']) &
                    (df_rotas['Indice_Rota'] > df_rotas['Indice_Medio_Cluster'])
                ]
                diffs = (grupo_sel['Indice_Rota'] - grupo_sel['Indice_Medio_Cluster']).sort_values()
                p33, p67 = diffs.quantile(0.33), diffs.quantile(0.67)
                diff = row['Indice_Rota'] - media_cluster

                volume = row['Volume_Total']
                fator_volume = 2 if volume > p90_vol else (1 if volume > p75_vol else 0)

                if diff <= p33:
                    nivel = 0
                elif diff <= p67:
                    nivel = 1
                else:
                    nivel = 2

                if row['Volume_Total'] == 0:
                    impacto.append("NÃO APLICÁVEL")
                    analise_temporal.append("-")
                    continue
                nivel_ajustado = min(nivel + fator_volume, 2)
                impacto.append(["BAIXO", "MÉDIO", "ALTO"][nivel_ajustado])
                analise_temporal.append("-")

        df_rotas['Impacto_Estrategico'] = impacto
        df_rotas['Analise_Temporal'] = analise_temporal

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

        ordem_colunas = [
            'Meso', 'Faixa', 'Rota',
            'Volume_Total', 'Frete_Total', 'Num_Entregas', 'Distancia',
            'Indice_Rota', 'Indice_Medio_Cluster',
            'Selecionada', 'Impacto_Estrategico', 'Analise_Temporal'
        ]
        df_rotas = df_rotas[ordem_colunas]

        return df_rotas