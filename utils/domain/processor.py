from typing import Dict
import polars as pl
import numpy as np
from sklearn.linear_model import LinearRegression
import logging
from datetime import timedelta

logging.basicConfig(level=logging.DEBUG)

class FretePorRotaService:
    def __init__(self, df: pl.DataFrame, date_col: str = '00.dt_doc_faturamento'):
        self.df_original = df.clone()
        self.col_datas = date_col

    def filtrar_por_datas(self, referencia: pl.Timestamp = None) -> Dict[str, pl.DataFrame]:
        if referencia is None:
            referencia = self.df_original.select(pl.col(self.col_datas).max()).item()
        logging.info("Base date usada: %s", referencia.date())

        periodos = {
            'df_3m': referencia - pl.duration(days=90),
            'df_6m': referencia - pl.duration(days=180),
            'df_12m': referencia - pl.duration(days=365)
        }
        result = {}
        for nome, inicio in periodos.items():
            df_filtrado = self.df_original.filter(
                (pl.col(self.col_datas) >= inicio) &
                (pl.col(self.col_datas) <= referencia)
            )
            logging.debug("%s: %s -> %s (%d registros)", nome, inicio.date(), referencia.date(), df_filtrado.shape[0])
            result[nome] = df_filtrado
        return result

    def analisar_rota_temporal(self, df_original: pl.DataFrame, rota: str) -> str:
        dados_rota = df_original.filter(pl.col('01.Rota_MuniOrigem_MuniDestino') == rota).sort(self.col_datas)

        dados_rota = dados_rota.filter(
            (pl.col('02.01.00 - Volume (ton)') > 0) &
            (pl.col('02.01.02 - DISTANCIA (KM)') > 0)
        )
        if dados_rota.shape[0] < 2:
            return "DADOS INSUFICIENTES"

        dados_rota = dados_rota.with_columns([
            (pl.col('02.01.01 - Frete Geral (BRL)') /
             (pl.col('02.01.00 - Volume (ton)') * pl.col('02.01.02 - DISTANCIA (KM)'))).alias('Indice')
        ])
        dados_rota = dados_rota.filter(pl.col('Indice').is_not_null())
        if dados_rota.shape[0] < 2:
            return "DADOS INSUFICIENTES"

        media = dados_rota.select(pl.col('Indice').mean()).item()
        ultimo = dados_rota.select(pl.col('Indice').last()).item()
        minimo = dados_rota.select(pl.col('Indice').min()).item()

        X = np.arange(dados_rota.shape[0]).reshape(-1, 1)
        y = dados_rota.select(pl.col('Indice')).to_series().to_numpy()
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

    def _calcular_impacto_estrategico(self, indice_rota, media_cluster, volume_total, cluster_id, df_rotas, p75_vol, p90_vol):
        grupo_sel = df_rotas.filter(
            (pl.col('Selecionada')) &
            (pl.col('Cluster_ID') == cluster_id) &
            (pl.col('Indice_Rota') > pl.col('Indice_Medio_Cluster'))
        )
        diffs = grupo_sel.select(
            (pl.col('Indice_Rota') - pl.col('Indice_Medio_Cluster')).sort()
        ).to_series()
        
        if diffs.len() == 0:
            return "NÃO APLICÁVEL"
            
        p33, p67 = diffs.quantile(0.33), diffs.quantile(0.67)
        diff = indice_rota - media_cluster

        fator_volume = 2 if volume_total > p90_vol else (1 if volume_total > p75_vol else 0)

        if diff <= p33:
            nivel = 0
        elif diff <= p67:
            nivel = 1
        else:
            nivel = 2

        if volume_total == 0:
            return "NÃO APLICÁVEL"
        nivel_ajustado = min(nivel + fator_volume, 2)
        return ["BAIXO", "MÉDIO", "ALTO"][nivel_ajustado]

    def calcular_agregados_por_rota(self, df: pl.DataFrame) -> pl.DataFrame:
        df_filtrado = df.clone()
        
        df_rotas = df_filtrado.group_by('01.Rota_MuniOrigem_MuniDestino').agg([
            pl.col('02.01.00 - Volume (ton)').sum().alias('Volume_Total'),
            pl.col('02.01.01 - Frete Geral (BRL)').sum().alias('Frete_Total'),
            pl.col('02.01.02 - DISTANCIA (KM)').first().alias('Distancia'),
            pl.len().alias('Num_Entregas'),
            pl.col('01.Rota_MesoOrigem_MesoDestino').first().alias('Meso')
        ]).with_columns([
            pl.struct(['Meso', 'Distancia']).map_elements(
                lambda x: self._classificar_faixa(x['Distancia']), 
                return_dtype=pl.Utf8
            ).alias('Faixa'),
            pl.struct(['Meso', 'Distancia']).map_elements(
                lambda x: f"{x['Meso']} | {self._classificar_faixa(x['Distancia'])}", 
                return_dtype=pl.Utf8
            ).alias('Cluster_ID'),
            pl.when(pl.col('Volume_Total') > 0)
            .then(pl.col('Frete_Total') / (pl.col('Volume_Total') * pl.col('Distancia')))
            .otherwise(float('nan')).alias('Indice_Rota')
        ])

        selecionadas = []
        for cluster_id, g in df_rotas.group_by('Cluster_ID'):
            if g.height == 0:
                continue
            rotas_selecionadas = set(g.select('01.Rota_MuniOrigem_MuniDestino').to_series().to_list())

            def calcular_EP_total(rotas):
                subset = g.filter(pl.col('01.Rota_MuniOrigem_MuniDestino').is_in(rotas))
                media_cluster = subset.select(pl.col('Indice_Rota').mean()).item()
                subset_contrib = subset.filter(pl.col('Indice_Rota') > media_cluster)
                if subset_contrib.height == 0:
                    return 0.0
                return subset_contrib.select(
                    ((pl.col('Indice_Rota') - media_cluster) * pl.col('Distancia') * pl.col('Volume_Total')).sum()
                ).item()

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

            selecionadas.extend(g.select('01.Rota_MuniOrigem_MuniDestino').to_series().to_list())

        df_rotas = df_rotas.with_columns([
            pl.col('01.Rota_MuniOrigem_MuniDestino').is_in(selecionadas).alias('Selecionada')
        ])

        medias = (
            df_rotas.filter(pl.col('Selecionada'))
            .group_by('Cluster_ID')
            .agg(pl.col('Indice_Rota').mean().alias('Indice_Medio_Cluster'))
        )
        df_rotas = df_rotas.join(medias, on='Cluster_ID', how='left')

        contagem_rotas_por_cluster = df_rotas.group_by(['Meso', 'Faixa']).agg([
            pl.len().alias('count')
        ]).select(['Meso', 'Faixa', 'count'])
        
        p50_vol = df_rotas.select(pl.col('Volume_Total').quantile(0.50)).item()
        p75_vol = df_rotas.select(pl.col('Volume_Total').quantile(0.75)).item()
        p90_vol = df_rotas.select(pl.col('Volume_Total').quantile(0.90)).item()

        df_rotas = df_rotas.with_columns([
            pl.struct(['Meso', 'Faixa']).map_elements(
                lambda x: contagem_rotas_por_cluster.filter(
                    (pl.col('Meso') == x['Meso']) & (pl.col('Faixa') == x['Faixa'])
                ).select('count').item(), 
                return_dtype=pl.Int64
            ).alias('contagem_cluster'),
            pl.when(pl.col('contagem_cluster') == 1)
            .then(pl.when(pl.col('Volume_Total') > p75_vol)
                  .then(pl.struct(['01.Rota_MuniOrigem_MuniDestino']).map_elements(
                      lambda x: self.analisar_rota_temporal(self.df_original, x['01.Rota_MuniOrigem_MuniDestino']),
                      return_dtype=pl.Utf8
                  ))
                  .otherwise("-"))
            .when(pl.col('Selecionada') == False)
            .then("-")
            .otherwise("-").alias('Analise_Temporal'),
            pl.when(pl.col('contagem_cluster') == 1)
            .then("AMOSTRAGEM INSUFICIENTE")
            .when(pl.col('Selecionada') == False)
            .then("NÃO APLICÁVEL")
            .when(pl.col('Indice_Rota') <= pl.col('Indice_Medio_Cluster'))
            .then("OK")
            .otherwise(pl.struct(['Indice_Rota', 'Indice_Medio_Cluster', 'Volume_Total', 'Cluster_ID']).map_elements(
                lambda x: self._calcular_impacto_estrategico(
                    x['Indice_Rota'], x['Indice_Medio_Cluster'], x['Volume_Total'], 
                    x['Cluster_ID'], df_rotas, p75_vol, p90_vol
                ), return_dtype=pl.Utf8
            )).alias('Impacto_Estrategico')
        ])

        ordem_meso = df_rotas.group_by('Meso').agg([
            pl.len().alias('count')
        ]).sort('count', descending=True).select('Meso').to_series().to_list()
        
        ordem_faixas = ["<= 100", "101 a 150", "151 a 200", "201 a 300", "301 a 400",
                        "401 a 500", "501 a 750", "751 a 1000", "1001 a 1500",
                        "1501 a 2000", "> 2000"]

        def centro_faixa(faixa):
            if faixa == "> 2000": return 2500
            if ' a ' in faixa:
                ini, fim = faixa.split(' a ')
            else:
                ini = fim = faixa.strip('<= ')
            return (int(ini) + int(fim)) / 2

        df_rotas = df_rotas.with_columns([
            pl.col('Faixa').map_elements(centro_faixa).alias('Faixa_Centro')
        ])
        
        df_rotas = df_rotas.sort(['Meso', 'Faixa_Centro', '01.Rota_MuniOrigem_MuniDestino']).drop('Faixa_Centro')

        ordem_colunas = [
            'Meso', 'Faixa', '01.Rota_MuniOrigem_MuniDestino',
            'Volume_Total', 'Frete_Total', 'Num_Entregas', 'Distancia',
            'Indice_Rota', 'Indice_Medio_Cluster',
            'Selecionada', 'Impacto_Estrategico', 'Analise_Temporal'
        ]
        df_rotas = df_rotas.select(ordem_colunas)

        return df_rotas
