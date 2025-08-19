import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import re

class AnalisadorTransporte:
    """
    Classe para analisar dados de transporte, calculando índices de performance
    para transportadoras com base em frete, volume e distância.
    """

    def __init__(self, caminho_arquivo: str):
        try:
            self.colunas_mapa = {
                '00.dt_doc_faturamento': 'data',
                '01.Rota_MuniOrigem_MuniDestino': 'rota',
                'nm_transportadora_aux': 'transportadora',
                '02.01.00 - Volume (ton)': 'volume',
                '02.03.02 - Preço_Frete Geral (BRL / TON / KM)': 'preco_ton_km'
            }
            self.df = self._carregar_dados(caminho_arquivo)
            self._preparar_dados()
            print("Dados carregados e preparados com sucesso.")
        except FileNotFoundError:
            print(f"Erro: O arquivo '{caminho_arquivo}' não foi encontrado.")
            self.df = None
        except Exception as e:
            print(f"Ocorreu um erro ao carregar ou preparar os dados: {e}")
            self.df = None

    def _carregar_dados(self, caminho_arquivo: str) -> pd.DataFrame:
        if caminho_arquivo.endswith('.xlsx'):
            try: return pd.read_excel(caminho_arquivo, skipfooter=1)
            except: return pd.read_excel(caminho_arquivo)
        elif caminho_arquivo.endswith('.csv'): return pd.read_csv(caminho_arquivo)
        else: raise ValueError("Formato de arquivo não suportado. Use .xlsx ou .csv")

    def _preparar_dados(self):
        self.df.rename(columns=self.colunas_mapa, inplace=True)
        colunas_necessarias = list(self.colunas_mapa.values())
        if not all(col in self.df.columns for col in colunas_necessarias):
            raise ValueError(f"Colunas esperadas não encontradas: {[c for c in colunas_necessarias if c not in self.df.columns]}")
        self.df['data'] = pd.to_datetime(self.df['data'], errors='coerce')
        self.df.dropna(subset=['data'], inplace=True)
        self.df['preco_ton_km'] = pd.to_numeric(self.df['preco_ton_km'], errors='coerce').fillna(0)
        self.df['volume'] = pd.to_numeric(self.df['volume'], errors='coerce').fillna(0)
    
    def filtrar_por_datas(self) -> None:
        """
        Gera três DataFrames (df_3m, df_6m, df_12m) com base na data mais
        recente do dataset e os armazena como atributos da classe.
        """
        if self.df is None or self.df.empty:
            print("DataFrame não carregado. Não é possível filtrar por datas.")
            return

        base_date = self.df['data'].max()
        print(f"\nGerando DataFrames de 3, 6 e 12 meses retroativos a partir de {base_date.strftime('%d/%m/%Y')}.")

        periodos = {
            'df_3m': base_date - timedelta(days=90),
            'df_6m': base_date - timedelta(days=180),
            'df_12m': base_date - timedelta(days=365)
        }

        for nome_df, data_inicio in periodos.items():
            df_filtrado = self.df[
                (self.df['data'] >= data_inicio) & (self.df['data'] <= base_date)
            ].copy()
            # Armazena o dataframe filtrado como um atributo da classe (ex: self.df_3m)
            setattr(self, nome_df, df_filtrado)
            print(f"\t- DataFrame '{nome_df}' gerado com {len(df_filtrado)} registros.")

    def calcular_indices_performance(self, df_filtrado: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula os índices de performance em um DataFrame de período já filtrado.
        """
        if df_filtrado is None or df_filtrado.empty:
            print(f"DataFrame do período fornecido está vazio. Análise pulada.")
            return pd.DataFrame()

        df_filtrado = df_filtrado.copy()
        
        df_filtrado['preco_x_volume'] = df_filtrado['preco_ton_km'] * df_filtrado['volume']
        somas_transportadora = df_filtrado.groupby(['rota', 'transportadora']).agg(total_preco_x_volume=('preco_x_volume', 'sum'), total_volume=('volume', 'sum')).reset_index()
        somas_transportadora['indice_transportadora'] = somas_transportadora.apply(lambda r: r['total_preco_x_volume'] / r['total_volume'] if r['total_volume'] != 0 else 0, axis=1)
        somas_rota = df_filtrado.groupby('rota').agg(total_preco_x_volume=('preco_x_volume', 'sum'), total_volume=('volume', 'sum')).reset_index()
        somas_rota['indice_mercado_rota'] = somas_rota.apply(lambda r: r['total_preco_x_volume'] / r['total_volume'] if r['total_volume'] != 0 else 0, axis=1)
        
        df_pivot = somas_transportadora.pivot(index='rota', columns='transportadora', values='indice_transportadora')
        df_final = df_pivot.merge(somas_rota[['rota', 'indice_mercado_rota']].set_index('rota'), on='rota', how='left')
        
        def natural_sort_key(s): return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]
        transportadoras_ordenadas = sorted(df_pivot.columns, key=natural_sort_key)

        df_numerico = df_final[transportadoras_ordenadas].copy()
        desvio_abs_da_media = df_numerico.sub(df_final['indice_mercado_rota'], axis=0).abs()

        volumes_pivot = somas_transportadora.pivot(index='rota', columns='transportadora', values='total_volume').fillna(0)
        total_volume_rota = somas_rota.set_index('rota')['total_volume']
        percentual_volume_df = volumes_pivot.div(total_volume_rota, axis=0).fillna(0) * 100

        index_df = abs(percentual_volume_df*desvio_abs_da_media)
        index_df['indice_impacto'] = index_df.sum(axis=1)

        p50 = index_df.indice_impacto.quantile(0.50)
        p90 = index_df.indice_impacto.quantile(0.90)

        conditions = [
            index_df.indice_impacto >= p90,
            index_df.indice_impacto.between(p50, p90),
            index_df.indice_impacto <= p50
        ]
        options = ['ALTO', 'MÉDIO', 'BAIXO']
        index_df['impacto_estrategico'] = np.select(conditions, options, default=None)

        df_final['indice_impacto'] = index_df.indice_impacto
        df_final['impacto_estrategico'] = index_df.impacto_estrategico

        novas_colunas = []
        for transportadora in transportadoras_ordenadas:
            if transportadora in df_final.columns:
                novas_colunas.append(transportadora)
                diff_col_name = f"{transportadora}_diff_%"
                df_final[diff_col_name] = df_final.apply(lambda r: ((r[transportadora] - r['indice_mercado_rota']) / r['indice_mercado_rota']) * 100 if pd.notna(r[transportadora]) and r['indice_mercado_rota'] != 0 else pd.NA, axis=1)
                novas_colunas.append(diff_col_name)

        df_final = df_final[novas_colunas + ['indice_mercado_rota', 'indice_impacto', 'impacto_estrategico']]

        return df_final

    @staticmethod
    def transpose_df(df: pd.DataFrame) -> pd.DataFrame:
        return df.transpose()

# --- Exemplo de como usar a classe ---
if __name__ == '__main__':
    caminho_do_arquivo = 'sample_data.xlsx'
    analisador = AnalisadorTransporte(caminho_do_arquivo)
    
    if hasattr(analisador, 'df') and analisador.df is not None:
        # 1. Gera os três dataframes filtrados: df_3m, df_6m, df_12m
        analisador.filtrar_por_datas()

        # 2. Calcula agregados para cada período passando o DataFrame correto
        df_resultado_3m = analisador.calcular_indices_performance(analisador.df_3m)
        df_resultado_6m = analisador.calcular_indices_performance(analisador.df_6m)
        df_resultado_12m = analisador.calcular_indices_performance(analisador.df_12m)

        # 2a. Transpor as tabelas para o output final
        df_resultado_3m = analisador.transpose_df(df_resultado_3m)
        df_resultado_6m = analisador.transpose_df(df_resultado_6m)
        df_resultado_12m = analisador.transpose_df(df_resultado_12m)

        # 3. Exporta ou imprime os resultados
        print("\n=== Resultado últimos 3 meses ===")
        if not df_resultado_3m.empty:
            print(df_resultado_3m.head())
            df_resultado_3m.to_excel('resultado_impacto_3m.xlsx', index=True)
            print("--> Salvo em 'resultado_impacto_3m.xlsx'")
        
        print("\n=== Resultado últimos 6 meses ===")
        if not df_resultado_6m.empty:
            print(df_resultado_6m.head())
            df_resultado_6m.to_excel('resultado_impacto_6m.xlsx', index=True)
            print("--> Salvo em 'resultado_impacto_6m.xlsx'")

        print("\n=== Resultado últimos 12 meses ===")
        if not df_resultado_12m.empty:
            print(df_resultado_12m.head())
            df_resultado_12m.to_excel('resultado_impacto_12m.xlsx', index=True)
            print("--> Salvo em 'resultado_impacto_12m.xlsx'")
