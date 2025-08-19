# app.py
import logging
import os
from adapters.excel_adapter import ExcelAdapter
from adapters.excel_exporter import ExcelExporter
from domain.processor import FretePorRotaService

logging.basicConfig(level=logging.INFO)

def main(caminho_entrada: str, pasta_saida: str):
    # Garantir que a pasta de saída existe
    os.makedirs(pasta_saida, exist_ok=True)

    reader = ExcelAdapter(caminho_entrada)
    df = reader.read()

    service = FretePorRotaService(df)
    janelas = service.filtrar_por_datas()

    resultado_3m = service.calcular_agregados_por_rota(janelas['df_3m'])
    resultado_6m = service.calcular_agregados_por_rota(janelas['df_6m'])
    resultado_12m = service.calcular_agregados_por_rota(janelas['df_12m'])

    exporter = ExcelExporter()
    exporter.write(resultado_3m, os.path.join(pasta_saida, 'resultado_frete_3m.xlsx'))
    exporter.write(resultado_6m, os.path.join(pasta_saida, 'resultado_frete_6m.xlsx'))
    exporter.write(resultado_12m, os.path.join(pasta_saida, 'resultado_frete_12m.xlsx'))

    logging.info(f"Pipeline concluído para {caminho_entrada}.")


if __name__ == "__main__":
    bases = [
        ("Base_ITA_BARRAMANSA_2024_FULL.csv", "saida_barramansa"),
        ("Base_ITA_HUB_2024_FULL.csv", "saida_hub"),
        ("Base_ITA_JF_2024_FULL.csv", "saida_jf"),
        ("Base_ITA_Monlevade_2024_FULL.csv", "saida_monlevade"),
        ("Base_ITA_PIRA_2024_FULL.csv", "saida_pira"),
        ("Base_ITA_RESENDE_2024_FULL.csv", "saida_resende"),
        ("Base_ITA_SABARA_2024_FULL.csv", "saida_sabara"),
        ("Base_ITA_SP_2024_FULL.csv", "saida_sp"),
        ("Base_ITA_CL_2024_FULL.csv", "saida_cl")
    ]

    for arquivo, pasta in bases:
        main(caminho_entrada=arquivo, pasta_saida=pasta)
