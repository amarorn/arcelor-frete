import pandas as pd
from utils.adapters.polars_adapter import PolarsAdapter

# Ler dados
adapter = PolarsAdapter()
df_polars = adapter.read_excel("sample_data.xlsx")
df_pd = adapter.prepare_data(df_polars)

print("Colunas disponíveis após preparação:")
print(df_pd.columns)

print("\nPrimeiras linhas:")
print(df_pd.head())

print("\nVerificar se as colunas esperadas existem:")
expected_cols = [
    'rota_municipio', 'rota', '02.01.00 - Volume (ton)', 
    '02.01.02 - DISTANCIA (KM)', '02.01.01 - Frete Geral (BRL)',
    '02.03.02 - Preço_Frete Geral (BRL / TON / KM)', 'rota_microregiao'
]

for col in expected_cols:
    exists = col in df_pd.columns
    print(f"{col}: {'✓' if exists else '✗'}")

print("\nVerificar valores únicos em algumas colunas:")
if 'rota_municipio' in df_pd.columns:
    print("rota_municipio:", df_pd['rota_municipio'].unique()[:5])
elif 'rota' in df_pd.columns:
    print("rota:", df_pd['rota'].unique()[:5])

if '02.01.00 - Volume (ton)' in df_pd.columns:
    print("Volume (ton):", df_pd['02.01.00 - Volume (ton)'].describe())

if '02.01.02 - DISTANCIA (KM)' in df_pd.columns:
    print("Distância (KM):", df_pd['02.01.02 - DISTANCIA (KM)'].describe())
