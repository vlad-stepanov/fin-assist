import pandas as pd

df_full = pd.read_csv("../2_merging/sber_multi_ticker_merged.csv")
print(df_full.head())

external_cols = [
    c for c in df_full.columns
    if c.startswith("IMOEX_")
    or c.startswith("USD_RUB_")
    or c.startswith("BRENT_")
    or c.startswith("SBER_")
]

df_full[external_cols] = (
    df_full[external_cols]
      .ffill()
      .bfill()
)

df_full.to_csv("multi_ticker_dataset.csv", index=False)