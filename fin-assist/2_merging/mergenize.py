import pandas as pd

def rename_columns(path, prefix, drop_volume=False):
    df = pd.read_csv(
        path,
        sep=",",
        parse_dates=["Date"],
        dayfirst=True,
        encoding="utf-8-sig"
    )
    # Переименование колонок
    mapping = {
        "Price":      f"{prefix}_Close",
        "Open":       f"{prefix}_Open",
        "High":       f"{prefix}_High",
        "Low":        f"{prefix}_Low",
        "Change_pct": f"{prefix}_PctChange",
    }
    if not drop_volume:
        mapping["Volume"] = f"{prefix}_Volume"
    df = df.rename(columns=mapping)
    # Удаляем Volume, если нужно
    if drop_volume:
        df = df.drop(columns=["Volume"], errors="ignore")
    return df

# 1) Загружаем SBER
df_s = pd.read_csv(
    '../1_preprocessing/feature_engineering/sber_with_features.csv',
    sep=",",
    parse_dates=['Date'],
    encoding="utf-8-sig"
).rename(columns={ #rename sber columns
    'Price':  'SBER_Close',
    'Open':   'SBER_Open',
    'High':   'SBER_High',
    'Low':    'SBER_Low',
    'Volume': 'SBER_Volume',
    'Value':  'SBER_Value'
}).rename(columns={ #rename indicator columns
    'SMA_10': 'SBER_SMA_10',
    'SMA_20': 'SBER_SMA_20',
    'SMA_50': 'SBER_SMA_50',
    'RSI_14': 'SBER_RSI_14'
})

# 2) Загружаем три clean-датафрейма
df_i = rename_columns('../1_preprocessing/cleaning/imoex_clean.csv',   'IMOEX',   drop_volume=True)
df_u = rename_columns('../1_preprocessing/cleaning/usd_rub_clean.csv','USD_RUB', drop_volume=True)
df_b = rename_columns('../1_preprocessing/cleaning/brent_clean.csv',   'BRENT',   drop_volume=False)

# 3) Нормализуем Date (отбросим время)
for df in (df_s, df_i, df_u, df_b):
    df['Date'] = pd.to_datetime(df['Date']).dt.normalize()

# 4) Диагностика (опционально)
print("SBER      :", df_s['Date'].min(), "-", df_s['Date'].max())
print("IMOEX     :", df_i['Date'].min(), "-", df_i['Date'].max())
print("USD_RUB   :", df_u['Date'].min(), "-", df_u['Date'].max())
print("BRENT     :", df_b['Date'].min(), "-", df_b['Date'].max())
common = set(df_s['Date']).intersection(df_i['Date'])
print("SBER∩IMOEX count:", len(common))

# 5) Последовательный merge по столбцу Date
df_full = (
    df_s
    .merge(df_i, on='Date', how='left')
    .merge(df_u, on='Date', how='left')
    .merge(df_b, on='Date', how='left')
)

# 6) Ставим Date индексом (если нужно) и сохраняем
df_full = df_full.set_index('Date')
df_full.to_csv('./sber_multi_ticker_merged.csv')
print("Saved to sber_multi_ticker_merged.csv — shape:", df_full.shape)
