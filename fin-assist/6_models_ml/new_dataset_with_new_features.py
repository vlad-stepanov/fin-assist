import pandas as pd
import numpy as np

df = pd.read_csv('../../3_postprocessing/rename_target_col/multi_ticker_dataset_encoded.csv', parse_dates=['Date']) # trimmed теперь не нужен
df = df.sort_values('Date').reset_index(drop=True)

df['SBER_pct']  = (df['SBER_Close']   - df['SBER_Open'])   / df['SBER_Open']
df['BRENT_pct'] = (df['BRENT_Close']  - df['BRENT_Open'])  / df['BRENT_Open']
df['USD_pct']   = (df['USD_RUB_Close']- df['USD_RUB_Open'])/ df['USD_RUB_Open']
df['IMOEX_pct'] = (df['IMOEX_Close']  - df['IMOEX_Open'])  / df['IMOEX_Open']
df['RATIO_SB_US']= df['SBER_Close'] / df['USD_RUB_Close']

for lag in [1, 3, 5, 10]:
    df[f'SBER_pct_lag_{lag}'] = df['SBER_pct'].shift(lag)

for window in [5, 10, 20]:
    df[f'vol_SBER_{window}d'] = df['SBER_pct'].rolling(window).std()

df['Target_bin'] = (df['Target'] == 2).astype(int)

df = df.dropna().reset_index(drop=True)

df.to_csv('multi_ticker_with_lags_vol.csv', index=False)
print("New dataset shape:", df.shape)
