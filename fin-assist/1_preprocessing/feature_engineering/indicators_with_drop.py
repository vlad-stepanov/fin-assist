import pandas as pd

def add_indicators(df: pd.DataFrame,
                   sma_windows: tuple[int, ...] = (10, 20, 50),
                   rsi_window: int = 14) -> pd.DataFrame:

    df = df.copy()
    if 'Date' in df.columns:
        df.sort_values('Date', inplace=True)

    for w in sma_windows:
        df[f'SMA_{w}'] = df['Price'].rolling(window=w, min_periods=1).mean()

    delta = df['Price'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    ma_gain = gain.rolling(window=rsi_window, min_periods=1).mean()
    ma_loss = loss.rolling(window=rsi_window, min_periods=1).mean()
    rs = ma_gain / ma_loss.replace(0, pd.NA)
    df[f'RSI_{rsi_window}'] = 100 - (100 / (1 + rs))

    return df

def label_target(df: pd.DataFrame,
                 horizon: int = 5,
                 eps: float = 0.01) -> pd.DataFrame:

    df = df.copy()
    df['Price_future'] = df['Price'].shift(-horizon)
    df['ret'] = (df['Price_future'] - df['Price']) / df['Price']
    df['Target'] = pd.cut(
        df['ret'],
        bins=[-float('inf'), -eps, eps, float('inf')],
        labels=['SELL', 'HOLD', 'BUY']
    )

    df.drop(columns=['Price_future', 'ret'], inplace=True)
    return df

def add_features_and_target(df: pd.DataFrame,
                            sma_windows: tuple[int, ...] = (10, 20, 50),
                            rsi_window: int = 14,
                            horizon: int = 5,
                            eps: float = 0.01) -> pd.DataFrame:

    df = df.copy()
    df = add_indicators(df, sma_windows=sma_windows, rsi_window=rsi_window)
    df = label_target(df, horizon=horizon, eps=eps)

    df.dropna(subset=[f'SMA_{sma_windows[-1]}', f'RSI_{rsi_window}', 'Target'], inplace=True)
    return df

merged_df = pd.read_csv('./sber_with_rates.csv', parse_dates=['Date'])
df_full = add_features_and_target(merged_df)
df_full.to_csv('./sber_with_features.csv', index=False)
