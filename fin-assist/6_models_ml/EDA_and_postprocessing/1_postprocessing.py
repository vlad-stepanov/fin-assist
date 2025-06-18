import pandas as pd
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('../../3_postprocessing/rename_target_col/multi_ticker_dataset_encoded.csv', parse_dates=['Date']) # trimmed теперь не нужен

drops = []
for prefix in ['SBER', 'IMOEX', 'USD_RUB', 'BRENT']:
    drops += [f'{prefix}_High', f'{prefix}_Low']
df = df.drop(columns=drops, errors='ignore')
print(df.head())

print("Class distribution:\n", df['Target'].value_counts(normalize=True))

X = df.drop(columns=['Date', 'Target'])
y = df['Target']

split_idx = int(len(df) * 0.8)
train_idx, test_idx = df.index[:split_idx], df.index[split_idx:]

X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

tscv = TimeSeriesSplit(n_splits=5)
for fold, (tr, val) in enumerate(tscv.split(X_train_scaled), 1):
    print(f"Fold {fold}: train indices {tr[0]}–{tr[-1]}, valid indices {val[0]}–{val[-1]}")
