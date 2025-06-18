import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from scipy.stats import randint

df = pd.read_csv('../../3_postprocessing/rename_target_col/multi_ticker_dataset_encoded.csv', parse_dates=['Date']) # trimmed теперь не нужен
df = df.sort_values('Date').reset_index(drop=True)

df['SBER_pct']    = (df['SBER_Close']    - df['SBER_Open'])    / df['SBER_Open']
df['BRENT_pct']   = (df['BRENT_Close']   - df['BRENT_Open'])   / df['BRENT_Open']
df['USD_pct']     = (df['USD_RUB_Close'] - df['USD_RUB_Open']) / df['USD_RUB_Open']
df['IMOEX_pct']   = (df['IMOEX_Close']   - df['IMOEX_Open'])   / df['IMOEX_Open']
df['RATIO_SB_US'] = df['SBER_Close'] / df['USD_RUB_Close']

print("Class distribution:\n", df['Target'].value_counts(normalize=True))

drop_cols = ['Date', 'Target']
features = [
    'SBER_Close','SBER_Volume','SBER_RSI_14','SBER_SMA_10','SBER_SMA_20','SBER_SMA_50',
    'IMOEX_pct','USD_pct','BRENT_pct','RATIO_SB_US'
]
X = df[features]
y = df['Target']

split_idx = int(len(df) * 0.8)

X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

tscv = TimeSeriesSplit(n_splits=5)
for fold, (tr, val) in enumerate(tscv.split(X_train_scaled), 1):
    print(f"Fold {fold}: train indices {tr[0]}–{tr[-1]}, valid indices {val[0]}–{val[-1]}")


param_dist = {
    'n_estimators': randint(100, 500),
    'max_depth':    randint(5, 30),
    'min_samples_leaf': randint(1, 10),
}
rf = RandomForestClassifier(
    class_weight='balanced_subsample',
    random_state=42,
    n_jobs=-1
)
rs = RandomizedSearchCV(
    rf,
    param_distributions=param_dist,
    n_iter=20,
    cv=tscv,
    scoring='f1_macro',
    verbose=1,
    random_state=42
)

rs.fit(X_train_scaled, y_train)

print("Best params:", rs.best_params_)
print("Best CV f1_macro:", rs.best_score_)

best_rf = rs.best_estimator_
y_pred_test_rf = best_rf.predict(X_test_scaled)

print("\n=== Random Forest Test Report ===")
print(classification_report(y_test, y_pred_test_rf, target_names=['SELL','HOLD','BUY']))
print("Confusion Matrix (Test):")
print(confusion_matrix(y_test, y_pred_test_rf))
