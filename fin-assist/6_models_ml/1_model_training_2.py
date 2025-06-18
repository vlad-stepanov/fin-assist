import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint

df = pd.read_csv('multi_ticker_dataset_trimmed.csv', parse_dates=['Date'])
df = df.sort_values('Date').reset_index(drop=True)

df = df.drop(columns=['begin','end'], errors='ignore')

df['SBER_pct']    = (df['SBER_Close']    - df['SBER_Open'])    / df['SBER_Open']
df['BRENT_pct']   = (df['BRENT_Close']   - df['BRENT_Open'])   / df['BRENT_Open']
df['USD_pct']     = (df['USD_RUB_Close'] - df['USD_RUB_Open']) / df['USD_RUB_Open']
df['IMOEX_pct']   = (df['IMOEX_Close']   - df['IMOEX_Open'])   / df['IMOEX_Open']
df['RATIO_SB_US'] = df['SBER_Close'] / df['USD_RUB_Close']

drop_cols = ['Date', 'Target']
features = [
    'SBER_Close','SBER_Volume','RSI_14','SMA_10','SMA_20','SMA_50',
    'IMOEX_pct','USD_pct','BRENT_pct','RATIO_SB_US'
]
X = df[features]
y = df['Target']

split = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

tscv = TimeSeriesSplit(n_splits=5)

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
y_pred = best_rf.predict(X_test_scaled)

from sklearn.metrics import classification_report, confusion_matrix
print("\n=== Test Classification Report ===")
print(classification_report(y_test, y_pred, target_names=['SELL','HOLD','BUY']))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
