import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from scipy.stats import randint, uniform
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv('multi_ticker_dataset_trimmed.csv', parse_dates=['Date'])
df = df.sort_values('Date').reset_index(drop=True)

df['SBER_pct']    = (df['SBER_Close']    - df['SBER_Open'])    / df['SBER_Open']
df['BRENT_pct']   = (df['BRENT_Close']   - df['BRENT_Open'])   / df['BRENT_Open']
df['USD_pct']     = (df['USD_RUB_Close'] - df['USD_RUB_Open']) / df['USD_RUB_Open']
df['IMOEX_pct']   = (df['IMOEX_Close']   - df['IMOEX_Open'])   / df['IMOEX_Open']
df['RATIO_SB_US'] = df['SBER_Close'] / df['USD_RUB_Close']

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

xgb = XGBClassifier(
    objective='multi:softmax',
    num_class=3,
    tree_method='hist',
    use_label_encoder=False,
    eval_metric='mlogloss',
    random_state=42
)
param_dist = {
    'n_estimators': randint(50, 500),
    'max_depth':    randint(3, 15),
    'learning_rate': uniform(0.01, 0.3),
    'subsample':     uniform(0.5, 0.5),
    'colsample_bytree': uniform(0.5, 0.5),
    'gamma':         uniform(0, 1),
    'min_child_weight': randint(1, 10)
}

rs_xgb = RandomizedSearchCV(
    xgb,
    param_distributions=param_dist,
    n_iter=30,
    cv=tscv,
    scoring='f1_macro',
    verbose=1,
    random_state=42,
    n_jobs=-1
)
rs_xgb.fit(X_train_scaled, y_train)

print("Best XGB params:", rs_xgb.best_params_)
print("Best CV f1_macro:", rs_xgb.best_score_)

best_xgb = rs_xgb.best_estimator_
y_pred = best_xgb.predict(X_test_scaled)

print("\n=== XGBoost Test Classification Report ===")
print(classification_report(y_test, y_pred, target_names=['SELL','HOLD','BUY']))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

import joblib

joblib.dump(best_xgb, 'best_xgb_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Модель сохранена в best_xgb_model.pkl\nСкейлер - в scaler.pkl")