import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv('multi_ticker_with_lags_vol.csv', parse_dates=['Date'])
df = df.sort_values('Date').reset_index(drop=True)

X = df.drop(columns=['Date', 'Target', 'Target_bin'])
y = df['Target_bin']  # 1=BUY, 0=SELL+HOLD

split = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

tscv = TimeSeriesSplit(n_splits=5)

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

rf_cv_preds = np.zeros_like(y_train)
for train_idx, val_idx in tscv.split(X_train_scaled):
    rf.fit(X_train_scaled[train_idx], y_train.iloc[train_idx])
    rf_cv_preds[val_idx] = rf.predict(X_train_scaled[val_idx])

print("=== RandomForest CV Report ===")
print(classification_report(y_train, rf_cv_preds, target_names=['Non-BUY','BUY']))
print("Confusion Matrix (CV):")
print(confusion_matrix(y_train, rf_cv_preds))

rf.fit(X_train_scaled, y_train)
y_pred_rf = rf.predict(X_test_scaled)

print("\n=== RandomForest Test Report ===")
print(classification_report(y_test, y_pred_rf, target_names=['Non-BUY','BUY']))
print("Confusion Matrix (Test):")
print(confusion_matrix(y_test, y_pred_rf))

xgb = XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    random_state=42,
    tree_method='hist'
)

xgb_cv_preds = np.zeros_like(y_train)
for train_idx, val_idx in tscv.split(X_train_scaled):
    xgb.fit(
        X_train_scaled[train_idx], 
        y_train.iloc[train_idx],
        verbose=False
    )
    xgb_cv_preds[val_idx] = xgb.predict(X_train_scaled[val_idx])

print("\n=== XGBoost CV Report ===")
print(classification_report(y_train, xgb_cv_preds, target_names=['Non-BUY','BUY']))
print("Confusion Matrix (CV):")
print(confusion_matrix(y_train, xgb_cv_preds))

xgb.fit(X_train_scaled, y_train)
y_pred_xgb = xgb.predict(X_test_scaled)

print("\n=== XGBoost Test Report ===")
print(classification_report(y_test, y_pred_xgb, target_names=['Non-BUY','BUY']))
print("Confusion Matrix (Test):")
print(confusion_matrix(y_test, y_pred_xgb))