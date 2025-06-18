import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
from scipy.stats import randint, uniform

df = pd.read_csv('multi_ticker_with_lags_vol.csv', parse_dates=['Date'])
df = df.sort_values('Date').reset_index(drop=True)

X = df.drop(columns=['Date', 'Target', 'Target_bin'])
y = df['Target_bin']  # 1=BUY, 0=SELL+HOLD

split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

tscv = TimeSeriesSplit(n_splits=5)
for fold, (tr, val) in enumerate(tscv.split(X_train_scaled), 1):
    print(f"Fold {fold}: train indices {tr[0]}–{tr[-1]}, valid indices {val[0]}–{val[-1]}")

print("Class distribution:\n", df['Target_bin'].value_counts(normalize=True))


rf = RandomForestClassifier(
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
param_dist = {
    'n_estimators': randint(100, 500),
    'max_depth':    randint(5, 30),
    'min_samples_leaf': randint(1, 10),
}
rs = RandomizedSearchCV(
    rf,
    param_distributions=param_dist,
    n_iter=20,
    cv=tscv,
    scoring='f1_macro',
    verbose=1,
    random_state=42
)

rf_cv_preds = np.zeros_like(y_train)
for train_idx, val_idx in tscv.split(X_train_scaled):
    rs.fit(X_train_scaled[train_idx], y_train.iloc[train_idx])
    rf_cv_preds[val_idx] = rs.predict(X_train_scaled[val_idx])

xgb = XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=42,
    tree_method='hist',
    device='CUDA'
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