import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.model_selection import train_test_split, cross_val_predict, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

# 1. Загрузка
df = pd.read_csv('multi_ticker_dataset_trimmed.csv', parse_dates=['Date'])

# 2. Удаляем избыточные колонки
drops = []
for prefix in ['SBER', 'IMOEX', 'USD_RUB', 'BRENT']:
    # убираем High и Low
    drops += [f'{prefix}_High', f'{prefix}_Low']
# не трогаем Close, Volume (где есть), Change_pct, RSI, SMA_*
# drops += ['Rate']  # Rate, begin/end уже убрали
df = df.drop(columns=drops, errors='ignore')
print(df.head())
# 3. Проверяем баланс классов
print("Class distribution:\n", df['Target'].value_counts(normalize=True))

# 4. Формируем X и y
X = df.drop(columns=['Date', 'Target'])
y = df['Target']

# 5. Хронологический сплит: первые 80% дат — train, последние 20% — test
split_idx = int(len(df) * 0.8)
train_idx, test_idx = df.index[:split_idx], df.index[split_idx:]

X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# 6. Масштабирование
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# 7. (Опционально) Подготовка TimeSeriesSplit для CV
tscv = TimeSeriesSplit(n_splits=5)
for fold, (tr, val) in enumerate(tscv.split(X_train_scaled), 1):
    print(f"Fold {fold}: train indices {tr[0]}–{tr[-1]}, valid indices {val[0]}–{val[-1]}")


# Допустим, X_train_scaled, y_train, X_test_scaled, y_test, tscv уже есть из предыдущего шага


# Предполагаем, что X_train_scaled, y_train, X_test_scaled, y_test, tscv уже подготовлены

def cvs_predict(model, X, y, tscv):
    """
    Возвращает массив предсказаний для каждого наблюдения X,
    полученных на валидационных фолдах TimeSeriesSplit.
    """
    y_pred = np.empty_like(y)
    for train_idx, val_idx in tscv.split(X):
        model.fit(X[train_idx], y.iloc[train_idx])
        y_pred[val_idx] = model.predict(X[val_idx])
    return y_pred

# 1) Logistic Regression
lr = LogisticRegression(
    multi_class='multinomial',
    solver='lbfgs',
    max_iter=1000,
    class_weight='balanced',
    random_state=42
)

y_pred_cv_lr = cvs_predict(lr, X_train_scaled, y_train, tscv)
print("=== Logistic Regression CV Report ===")
# print(classification_report(y_train, y_pred_cv_lr, target_names=['SELL','HOLD','BUY']))
print(classification_report(
    y_train,
    y_pred_cv_lr,
    labels=[0,1,2],
    target_names=['SELL','HOLD','BUY']
))


# Финальное обучение на всём train и тест
lr.fit(X_train_scaled, y_train)
y_pred_test_lr = lr.predict(X_test_scaled)
print("=== Logistic Regression Test Report ===")
print(classification_report(y_test, y_pred_test_lr, target_names=['SELL','HOLD','BUY']))
print("Confusion Matrix (Test):")
print(confusion_matrix(y_test, y_pred_test_lr))


# 2) Random Forest
rf = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced_subsample',
    random_state=42,
    n_jobs=-1
)

y_pred_cv_rf = cvs_predict(rf, X_train_scaled, y_train, tscv)
print("\n=== Random Forest CV Report ===")
# print(classification_report(y_train, y_pred_cv_rf, target_names=['SELL','HOLD','BUY']))
print(classification_report(
    y_train,
    y_pred_cv_rf,
    labels=[0,1,2],
    target_names=['SELL','HOLD','BUY']
))


rf.fit(X_train_scaled, y_train)
y_pred_test_rf = rf.predict(X_test_scaled)
print("=== Random Forest Test Report ===")
print(classification_report(y_test, y_pred_test_rf, target_names=['SELL','HOLD','BUY']))
print("Confusion Matrix (Test):")
print(confusion_matrix(y_test, y_pred_test_rf))
