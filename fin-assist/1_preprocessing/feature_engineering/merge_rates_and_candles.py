import pandas as pd

# # 1) Считаем файлы
# sber = pd.read_csv('sber_candles.csv', parse_dates=['begin', 'end'])
# rates = pd.read_csv('rates.csv', parse_dates=['Date'])

# # 2) Заводим в sber столбец Date из begin
# sber['Date'] = sber['begin'].dt.date  # тип object

# # 3) Приводим rates.Date тоже к object, оставляя только дату
# rates['Date'] = rates['Date'].dt.date

# # 4) Объединяем
# merged = pd.merge(sber, rates, on='Date', how='left')

# # 5) Заполняем пропуски вперед
# merged['Rate'].fillna(method='ffill', inplace=True)

# # 6) Сохраняем или работаем дальше
# merged.to_csv('sber_with_rates.csv', index=False)

# 1) Считаем файлы
sber = pd.read_csv('../cleaning/sber_clean.csv', parse_dates=['Date'])
rates = pd.read_csv('../cleaning/cleaned_key_rate_cbr.csv', parse_dates=['Date'])

# 2) Заводим в sber столбец Date из begin
sber['Date'] = sber['Date'].dt.date  # тип object

# 3) Приводим rates.Date тоже к object, оставляя только дату
rates['Date'] = rates['Date'].dt.date

# 4) Объединяем
merged = pd.merge(sber, rates, on='Date', how='left')

# 5) Заполняем пропуски вперед и назад
merged['Rate'].fillna(method='ffill', inplace=True)
merged['Rate'].fillna(method='bfill', inplace=True)

# 6) Сохраняем или работаем дальше
merged.to_csv('./sber_with_rates_with_bfill.csv', index=False)