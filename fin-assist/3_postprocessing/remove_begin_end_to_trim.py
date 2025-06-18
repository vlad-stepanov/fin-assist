import pandas as pd

# 1. Читаем датасет
df = pd.read_csv('multi_ticker_dataset_encoded.csv', parse_dates=['Date'], encoding='utf-8-sig')

# 2. Убираем ненужные колонки
df = df.drop(columns=['begin', 'end'], errors='ignore')

# 3. Сохраняем результат
df.to_csv('multi_ticker_dataset_trimmed.csv', index=False)
print("Сохранено в multi_ticker_dataset_trimmed.csv — форма:", df.shape)
