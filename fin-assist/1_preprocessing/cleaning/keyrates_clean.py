import pandas as pd

df = pd.read_csv('../../0_fetch/key_rate_cbr.csv')

df['Rate'] = df['Rate'] / 10000

output_path = 'cleaned_key_rate_cbr.csv'
df.to_csv(output_path, index=False)

print(f'Преобразованный датасет сохранён в файл: {output_path}')
