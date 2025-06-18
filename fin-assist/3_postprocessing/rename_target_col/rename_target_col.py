import pandas as pd

df = pd.read_csv('../multi_ticker_dataset.csv', parse_dates=['Date'], encoding='utf-8-sig')

target_map = {
    'SELL': 0,
    'HOLD': 1,
    'BUY':  2
}
df['Target'] = df['Target'].map(target_map)

df.to_csv('multi_ticker_dataset_encoded.csv', index=False)
print("Готово: saved to multi_ticker_dataset_encoded.csv")
