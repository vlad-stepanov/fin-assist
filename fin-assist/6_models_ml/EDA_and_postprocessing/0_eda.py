import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('../../3_postprocessing/rename_target_col/multi_ticker_dataset_encoded.csv', parse_dates=['Date']) # trimmed теперь не нужен

df_info = pd.DataFrame({
    'Column': df.columns,
    'Dtype': df.dtypes.astype(str),
    'Non-Null Count': df.notna().sum().values
})
print("==ИНФОРМАЦИЯ==")
print(df_info)

df_describe = df.describe()
print("==ОПИСАНИЕ==")
print(df_describe)

features = ['SBER_Close', 'IMOEX_Close', 'USD_RUB_Close', 'BRENT_Close', 'SBER_RSI_14']
for feat in features:
    plt.figure()
    df[feat].hist()
    plt.title(f'Histogram of {feat}')
    plt.xlabel(feat)
    plt.ylabel('Frequency')
    plt.show()

corr = df.select_dtypes(include='number').corr()
plt.figure(figsize=(8, 6))
plt.imshow(corr, aspect='auto')
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.index)), corr.index)
plt.title('Correlation Matrix')
plt.colorbar()
plt.show()
