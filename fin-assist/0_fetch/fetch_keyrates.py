import pandas as pd

URL = (
    "https://www.cbr.ru/hd_base/KeyRate/"
    "?UniDbQuery.Posted=True"
    "&UniDbQuery.From=01.05.2015"
    "&UniDbQuery.To=31.05.2025"
)


def fetch_key_rate(url: str) -> pd.DataFrame:
    """
    Загружает таблицу ключевой ставки с сайта ЦБ РФ

    Args:
        url (str): Ссылка на страницу с историческими данными ключевой ставки

    Returns:
        pd.DataFrame: Датафрейм с колонками ['Дата', 'Ставка']
    """
    tables = pd.read_html(url, header=0)
    if not tables:
        raise ValueError("Не удалось найти таблицу на странице")

    df = tables[0]

    df = df.rename(columns={df.columns[0]: 'Date', df.columns[1]: 'Rate'})
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

    return df

if __name__ == '__main__':
    df_key_rate = fetch_key_rate(URL)
    print(df_key_rate)
    df_key_rate.to_csv('key_rate_cbr.csv', index=False, encoding='utf-8-sig')
    print("Данные успешно сохранены в key_rate_cbr.csv")
