import pandas as pd
import numpy as np

def parse_euro_number(s):
    if pd.isna(s) or s == "":
        return np.nan
    s = str(s).strip().replace('%', '')
    s = s.replace('.', '').replace(',', '.')
    try:
        return float(s)
    except ValueError:
        return np.nan

def parse_volume(s):
    if pd.isna(s) or s == "":
        return np.nan
    s = s.strip()
    mult = 1
    if s.endswith('K'): mult = 1_000; s = s[:-1]
    elif s.endswith('M'): mult = 1_000_000; s = s[:-1]
    elif s.endswith('B'): mult = 1_000_000_000; s = s[:-1]
    return parse_euro_number(s) * mult

def show_cols(df, name):
    print(f"{name} columns:", df.columns.tolist())

rename_map = {
    "Дата": "Date",
    "Цена": "Price",
    "Откр.": "Open",
    "Макс.": "High",
    "Мин.": "Low",
    "Объём": "Volume",
    "Изм. %": "Change_pct"
}

def clean_file(fname, to, drop_volume=False):
    df = pd.read_csv(f"{fname}.csv", sep=",", encoding="utf-8-sig")
    show_cols(df, fname)
    df.rename(columns=rename_map, inplace=True)
    if drop_volume and "Volume" in df.columns:
        df.drop(columns=["Volume"], inplace=True)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True).dt.strftime("%Y-%m-%d")
    for col in ["Price", "Open", "High", "Low", "Change_pct"]:
        if col in df.columns:
            df[col] = df[col].apply(parse_euro_number)
    if "Volume" in df.columns:
        df["Volume"] = df["Volume"].apply(parse_volume)
    df.to_csv(f"{to}_clean.csv", index=False)
    print(f"Saved {to}_clean.csv — {df.shape[0]} строк, {df.shape[1]} столбцов.")

clean_file("../../0_fetch/brent", "./brent", drop_volume=False)
clean_file("../../0_fetch/imoex", "./imoex", drop_volume=True)
clean_file("../../0_fetch/usd_rub", "./usd_rub", drop_volume=True)
clean_file("../../0_fetch/sber", "./sber", drop_volume=False)

