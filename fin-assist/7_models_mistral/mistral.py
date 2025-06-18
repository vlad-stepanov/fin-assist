import os
import json
from mistralai import Mistral


client = Mistral(api_key="MISTRAL_API_KEY")


def build_portfolio_prompt(data, include_date=False, user_recommendations=None, tickers_filter=None):
    
    portfolio = data.get("portfolio", [])
    predictions = data.get("predictions", {})
    missing = data.get("missing_models", [])

    lines = [
        "Ты — финансовый аналитик. Сгенерируй одностраничный отчёт по портфелю пользователя с аналитикой ситуации на рынке и рекомендациями."
    ]

    if missing:
        lines.append(
            "К сожалению, прогнозы недоступны для следующих тикеров (отсутствуют обученные модели): "
            + ", ".join(missing)
        )
    else:
        lines.append("Для всех позиций портфеля найдены обученные модели.")

    if predictions:
        lines.append("Предсказания моделей для доступных тикеров:")
        for ticker, info in predictions.items():
            signal = info.get("signal", "—")
            confidence = info.get("confidence", None)
            if confidence is not None:
                lines.append(f"- {ticker}: сигнал «{signal}», достоверность {confidence:.2f}")
            else:
                lines.append(f"- {ticker}: сигнал «{signal}»")
    else:
        lines.append("Пока нет доступных предсказаний для каких-либо тикеров.")

    if tickers_filter:
        lines.append(f"Анализировать только следующие тикеры: {', '.join(tickers_filter)}")
        filtered = [item for item in portfolio if item['ticker'] in tickers_filter]
    else:
        lines.append("Портфель пользователя:")
        filtered = portfolio

    for item in filtered:
        line = f"- {item['ticker']}: {item['quantity']} шт. по цене {item['purchase_price']}"
        if include_date and item.get('purchase_date'):
            line += f" (дата покупки: {item['purchase_date']})"
        lines.append(line)

    if user_recommendations:
        lines.append("Пользовательские рекомендации/предпочтения:")
        for rec in user_recommendations:
            lines.append(f"- {rec}")

    lines.append("Структура отчёта: Введение, Анализ текущей ситуации, Выводы и Рекомендации.")
    return "\n".join(lines)


def get_financial_report(data, include_date=False, user_recommendations=None,
                         tickers_filter=None, temperature=0.2, max_tokens=3000):

    prompt = build_portfolio_prompt(
        data,
        include_date=include_date,
        user_recommendations=user_recommendations,
        tickers_filter=tickers_filter
    )
    resp = client.chat.complete(
        model="mistral-small-latest",
        messages=[{"role": "system", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens
    )
    return resp.choices[0].message.content


def load_data_from_json():
    path = input("Укажите путь до JSON-файла (портфель + предсказания): ").strip()
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("Ожидаем JSON-объект с ключами 'portfolio', 'predictions', 'missing_models'")
        if "portfolio" not in data or not isinstance(data["portfolio"], list):
            raise ValueError("Ключ 'portfolio' должен присутствовать и быть списком")
        data.setdefault("predictions", {})
        data.setdefault("missing_models", [])
        return data
    except Exception as e:
        print(f"Ошибка при загрузке данных: {e}")
        return load_data_from_json()


def main():
    print("=== Генератор финансового отчёта ===")
    data = load_data_from_json()

    include_date = input("Включать даты покупки в отчёт? (y/N): ").strip().lower() == 'y'

    user_recs = []
    print("Введите рекомендации пользователя по одному (Enter для окончания):")
    while True:
        rec = input("- ").strip()
        if not rec:
            break
        user_recs.append(rec)
    if not user_recs:
        user_recs = None

    tickers_filter = None
    if input("Хотите ли анализировать только часть портфеля? (y/N): ").strip().lower() == 'y':
        tickers_filter = input("Укажите через запятую нужные тикеры: ").strip().upper().split(',')
        tickers_filter = [t.strip() for t in tickers_filter if t.strip()]

    print("\nГенерируется отчёт, подождите...\n")
    report = get_financial_report(
        data,
        include_date=include_date,
        user_recommendations=user_recs,
        tickers_filter=tickers_filter
    )

    output_path = input("Укажите имя выходного текстового файла (например, report.txt): ").strip()
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Отчёт успешно сохранён в {output_path}")
    except Exception as e:
        print(f"Ошибка при сохранении файла: {e}")
        print("Выводим отчёт в консоль:\n")
        print(report)


if __name__ == '__main__':
    main()
