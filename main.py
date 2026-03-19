import os
import time
import io
import threading
from datetime import datetime, timedelta
import csv

import ccxt
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from telegram import Bot
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

from flask import Flask

app = Flask(__name__)

@app.route('/')
def home():
    return "Dump Hunter работает!"

@app.route('/ping')
def ping():
    return "pong"

# Константы
TIMEFRAME = '1h'
MODEL_FILE = 'catboost_dump_v2.cbm'
LAST_INDEX_FILE = 'last_pair_index.txt'
DATASET_FILE = 'dump_dataset_v2.csv'
SIGNALS_LOG = 'signals_log.csv'

MIN_DATA_LENGTH = 50
PROBABILITY_THRESHOLD = 0.52
HIGH_PROB_NOTIFY_THRESHOLD = 0.65
SIGNAL_LIFETIME = 9000  # 2.5 часа

VOLUME_SURGE = 1.5
RSI_OVERBOUGHT = 70
BB_WIDTH_MIN = 0.05

TP1_LEVEL = 0.95   # -5%
TP2_LEVEL = 0.90   # -10%
TRAIL_AFTER_TP1 = 1.03  # +3% в безубыток для шорта

FEATURES = ['ema200', 'rsi', 'macd', 'bb_lower', 'price_change', 'volume_change', 'bb_width']

TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
CHAT_ID = os.getenv('CHAT_ID')
MEXC_API_KEY = os.getenv('MEXC_API_KEY')
MEXC_API_SECRET = os.getenv('MEXC_API_SECRET')

bot = Bot(token=TELEGRAM_TOKEN)

futures_exchange = ccxt.mexc({
    'apiKey': MEXC_API_KEY,
    'secret': MEXC_API_SECRET,
    'enableRateLimit': True,
    'options': {'defaultType': 'swap'},
})

PAIRS = []
ACTIVE_SIGNALS = []
last_report_time = time.time()


def fetch_ohlcv(symbol: str, limit: int = 2000):
    try:
        time.sleep(0.45)
        bars = futures_exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=limit)
        df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        print(f"Ошибка загрузки {symbol}: {e}")
        return pd.DataFrame()


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) < MIN_DATA_LENGTH:
        return df

    df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()
    delta = df['close'].diff(1)
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26

    sma20 = df['close'].rolling(20).mean()
    std20 = df['close'].rolling(20).std()
    df['bb_lower'] = sma20 - 2*std20
    df['bb_upper'] = sma20 + 2*std20
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['close']

    df['price_change'] = df['close'].pct_change()
    df['volume_change'] = df['volume'].pct_change()

    return df.dropna()


def load_or_train_model():
    if os.path.exists(MODEL_FILE):
        print("Загружаем существующую модель дамп-бота...")
        model = CatBoostClassifier()
        model.load_model(MODEL_FILE)
        return model

    if not os.path.exists(DATASET_FILE):
        print(f"Файл {DATASET_FILE} не найден!")
        return None

    print(f"Обучение дамп-бота на {DATASET_FILE}...")
    df_all = pd.read_csv(DATASET_FILE)
    if df_all.empty:
        print("Датасет пуст!")
        return None

    X = df_all[FEATURES]
    y = df_all['target']

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    model = CatBoostClassifier(iterations=1200, depth=8, learning_rate=0.04, verbose=0)
    model.fit(X_tr, y_tr)

    acc = accuracy_score(y_te, model.predict(X_te))
    print(f"Дамп-модель обучена | Accuracy: {acc:.4f} ({acc*100:.2f}%) | Строк: {len(df_all)}")

    model.save_model(MODEL_FILE)
    return model


def get_market_data(symbol):
    try:
        ticker = futures_exchange.fetch_ticker(symbol)
        return ticker['last'], ticker.get('percentage', 0), round(ticker.get('quoteVolume', 0) / 1_000_000, 1)
    except Exception as e:
        print(f"Ошибка get_market_data {symbol}: {e}")
        return 0.0, 0.0, 0.0


def get_funding_rate(symbol):
    try:
        funding = futures_exchange.fetch_funding_rate(symbol)
        return funding.get('fundingRate', 0) * 100
    except Exception as e:
        print(f"Funding ошибка {symbol}: {e}")
        return 0.0


def send_funding_update(pair, funding_rate):
    sign = "📈" if funding_rate > 0 else "📉"
    text = f"Фандинг {pair}: {sign} {funding_rate:.4f}%"
    if funding_rate > 0.015:
        text += " (вы платите)"
    elif funding_rate < -0.01:
        text += " (вам платят)"
    bot.send_message(CHAT_ID, text)


def log_signal(signal_data):
    file_exists = os.path.exists(SIGNALS_LOG)
    with open(SIGNALS_LOG, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=signal_data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(signal_data)


def check_signals_status():
    now = time.time()
    to_remove = []
    report = []

    for s in ACTIVE_SIGNALS:
        pair = s['pair']
        entry = s['entry_price']
        atr = s['atr']
        max_price = s.get('max_price', entry)
        trail_sl = s.get('trail_sl', entry + atr * 3.0)
        tp1_hit = s.get('tp1_hit', False)

        try:
            price, _, _ = get_market_data(pair)
            if price <= 0:
                continue

            if price < max_price:
                s['max_price'] = price
                new_sl = price + atr * 3.0
                if new_sl < trail_sl:
                    s['trail_sl'] = new_sl

            if not tp1_hit and price <= entry * TP1_LEVEL:
                s['tp1_hit'] = True
                s['trail_sl'] = entry * TRAIL_AFTER_TP1
                report.append(f"TP1 (SHORT): {pair} | {price:.8f}")

            if price <= entry * TP2_LEVEL:
                report.append(f"TP2 (SHORT): {pair} | {price:.8f}")
                to_remove.append(s)
                continue

            if price >= trail_sl:
                report.append(f"SL (SHORT): {pair} | {price:.8f}")
                to_remove.append(s)
                continue

            if now - s['timestamp'] > SIGNAL_LIFETIME:
                report.append(f"Тайм-аут: {pair} | {price:.8f}")
                to_remove.append(s)
                continue

        except Exception as e:
            print(f"Ошибка проверки {pair}: {e}")

    for s in to_remove:
        ACTIVE_SIGNALS.remove(s)

    if report:
        bot.send_message(CHAT_ID, "\n".join(report))


def daily_report():
    global last_report_time
    now = time.time()
    if now - last_report_time < 86400:
        return

    if not os.path.exists(SIGNALS_LOG):
        return

    df = pd.read_csv(SIGNALS_LOG)
    if df.empty:
        return

    total = len(df)
    tp_hit = len(df[df['status'].str.contains('tp', na=False)])
    sl_hit = len(df[df['status'] == 'sl_hit'])
    timeout = len(df[df['status'] == 'timeout'])
    winrate = tp_hit / total * 100 if total > 0 else 0

    text = f"""📊 Отчёт за 24 часа (DUMP)
Всего сигналов: {total}
TP достигнуто: {tp_hit} ({winrate:.1f}%)
SL сработал: {sl_hit}
Тайм-аут: {timeout}
Активных позиций: {len(ACTIVE_SIGNALS)}"""

    bot.send_message(CHAT_ID, text)
    last_report_time = now


def send_signal(pair: str, price: float, prob: float, vol_m: float, change: float):
    df = fetch_ohlcv(pair)
    if df.empty: return
    df = add_features(df)
    if df.empty: return

    row = df.iloc[-1]

    if row['rsi'] < 70:
        return

    if 0.40 < prob < PROBABILITY_THRESHOLD:
        print(f"Близко к дампу {pair}: prob = {prob:.4f}")

    if prob > PROBABILITY_THRESHOLD:
        text = f"""🔴 {pair.split('USDT')[0]} — ПИК ПАМПА!
prob дампа = {prob:.4f} | цена = {price:.8f}
RSI = {row['rsi']:.1f} | объём x{row['volume_ratio']:.1f}

SHORT MEXC Futures
Цель 1: {round(price * TP1_LEVEL, 8):.8f}
Цель 2: {round(price * TP2_LEVEL, 8):.8f}
Стоп: {round(price + row['atr'] * 3.0, 8):.8f}"""

        try:
            bot.send_message(CHAT_ID, text)
            print(f"Сигнал шорт → {pair}")

            ACTIVE_SIGNALS.append({
                'pair': pair,
                'entry_price': price,
                'atr': row['atr'],
                'timestamp': time.time(),
                'max_price': price,
                'trail_sl': price + row['atr'] * 3.0,
                'tp1_hit': False,
                'status': 'open'
            })

            log_signal({
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'pair': pair,
                'entry_price': price,
                'prob': prob,
                'rsi': row['rsi'],
                'v_ratio': row['volume_ratio'],
                'atr': row['atr'],
                'status': 'open'
            })

        except Exception as e:
            print(f"Ошибка отправки {pair}: {e}")


def update_pairs_list():
    global PAIRS
    for attempt in range(3):
        try:
            print(f"Попытка {attempt+1}/3 обновления списка пар...")
            markets = futures_exchange.load_markets(reload=True)
            futures_pairs = [s for s, m in markets.items() if m.get('swap') and 'USDT' in s and m.get('active')]
            new_pairs = sorted(futures_pairs, key=lambda s: float(markets[s].get('info', {}).get('quoteVolume', 0) or 0), reverse=True)
            print(f"Загружено новых пар: {len(new_pairs)}")
            if len(new_pairs) > 0:
                PAIRS[:] = new_pairs
                print(f"Список обновлён: {len(PAIRS)} пар")
                return
            else:
                print("Список пуст, ждём 5 сек...")
                time.sleep(5)
        except Exception as e:
            print(f"Ошибка {attempt+1}/3: {type(e).__name__} — {str(e)}")
            time.sleep(5)
    print("Все попытки провалились. Продолжаем со старым списком.")


def load_last_index():
    if os.path.exists(LAST_INDEX_FILE):
        try:
            with open(LAST_INDEX_FILE, 'r') as f:
                idx = int(f.read().strip())
                print(f"Загружен индекс: {idx}")
                return idx
        except:
            print("Ошибка чтения индекса, начинаем с 0")
            return 0
    print("Файл индекса не найден, начинаем с 0")
    return 0


def save_last_index(idx):
    try:
        with open(LAST_INDEX_FILE, 'w') as f:
            f.write(str(idx))
        print(f"Сохранён индекс: {idx}")
    except Exception as e:
        print(f"Ошибка сохранения индекса: {e}")


def main_loop():
    model = load_or_train_model()
    if model is None:
        print("Модель не загружена")
    else:
        print("Модель готова")

    bot.send_message(CHAT_ID, f"🚀 Dump Hunter запущен (prob > {PROBABILITY_THRESHOLD}) | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    iteration = 0
    last_funding_check = time.time()

    while True:
        iteration += 1
        now_str = datetime.now().strftime('%H:%M:%S')
        print(f"[{now_str}] Итерация {iteration} | пар: {len(PAIRS)}")

        update_pairs_list()
        check_signals_status()
        daily_report()

        if len(PAIRS) == 0:
            update_pairs_list()
            time.sleep(60)
            continue

        start_idx = load_last_index()
        if start_idx >= len(PAIRS):
            start_idx = 0
            save_last_index(0)

        scanned = 0
        for pair in PAIRS[start_idx:]:
            scanned += 1
            try:
                df = fetch_ohlcv(pair)
                df = add_features(df)
                if len(df) < MIN_DATA_LENGTH: continue

                row = df.iloc[-1]
                feats = row[FEATURES].values.reshape(1, -1)
                prob = model.predict_proba(feats)[0][1]

                print(f"{pair:20} prob = {prob:.4f}")

                if prob > HIGH_PROB_NOTIFY_THRESHOLD:
                    bot.send_message(CHAT_ID, f"Высокая вероятность дампа {pair}: prob = {prob:.4f}")

                if prob > PROBABILITY_THRESHOLD:
                    price, ch, vm = get_market_data(pair)
                    send_signal(pair, price, prob, vm, ch)

            except Exception as e:
                print(f"Ошибка {pair}: {e}")

            time.sleep(0.45)

        if time.time() - last_funding_check > 1800:
            for s in ACTIVE_SIGNALS[:]:
                try:
                    funding = get_funding_rate(s['pair'])
                    send_funding_update(s['pair'], funding)
                except:
                    pass
            last_funding_check = time.time()

        print(f"[{now_str}] Итерация завершена → сразу следующая")


if __name__ == '__main__':
    update_pairs_list()
    threading.Thread(target=main_loop, daemon=True).start()

    port = int(os.getenv("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
