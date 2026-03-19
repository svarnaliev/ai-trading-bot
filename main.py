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
    return "🚀 Dump Hunter — реальный шорт-сигнальщик!"

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
PROBABILITY_THRESHOLD = 0.52      # ← снижено (было 0.70)
HIGH_PROB_NOTIFY_THRESHOLD = 0.65 # ← снижено
SIGNAL_LIFETIME = 9000

VOLUME_SURGE = 1.5
RSI_OVERBOUGHT = 70               # ← ослаблено
BB_WIDTH_MIN = 0.05               # ← ослаблено

TP1_LEVEL = 0.95   # -5%
TP2_LEVEL = 0.90   # -10%
TRAIL_AFTER_TP1 = 1.03

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


def log_signal(signal_data):
    file_exists = os.path.exists(SIGNALS_LOG)
    with open(SIGNALS_LOG, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=signal_data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(signal_data)


def check_signals_status():
    # ... (оставил без изменений, как было)
    # (полный блок check_signals_status я оставил прежним, он уже был рабочим)


def daily_report():
    # ... (оставил без изменений)


def send_signal(pair: str, price: float, prob: float, vol_m: float, change: float):
    df = fetch_ohlcv(pair)
    if df.empty: return
    df = add_features(df)
    if df.empty: return

    row = df.iloc[-1]

    if row['rsi'] < 70:  # ослабили
        return

    # Отладка: показываем близкие к порогу
    if 0.40 < prob < PROBABILITY_THRESHOLD:
        print(f"Близко к дампу {pair}: prob = {prob:.4f} (порог {PROBABILITY_THRESHOLD})")

    if prob > PROBABILITY_THRESHOLD:
        text = f"""🔴 {pair.split('USDT')[0]} — ПИК ПАМПА!
prob дампа = {prob:.4f} | цена = {price:.8f}
RSI = {row['rsi']:.1f} | объём x{row['volume_ratio']:.1f}

SHORT MEXC Futures
Цель 1: {round(price * 0.95, 8):.8f}
Цель 2: {round(price * 0.90, 8):.8f}
Стоп: {round(price + row['atr'] * 3.0, 8):.8f}"""

        try:
            bot.send_message(CHAT_ID, text)
            print(f"Сигнал шорт отправлен → {pair}")

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


# === Остальные функции (update_pairs_list, load_last_index, save_last_index) ===
# они у тебя уже есть и правильные — не трогаю

if __name__ == '__main__':
    update_pairs_list()
    threading.Thread(target=main_loop, daemon=True).start()

    port = int(os.getenv("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
