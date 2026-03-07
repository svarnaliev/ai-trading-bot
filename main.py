import os
import time
import traceback
from datetime import datetime
import io

import ccxt
import pandas as pd
import pandas_ta as ta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from telegram import Bot
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

from keep_alive import keep_alive


# ────────────────────────────────────────────────
#  Константы
# ────────────────────────────────────────────────

TIMEFRAME = '1h'
INTERVAL_SECONDS = 900
MODEL_FILE = 'catboost_model.cbm'

MIN_DATA_LENGTH = 50
PROBABILITY_THRESHOLD = 0.65

FEATURES = ['ema200', 'rsi', 'macd', 'bb_lower', 'price_change', 'volume_change']


# ────────────────────────────────────────────────
#  Инициализация — отдельный exchange для фьючерсов
# ────────────────────────────────────────────────

TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
CHAT_ID = os.getenv('CHAT_ID')
MEXC_API_KEY = os.getenv('MEXC_API_KEY')
MEXC_API_SECRET = os.getenv('MEXC_API_SECRET')

bot = Bot(token=TELEGRAM_TOKEN)

# Exchange для фьючерсов (swap = perpetual USDT-M)
futures_exchange = ccxt.mexc({
    'apiKey': MEXC_API_KEY,
    'secret': MEXC_API_SECRET,
    'enableRateLimit': True,
    'options': {
        'defaultType': 'swap',  # perpetual futures
    },
})

PAIRS = []  # будет заполнено фьючерсами


# ────────────────────────────────────────────────
#  Данные и фичи (используем futures_exchange)
# ────────────────────────────────────────────────

def fetch_ohlcv(symbol: str, limit: int = 800) -> pd.DataFrame:
    try:
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

    df['ema200']       = ta.ema(df['close'], length=200)
    df['rsi']          = ta.rsi(df['close'], length=14)
    df['macd']         = ta.macd(df['close'])['MACD_12_26_9']
    bb                 = ta.bbands(df['close'], length=20, std=2.0)
    df['bb_lower']     = bb.iloc[:, 0]
    df['price_change'] = df['close'].pct_change()
    df['volume_change']= df['volume'].pct_change()

    return df.dropna()


# ────────────────────────────────────────────────
#  Модель (обучение на споте, но предсказание на фьючерсах — ок для теста)
# ────────────────────────────────────────────────

def load_or_train_model() -> CatBoostClassifier:
    if os.path.exists(MODEL_FILE):
        print("Загружаем модель...")
        model = CatBoostClassifier()
        model.load_model(MODEL_FILE)
        return model

    print("Обучение модели...")
    all_data = []
    training_pairs = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT',
                      'ADA/USDT', 'DOGE/USDT', 'SHIB/USDT', 'AVAX/USDT', 'TRX/USDT']
    for symbol in training_pairs:
        df = fetch_ohlcv(symbol)
        df = add_features(df)
        if df.empty: continue
        df['target'] = (
            (df['close'].shift(-1) < df['ema200'].shift(-1)) &
            (df['price_change'].shift(-1) < -0.01)
        ).astype(int)
        all_data.append(df)

    df_all = pd.concat(all_data).dropna()
    X = df_all[FEATURES]
    y = df_all['target']

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    model = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.05, verbose=0)
    model.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, model.predict(X_te))
    print(f"Модель обучена | Accuracy: {acc:.4f}")
    model.save_model(MODEL_FILE)
    return model


# ────────────────────────────────────────────────
#  Рыночные данные (тоже на фьючерсах)
# ────────────────────────────────────────────────

def get_market_data(symbol: str):
    try:
        ticker = futures_exchange.fetch_ticker(symbol)
        price = ticker['last']
        change = ticker.get('percentage', 0)
        vol = ticker.get('quoteVolume', 0)
        vol_m = round(vol / 1_000_000, 1)
        return price, change, vol_m
    except Exception as e:
        print(f"Ошибка тикера {symbol}: {e}")
        return 0.0, 0.0, 0.0


# ────────────────────────────────────────────────
#  Сигнал (без изменений)
# ────────────────────────────────────────────────

def build_signal_text(pair: str, price: float, prob: float, vol_m: float, change: float) -> str:
    coin = pair.replace('USDT', '')  # для фьючерсов без /
    strength = "СИЛЬНЫЙ" if prob > 0.85 else "СРЕДНИЙ" if prob > 0.75 else "СЛАБЫЙ"
    fires    = "🔥🔥🔥" if prob > 0.85 else "🔥🔥" if prob > 0.75 else "🔥"
    pos_size = round(price * 200 * 50, 0)

    return f"""🔴 {coin} {fires} {strength}
x200 / {pos_size}$ / {vol_m}M / {change:+.4f}

Trade: Mexc Futures

Направление: Разворот ВНИЗ
Действие: SHORT

Текущая цена: {price}
Цель 1: {round(price * 0.95, 6)}
Цель 2: {round(price * 0.90, 6)}

Уверенность: {int(prob * 100)}%
Сила сигнала: {int(prob * 100 - 20)}/100
Общий Score: {int(prob * 100 + int(prob * 100 - 20))}"""


def create_chart(pair: str) -> io.BytesIO | None:
    df = fetch_ohlcv(pair)
    if df.empty: return None
    df = add_features(df)
    if df.empty: return None

    fig, ax = plt.subplots(figsize=(10, 5), facecolor='#1e1e1e')
    ax.plot(df['timestamp'], df['close'], color='white', lw=1.2)
    ax.plot(df['timestamp'], df['ema200'], color='orange', lw=1.5)
    ax.bar(df['timestamp'], df['volume'], color='gray', alpha=0.4)
    ax.set_title(f'{pair} — Разворот ВНИЗ', color='white')
    ax.grid(True, alpha=0.25, color='gray')
    ax.tick_params(colors='white')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', facecolor='#1e1e1e', bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return buf


def send_signal(pair: str, price: float, prob: float, vol_m: float, change: float):
    # if vol_m < 50: return

    text = build_signal_text(pair, price, prob, vol_m, change)
    buf = create_chart(pair)
    if buf is None:
        print(f"График не создан: {pair}")
        return

    try:
        bot.send_photo(chat_id=CHAT_ID, photo=buf, caption=text)
        print(f"Сигнал отправлен → {pair}  ({int(prob*100)}%)")
    except Exception as e:
        print(f"Ошибка отправки {pair}: {e}")


# ────────────────────────────────────────────────
#  Обновление списка пар (для фьючерсов)
# ────────────────────────────────────────────────

def update_pairs_list():
    try:
        markets = futures_exchange.load_markets(reload=True)

        futures_pairs = []
        for symbol, market in markets.items():
            if market.get('swap', False) and market.get('linear', False) and 'USDT' in symbol and market.get('active', True):
                futures_pairs.append(symbol)

        sorted_pairs = sorted(
            futures_pairs,
            key=lambda s: float(markets[s].get('info', {}).get('quoteVolume', '0') or 0),
            reverse=True
        )

        PAIRS[:] = sorted_pairs[:1000]

        print(f"Загружено {len(PAIRS)} USDT-M Perpetual Futures пар (отсортировано по объёму)")
        if len(PAIRS) > 0:
            print("Первые 5 пар:", PAIRS[:5])
        else:
            print("Фьючерсы НЕ найдены! Первые 10 символов markets:", list(markets.keys())[:10])

    except Exception as e:
        print(f"Ошибка обновления списка фьючерсных пар: {e}")


# ────────────────────────────────────────────────
#  Основной цикл
# ────────────────────────────────────────────────

def main_loop():
    model = load_or_train_model()
    last_retrain = time.time()

    print("Тест Telegram...")
    try:
        bot.send_message(
            chat_id=CHAT_ID,
            text=f"🤖 Бот запущен (Futures) | {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        )
        print("Тестовое сообщение отправлено!")
    except Exception as e:
        print(f"Telegram ошибка: {e}")

    while True:
        update_pairs_list()

        for pair in PAIRS:
            try:
                df = fetch_ohlcv(pair)
                df = add_features(df)
                if len(df) < MIN_DATA_LENGTH:
                    continue

                row = df.iloc[-1]
                feats = row[FEATURES].values.reshape(1, -1)
                prob = model.predict_proba(feats)[0][1]

                print(f"{pair:12}  prob = {prob:.4f}")

                if prob > PROBABILITY_THRESHOLD:
                    price, ch, vm = get_market_data(pair)
                    send_signal(pair, price, prob, vm, ch)

            except Exception as e:
                print(f"Ошибка {pair}: {type(e).__name__}")

            time.sleep(0.35)

        if time.time() - last_retrain > 6 * 3600:
            print("Переобучение модели...")
            model = load_or_train_model()
            last_retrain = time.time()

        time.sleep(INTERVAL_SECONDS)


if __name__ == '__main__':
    keep_alive()
    print("🚀 Бот запущен — сканируем USDT-M Futures MEXC")
    main_loop()
