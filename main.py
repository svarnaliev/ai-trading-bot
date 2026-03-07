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
import matplotlib.dates as mdates

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
PROBABILITY_THRESHOLD = 0.25

FEATURES = ['ema200', 'rsi', 'macd', 'bb_lower', 'price_change', 'volume_change']


# ────────────────────────────────────────────────
#  Инициализация
# ────────────────────────────────────────────────

TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
CHAT_ID = os.getenv('CHAT_ID')
MEXC_API_KEY = os.getenv('MEXC_API_KEY')
MEXC_API_SECRET = os.getenv('MEXC_API_SECRET')

bot = Bot(token=TELEGRAM_TOKEN)

futures_exchange = ccxt.mexc({
    'apiKey': MEXC_API_KEY,
    'secret': MEXC_API_SECRET,
    'enableRateLimit': True,
    'options': {
        'defaultType': 'swap',
    },
})

PAIRS = []


# ────────────────────────────────────────────────
#  Данные и фичи
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
#  Модель (переобучена на фьючерсах)
# ────────────────────────────────────────────────

def load_or_train_model() -> CatBoostClassifier:
    # Принудительно удаляем старую модель, чтобы переобучить на фьючерсах
    if os.path.exists(MODEL_FILE):
        print("Удаляем старую модель для переобучения на фьючерсах...")
        os.remove(MODEL_FILE)

    print("Обучение модели на фьючерсных данных...")
    all_data = []
    training_pairs = [
        'BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT', 'XRP/USDT:USDT',
        'BNB/USDT:USDT', 'ADA/USDT:USDT', 'DOGE/USDT:USDT', 'AVAX/USDT:USDT',
        'TRX/USDT:USDT', 'TON/USDT:USDT', 'NEAR/USDT:USDT', 'SUI/USDT:USDT'
    ]
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
    print(f"Модель обучена на фьючерсах | Accuracy: {acc:.4f}")
    model.save_model(MODEL_FILE)
    return model


# ────────────────────────────────────────────────
#  Рыночные данные
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
#  Сигнал
# ────────────────────────────────────────────────

def build_signal_text(pair: str, price: float, prob: float, vol_m: float, change: float) -> str:
    coin = pair.split('/')[0].replace(':USDT', '')
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

    fig, ax = plt.subplots(figsize=(10, 6), facecolor='#0d1117')

    ax.plot(df['timestamp'], df['close'], color='#00ff9d', linewidth=2, label='Цена')
    ax.plot(df['timestamp'], df['ema200'], color='#ff4444', linewidth=1.8, label='EMA200')

    ax_vol = ax.twinx()
    ax_vol.bar(df['timestamp'], df['volume'], color='gray', alpha=0.35, width=0.0008)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %H:%M'))
    plt.xticks(rotation=45)
    ax.grid(True, alpha=0.15, color='gray')

    ax.set_title(f'{pair} — Разворот ВНИЗ', color='white', fontsize=14, pad=15)
    ax.set_facecolor('#0d1117')
    ax.tick_params(colors='white')
    ax_vol.tick_params(colors='gray')
    ax.legend(loc='upper left', fontsize=10)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', facecolor='#0d1117', bbox_inches='tight', dpi=120)
    buf.seek(0)
    plt.close(fig)
    return buf


def send_signal(pair: str, price: float, prob: float, vol_m: float, change: float):
    if vol_m < 1:
        print(f"Пропуск {pair} — объём {vol_m}M < 1M")
        return

    if change <= -30:
        print(f"Пропуск {pair} — уже упала на {change:.2f}%")
        return

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
#  Обновление списка пар
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

        print(f"Загружено {len(PAIRS)} USDT-M Perpetual Futures пар")
        if len(PAIRS) > 0:
            print("Первые 5:", PAIRS[:5])

    except Exception as e:
        print(f"Ошибка обновления списка: {e}")


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

                print(f"{pair:20} prob = {prob:.4f}")

                if prob > PROBABILITY_THRESHOLD:
                    price, ch, vm = get_market_data(pair)
                    send_signal(pair, price, prob, vm, ch)

            except Exception as e:
                print(f"Ошибка {pair}: {type(e).__name__}")

            time.sleep(0.4)

        if time.time() - last_retrain > 6 * 3600:
            print("Переобучение модели...")
            model = load_or_train_model()
            last_retrain = time.time()

        time.sleep(INTERVAL_SECONDS)


if __name__ == '__main__':
    keep_alive()
    print("🚀 Бот запущен — сканируем USDT-M Futures MEXC")
    main_loop()
