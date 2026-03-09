import os
import time
import traceback
from datetime import datetime, timedelta
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
PROBABILITY_THRESHOLD = 0.55
SIGNAL_LIFETIME = 10800  # 3 часа

FEATURES = ['ema200', 'rsi', 'macd', 'bb_lower', 'price_change', 'volume_change', 'bb_width']


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
    'rateLimit': 1200,
    'options': {'defaultType': 'swap'},
})

PAIRS = []
ACTIVE_SIGNALS = []


# ────────────────────────────────────────────────
#  Данные и фичи
# ────────────────────────────────────────────────

def fetch_ohlcv(symbol: str, limit: int = 2000) -> pd.DataFrame:
    try:
        time.sleep(1.2)
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
    df['bb_upper']     = bb.iloc[:, 2]
    df['bb_width']     = (df['bb_upper'] - df['bb_lower']) / df['close']
    df['price_change'] = df['close'].pct_change()
    df['volume_change']= df['volume'].pct_change()

    return df.dropna()


# ────────────────────────────────────────────────
#  Модель
# ────────────────────────────────────────────────

def load_or_train_model() -> CatBoostClassifier:
    if os.path.exists(MODEL_FILE):
        print("Удаляем старую модель...")
        os.remove(MODEL_FILE)

    print("Обучение модели...")
    training_pairs = [
        'BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT', 'XRP/USDT:USDT',
        'BNB/USDT:USDT', 'ADA/USDT:USDT', 'DOGE/USDT:USDT', 'AVAX/USDT:USDT',
        'TRX/USDT:USDT', 'POWER/USDT:USDT', 'NEAR/USDT:USDT', 'SUI/USDT:USDT',
        'PEPE/USDT:USDT', 'WIF/USDT:USDT', 'SIREN/USDT:USDT', 'POPCAT/USDT:USDT',
        'BRETT/USDT:USDT', 'PNUT/USDT:USDT', 'GOAT/USDT:USDT', 'FARTCOIN/USDT:USDT',
        'RIVER/USDT:USDT', 'TURBO/USDT:USDT', 'MYX/USDT:USDT', 'AERO/USDT:USDT',
        'JUP/USDT:USDT', 'MOODENG/USDT:USDT', 'KITE/USDT:USDT', 'UAI/USDT:USDT',
        'PENGU/USDT:USDT', 'FLOKI/USDT:USDT', 'SHIB/USDT:USDT', 'DOGS/USDT:USDT',
        'MEW/USDT:USDT', 'APT/USDT:USDT', 'ARB/USDT:USDT', 'OP/USDT:USDT',
        'PEPE/USDT:USDT', '1000BONK/USDT:USDT', '1000SHIB/USDT:USDT', '1000FLOKI/USDT:USDT'
    ]
    all_data = []
    loaded_count = 0
    for symbol in training_pairs:
        try:
            time.sleep(2)
            df = fetch_ohlcv(symbol)
            if df.empty: continue
            df = add_features(df)
            if df.empty: continue
            df['target'] = (df['price_change'].shift(-1) < -0.005).astype(int)
            all_data.append(df)
            loaded_count += 1
        except Exception as e:
            print(f"Пропуск {symbol}: {e}")
            continue

    print(f"Успешно загружено {loaded_count} пар для обучения")
    if not all_data:
        raise ValueError("Нет данных для обучения модели!")

    df_all = pd.concat(all_data).dropna()
    X = df_all[FEATURES]
    y = df_all['target']

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    model = CatBoostClassifier(iterations=1000, depth=8, learning_rate=0.05, verbose=0)
    model.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, model.predict(X_te))
    print(f"Модель обучена | Accuracy: {acc:.4f}")
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
#  График
# ────────────────────────────────────────────────

def create_chart(pair: str, entry_price: float) -> io.BytesIO | None:
    df = fetch_ohlcv(pair)
    if df.empty: return None
    df = add_features(df)
    if df.empty: return None

    tp1 = round(entry_price * 0.95, 6)
    tp2 = round(entry_price * 0.90, 6)
    avg_level = round(entry_price * 1.06, 6)

    fig, ax = plt.subplots(figsize=(10, 6), facecolor='#0d1117')

    ax.plot(df['timestamp'], df['close'], color='#00ff9d', linewidth=2, label='Цена')
    ax.plot(df['timestamp'], df['ema200'], color='#ff4444', linewidth=1.8, label='EMA200')

    ax.axhline(entry_price, color='white', linestyle='--', linewidth=1.3, label='Вход')
    ax.axhline(tp1, color='#00ff00', linestyle='-', linewidth=1.1, label='Цель 1')
    ax.axhline(tp2, color='#00cc00', linestyle='-', linewidth=1.1, label='Цель 2')

    if avg_level > entry_price:
        ax.axhline(avg_level, color='orange', linestyle='--', linewidth=1.3, label='Усреднение +6%')

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %H:%M'))
    plt.xticks(rotation=45)
    ax.grid(True, alpha=0.15, color='gray')
    ax.set_title(f'{pair} — Разворот ВНИЗ', color='white', fontsize=14)
    ax.legend(loc='upper left', fontsize=9)
    ax.tick_params(colors='white')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', facecolor='#0d1117', bbox_inches='tight', dpi=130)
    buf.seek(0)
    plt.close(fig)
    return buf


# ────────────────────────────────────────────────
#  Сигнал с новой рабочей ссылкой на Liquidation Levels [LuxAlgo]
# ────────────────────────────────────────────────

def build_signal_text(pair: str, price: float, prob: float, vol_m: float, change: float) -> str:
    coin = pair.split('/')[0].replace(':USDT', '')   # SQD
    tv_link = f"https://www.tradingview.com/chart/hXHfYTh0/?symbol=MEXC%3A{coin}USDT.P&interval=60&script=VBLeqKvy-Liquidation-Levels-LuxAlgo"

    strength = "СИЛЬНЫЙ" if prob > 0.85 else "СРЕДНИЙ" if prob > 0.75 else "СЛАБЫЙ"
    fires = "🔥🔥🔥" if prob > 0.85 else "🔥🔥" if prob > 0.75 else "🔥"
    pos_size = round(price * 200 * 50, 0)

    text = f"""🔴 {coin} {fires} {strength}
x200 / {pos_size}$ / {vol_m}M / {change:+.4f}

Trade: Mexc Futures

Направление: Разворот ВНИЗ
Действие: SHORT

Текущая цена: {price}
Цель 1: {round(price * 0.95, 6)}
Цель 2: {round(price * 0.90, 6)}"""

    avg_level = round(price * 1.06, 6)
    if avg_level > price:
        text += f"\nУсреднение: на +6% ≈ {avg_level}"

    text += f"""

Уверенность: {int(prob * 100)}%
Сила сигнала: {int(prob * 100 - 20)}/100
Общий Score: {int(prob * 100 + int(prob * 100 - 20))}

Проверь зоны ликвидаций в TradingView (Liquidation Levels [LuxAlgo]):
{tv_link}"""

    return text


def send_signal(pair: str, price: float, prob: float, vol_m: float, change: float):
    df = fetch_ohlcv(pair)
    if df.empty: return
    df = add_features(df)
    if df.empty: return

    if df['rsi'].iloc[-1] < 72:
        print(f"Пропуск {pair} — RSI {df['rsi'].iloc[-1]:.1f} < 72")
        return
    if df['bb_width'].iloc[-1] < 0.06:
        print(f"Пропуск {pair} — BB width {df['bb_width'].iloc[-1]:.4f} < 0.06")
        return

    text = build_signal_text(pair, price, prob, vol_m, change)
    buf = create_chart(pair, price)

    try:
        bot.send_photo(chat_id=CHAT_ID, photo=buf, caption=text)
        print(f"Сигнал отправлен → {pair} ({int(prob*100)}%)")

        avg_level = round(price * 1.06, 6) if round(price * 1.06, 6) > price else None
        ACTIVE_SIGNALS.append({
            'pair': pair,
            'entry_price': price,
            'avg_price': avg_level,
            'timestamp': time.time()
        })
    except Exception as e:
        print(f"Ошибка отправки {pair}: {e}")


# ────────────────────────────────────────────────
#  Проверка истёкших сигналов
# ────────────────────────────────────────────────

def check_expired_signals():
    global ACTIVE_SIGNALS
    current_time = time.time()
    to_remove = []

    for signal in ACTIVE_SIGNALS:
        if current_time - signal['timestamp'] > SIGNAL_LIFETIME:
            pair = signal['pair']
            try:
                price, _, _ = get_market_data(pair)
                entry = signal['entry_price']
                avg = signal['avg_price']

                if price < entry:
                    msg = f"✅ Сигнал {pair} отработал! Цена ниже входа ({price:.6f} < {entry:.6f}) — профит."
                else:
                    close_level = avg if avg else entry
                    msg = f"⚠️ Сигнал {pair} не сработал за 3 часа. Закрывай на {close_level:.6f} (не в минус). Текущая цена: {price:.6f}"

                bot.send_message(chat_id=CHAT_ID, text=msg)
                print(msg)
            except Exception as e:
                print(f"Ошибка проверки {pair}: {e}")

            to_remove.append(signal)

    ACTIVE_SIGNALS = [s for s in ACTIVE_SIGNALS if s not in to_remove]


# ────────────────────────────────────────────────
#  Обновление списка пар и цикл
# ────────────────────────────────────────────────

def update_pairs_list():
    try:
        markets = futures_exchange.load_markets(reload=True)
        futures_pairs = [s for s, m in markets.items() if m.get('swap') and m.get('linear') and 'USDT' in s and m.get('active')]
        sorted_pairs = sorted(futures_pairs, key=lambda s: float(markets[s].get('info', {}).get('quoteVolume', 0) or 0), reverse=True)
        PAIRS[:] = sorted_pairs[:1000]
        print(f"Загружено {len(PAIRS)} USDT-M Perpetual Futures пар")
    except Exception as e:
        print(f"Ошибка обновления списка: {e}")


def main_loop():
    model = load_or_train_model()
    last_retrain = time.time()

    print("Тест Telegram...")
    bot.send_message(chat_id=CHAT_ID, text=f"🤖 Бот запущен (Futures) | {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    while True:
        update_pairs_list()
        check_expired_signals()

        for pair in PAIRS:
            try:
                df = fetch_ohlcv(pair)
                df = add_features(df)
                if len(df) < MIN_DATA_LENGTH: continue

                row = df.iloc[-1]
                feats = row[FEATURES].values.reshape(1, -1)
                prob = model.predict_proba(feats)[0][1]

                print(f"{pair:20} prob = {prob:.4f}")

                if prob > PROBABILITY_THRESHOLD:
                    price, ch, vm = get_market_data(pair)
                    send_signal(pair, price, prob, vm, ch)

            except Exception as e:
                print(f"Ошибка {pair}: {type(e).__name__}")

            time.sleep(1.2)

        if time.time() - last_retrain > 12 * 3600:
            model = load_or_train_model()
            last_retrain = time.time()

        time.sleep(INTERVAL_SECONDS)


if __name__ == '__main__':
    keep_alive()
    print("🚀 Бот запущен — сканируем USDT-M Futures MEXC")
    main_loop()
