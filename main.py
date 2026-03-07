import os
import time
import ccxt
import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt
from datetime import datetime
from telegram import Bot
from keep_alive import keep_alive
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import io

# === КЛЮЧИ ===
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
CHAT_ID = os.getenv('CHAT_ID')
MEXC_API_KEY = os.getenv('MEXC_API_KEY')
MEXC_API_SECRET = os.getenv('MEXC_API_SECRET')

bot = Bot(token=TELEGRAM_TOKEN)
exchange = ccxt.mexc({'apiKey': MEXC_API_KEY, 'secret': MEXC_API_SECRET, 'enableRateLimit': True})

PAIRS = ['ENA/USDT']
TIMEFRAME = '1h'
INTERVAL = 900
MODEL_FILE = 'catboost_model.cbm'

def get_data(symbol, limit=800):
    try:
        bars = exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=limit)
        df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except:
        return pd.DataFrame()

def calculate_features(df):
    if len(df) < 50: return df
    df['ema200'] = ta.ema(df['close'], length=200)
    df['rsi'] = ta.rsi(df['close'], length=14)
    df['macd'] = ta.macd(df['close'])['MACD_12_26_9']
    bb = ta.bbands(df['close'], length=20, std=2.0)
    df['bb_lower'] = bb.iloc[:, 0]
    df['price_change'] = df['close'].pct_change()
    df['volume_change'] = df['volume'].pct_change()  # новая фича
    return df.dropna()

# === ОБУЧЕНИЕ МОДЕЛИ ===
def train_model():
    print("Обучаем модель...")
    all_df = pd.DataFrame()
    training_pairs = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT', 'ADA/USDT', 'DOGE/USDT', 'SHIB/USDT', 'AVAX/USDT', 'TRX/USDT']
    for pair in training_pairs:
        df = get_data(pair)
        df = calculate_features(df)
        df['target'] = np.where((df['close'].shift(-1) < df['ema200'].shift(-1)) & (df['price_change'].shift(-1) < -0.01), 1, 0)
        all_df = pd.concat([all_df, df.dropna()])
    
    features = ['ema200', 'rsi', 'macd', 'bb_lower', 'price_change', 'volume_change']
    X = all_df[features]
    y = all_df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.05, verbose=0)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"Модель обучена! Accuracy: {acc:.2f}")
    model.save_model(MODEL_FILE)
    return model, acc

# Загружаем или обучаем
if not os.path.exists(MODEL_FILE):
    model, _ = train_model()
else:
    model = CatBoostClassifier()
    model.load_model(MODEL_FILE)
    print("Модель загружена")

def get_market_info(symbol):
    try:
        ticker = exchange.fetch_ticker(symbol)
        price = ticker['last']
        change = ticker['percentage']
        volume = ticker['quoteVolume']
        mcap = round(volume / 1_000_000, 1)
        return price, change, mcap
    except:
        return 0, 0, 0

def send_signal(pair, price, prob, mcap, change):
    if mcap < 50: return  # фильтр
    strength = "СИЛЬНЫЙ" if prob > 0.85 else "СРЕДНИЙ" if prob > 0.75 else "СЛАБЫЙ"
    fires = "🔥🔥🔥" if prob > 0.85 else "🔥🔥" if prob > 0.75 else "🔥"
    position_size = round(price * 200 * 50, 0)  # пример
    
    text = f"""🔴 {pair.split('/')[0]} {fires} {strength}
x200 / {position_size}$ / {mcap}M / {change:+.4f}

Trade: Mexc

Направление: Разворот ВНИЗ
Действие: SHORT

Текущая цена: {price}
Цель 1: {round(price * 0.95, 6)}
Цель 2: {round(price * 0.90, 6)}

Уверенность: {int(prob * 100)}%
Сила сигнала: {int(prob * 100 - 20)}/100
Общий Score: {int(prob * 100 + int(prob * 100 - 20))}"""

    # График с свечами
    df = get_data(pair)
    df = calculate_features(df)
    fig, ax = plt.subplots(figsize=(10,5), facecolor='#1e1e1e')
    ax.plot(df['timestamp'], df['close'], color='white')
    ax.plot(df['timestamp'], df['ema200'], color='orange')
    ax.bar(df['timestamp'], df['volume'], color='gray', alpha=0.5, bottom=0)  # объём
    ax.set_title(f'{pair} — Разворот ВНИЗ')
    ax.grid(True, alpha=0.3)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', facecolor='#1e1e1e')
    buf.seek(0)
    plt.close()

    bot.send_photo(chat_id=CHAT_ID, photo=buf, caption=text)

def trading_loop():
    last_retrain = time.time()
    while True:
        # Авто-пары
        try:
            markets = exchange.load_markets()
            top_pairs = sorted([p for p in markets if p.endswith('/USDT')], key=lambda p: markets[p].get('quoteVolume', 0), reverse=True)[:150]
            PAIRS[:] = top_pairs
        except: pass

        for pair in PAIRS:
            try:
                df = get_data(pair)
                df = calculate_features(df)
                if len(df) < 50: continue
                price = df['close'].iloc[-1]
                
                features = df.iloc[-1][['ema200', 'rsi', 'macd', 'bb_lower', 'price_change', 'volume_change']].values.reshape(1, -1)
                prob = model.predict_proba(features)[0][1]
                print(f"{pair} prob: {prob:.2f}")
                
                if prob > 0.65:
                    price, change, mcap = get_market_info(pair)
                    send_signal(pair, price, prob, mcap, change)
                    print(f"✅ Сигнал {pair} — {int(prob*100)}%")
            except: pass

        # Переобучение каждые 6 часов
        if time.time() - last_retrain > 21600:
            train_model()
            last_retrain = time.time()

        time.sleep(INTERVAL)

if __name__ == '__main__':
    keep_alive()
    print("🚀 СУПЕР ИИ-БОТ ЗАПУЩЕН!")
    trading_loop()
