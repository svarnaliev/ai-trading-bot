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
    return df.dropna()

# === ПОЛНОЕ ОБУЧЕНИЕ МОДЕЛИ ===
def train_model():
    print("Обучаем модель...")
    all_df = pd.DataFrame()
    training_pairs = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT', 'ADA/USDT']
    for pair in training_pairs:
        df = get_data(pair)
        df = calculate_features(df)
        df['target'] = np.where((df['close'].shift(-1) < df['ema200'].shift(-1)) & (df['price_change'].shift(-1) < -0.01), 1, 0)
        all_df = pd.concat([all_df, df.dropna()])
    
    features = ['ema200', 'rsi', 'macd', 'bb_lower', 'price_change']
    X = all_df[features]
    y = all_df['target']
    
    model = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.05, verbose=0, random_state=42)
    model.fit(X, y)
    
    print(f"Модель обучена! Accuracy на обучающих данных: {model.score(X, y):.2f}")
    model.save_model(MODEL_FILE)
    return model

# Загружаем или обучаем
if not os.path.exists(MODEL_FILE):
    model = train_model()
else:
    model = CatBoostClassifier()
    model.load_model(MODEL_FILE)
    print("✅ Модель загружена из файла")

def get_market_info(symbol):
    try:
        ticker = exchange.fetch_ticker(symbol)
        price = ticker['last']
        change = round(ticker.get('percentage', 0), 4)
        volume = ticker.get('quoteVolume', 0)
        mcap = round(volume * 8 / 1_000_000, 1)  # приближённая капитализация
        return price, change, mcap
    except:
        return 0, 0, 0

def send_signal(pair, price, prob, mcap, change):
    strength = "СИЛЬНЫЙ" if prob > 0.85 else "СРЕДНИЙ" if prob > 0.75 else "СЛАБЫЙ"
    fires = "🔥🔥🔥" if prob > 0.88 else "🔥🔥" if prob > 0.82 else "🔥"
    position_size = round(11440 / price * 200, 0)
    
    text = f"""🔴 {pair.split('/')[0]} {fires} {strength}

x200 / {position_size}$ / {mcap}M / {change:+.4f}%

💰 Trade: Mexc

Направление: Разворот ВНИЗ
Действие: SHORT

Текущая цена: {price:.6f}
Зона:
▲ Цель 1: {round(price * 0.95, 6)}
▼ Цель 2: {round(price * 0.90, 6)}

Уверенность: {int(prob * 100)}%
Сила сигнала: {int(prob * 100 - 25)}/100
Общий Score: {int(prob * 100 + int(prob * 100 - 25))}"""

    # График
    df = get_data(pair)
    df = calculate_features(df)
    fig, ax = plt.subplots(figsize=(10,5), facecolor='#1e1e1e')
    ax.plot(df['timestamp'], df['close'], color='white', linewidth=1.5)
    ax.plot(df['timestamp'], df['ema200'], color='orange', linewidth=2)
    ax.set_title(f'{pair} — Разворот ВНИЗ')
    ax.grid(True, alpha=0.3)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', facecolor='#1e1e1e')
    buf.seek(0)
    plt.close()

    bot.send_photo(chat_id=CHAT_ID, photo=buf, caption=text)

def trading_loop():
    while True:
        try:
            markets = exchange.load_markets()
            top_pairs = sorted([p for p in markets if p.endswith('/USDT')], 
                             key=lambda p: markets[p].get('quoteVolume', 0), reverse=True)[:150]
            PAIRS[:] = top_pairs
        except: pass

        for pair in PAIRS[:30]:  # проверяем только топ-30 для скорости
            try:
                df = get_data(pair)
                df = calculate_features(df)
                if len(df) < 50: continue
                price = df['close'].iloc[-1]
                
                features = df.iloc[-1][['ema200', 'rsi', 'macd', 'bb_lower', 'price_change']].values.reshape(1, -1)
                prob = model.predict_proba(features)[0][1]

                if prob > 0.78:
                    price, change, mcap = get_market_info(pair)
                    send_signal(pair, price, prob, mcap, change)
                    print(f"✅ СИГНАЛ {pair} — {int(prob*100)}%")
            except:
                pass

        time.sleep(INTERVAL)

if __name__ == '__main__':
    keep_alive()
    print("🚀 СУПЕР ИИ-БОТ ЗАПУЩЕН И РАБОТАЕТ!")
    trading_loop()
