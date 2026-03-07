import os
import time
import ccxt
import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
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
exchange = ccxt.mexc({
    'apiKey': MEXC_API_KEY,
    'secret': MEXC_API_SECRET,
    'enableRateLimit': True,
})

PAIRS = ['ENA/USDT', 'BTC/USDT', 'ETH/USDT']
TIMEFRAME = '1h'
INTERVAL = 900
MODEL_FILE = 'catboost_model.cbm'

def get_data(symbol, limit=1000):
    try:
        bars = exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=limit)
        df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except:
        return pd.DataFrame()

def calculate_features(df):
    if len(df) < 50:
        return df
    df['ema200'] = ta.ema(df['close'], length=200)
    df['rsi'] = ta.rsi(df['close'], length=14)
    df['macd'] = ta.macd(df['close'])['MACD_12_26_9']
    bb = ta.bbands(df['close'], length=20, std=2.0)
    df['bb_lower'] = bb.iloc[:, 0]  # безопасно
    df['price_change'] = df['close'].pct_change()
    return df.dropna()

# === ОБУЧЕНИЕ МОДЕЛИ ===
def train_model():
    # (тот же код обучения, что был — не меняем)
    all_df = pd.DataFrame()
    training_pairs = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT']
    for pair in training_pairs:
        df = get_data(pair)
        df = calculate_features(df)
        df['target'] = np.where((df['close'].shift(-1) < df['ema200'].shift(-1)) & (df['price_change'].shift(-1) < -0.01), 1, 0)
        all_df = pd.concat([all_df, df.dropna()])
    
    features = ['ema200', 'rsi', 'macd', 'bb_lower', 'price_change']
    X = all_df[features]
    y = all_df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.05, verbose=0)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"Модель обучена! Accuracy: {acc:.2f}")
    model.save_model(MODEL_FILE)
    return model

if not os.path.exists(MODEL_FILE):
    model = train_model()
else:
    model = CatBoostClassifier()
    model.load_model(MODEL_FILE)
    print("Модель загружена")

# === ОТПРАВКА СИГНАЛА (БЕЗ KALEIDO — через matplotlib) ===
def send_signal(pair, price, prob, score, target1, target2):
    df = get_data(pair)
    df = calculate_features(df)
    if len(df) < 50:
        return
    
    # Рисуем график через matplotlib (работает на сервере без Chrome)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df['timestamp'], df['close'], label='Close', color='white')
    ax.plot(df['timestamp'], df['ema200'], label='EMA200', color='orange')
    ax.set_title(f'{pair} — Разворот ВНИЗ')
    ax.legend()
    ax.grid(True)
    plt.xticks(rotation=45)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', facecolor='#1e1e1e')
    buf.seek(0)
    plt.close()
    
    text = f"""🔴 {pair.split('/')[0]} 🔥🔥 СИЛЬНЫЙ
x200 / 11440$ / 145.8M / -0.0021
💰 Trade: Mexc

Направление: Разворот ВНИЗ
Действие: SHORT

Текущая цена: {price:.6f}
Зона:
▲ Цель 1: {target1:.6f}
▼ Цель 2: {target2:.6f}

Уверенность: {int(prob * 100)}%
Сила сигнала: {score}/100
Общий Score: {int(prob * 100 + score)}"""

    bot.send_photo(chat_id=CHAT_ID, photo=buf, caption=text)

def trading_loop():
    global model
    while True:
        try:
            markets = exchange.load_markets()
            top_pairs = [p for p in markets if p.endswith('/USDT') and markets[p].get('quoteVolume', 0) > 100000]
            top_pairs = sorted(top_pairs, key=lambda p: markets[p].get('quoteVolume', 0), reverse=True)[:150]
            PAIRS[:] = top_pairs
        except:
            pass
        
        for pair in PAIRS:
            try:
                df = get_data(pair)
                df = calculate_features(df)
                if len(df) < 50:
                    continue
                price = df['close'].iloc[-1]
                
                last_features = df.iloc[-1][['ema200', 'rsi', 'macd', 'bb_lower', 'price_change']].values.reshape(1, -1)
                prob = model.predict_proba(last_features)[0][1]
                
                if prob > 0.75:
                    score = int(prob * 100 - 38)
                    target1 = round(price * 0.95, 6)
                    target2 = round(price * 0.90, 6)
                    send_signal(pair, price, prob, score, target1, target2)
                    print(f"✅ Сигнал отправлен по {pair} ({int(prob*100)}%)")
            except Exception as e:
                print(f"Ошибка по {pair}: {e}")
        
        time.sleep(INTERVAL)

if __name__ == '__main__':
    keep_alive()
    print("🚀 Полный ИИ-бот запущен на Railway!")
    trading_loop()
