import os
import time
import ccxt
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from datetime import datetime
from telegram import Bot
from keep_alive import keep_alive
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pickle

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

# === НАСТРОЙКИ ===
PAIRS = ['ENA/USDT', 'BTC/USDT', 'ETH/USDT']  # стартовые, добавим авто ниже
TIMEFRAME = '1h'
INTERVAL = 900  # 15 мин
MODEL_FILE = 'catboost_model.cbm'  # файл модели

# Функция для загрузки данных и фич
def get_data(symbol, limit=1000):
    bars = exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=limit)
    df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

def calculate_features(df):
    # EMA200
    df['ema200'] = ta.ema(df['close'], length=200)
    
    # RSI
    df['rsi'] = ta.rsi(df['close'], length=14)
    
    # MACD
    df['macd'] = ta.macd(df['close'])['MACD_12_26_9']
    
    # Bollinger Bands — БЕЗОПАСНЫЙ способ (работает в любой версии pandas_ta)
    bb = ta.bbands(df['close'], length=20, std=2.0)
    df['bb_lower'] = bb.iloc[:, 0]   # первая колонка всегда = нижняя полоса
    
    # Дополнительная фича
    df['price_change'] = df['close'].pct_change()
    
    return df.dropna()

# === ОБУЧЕНИЕ МОДЕЛИ (раз в день) ===
def train_model():
    # Собираем данные по 10+ парам для обучения
    all_df = pd.DataFrame()
    training_pairs = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT', 'ADA/USDT', 'DOGE/USDT', 'SHIB/USDT', 'AVAX/USDT', 'TRX/USDT']
    for pair in training_pairs:
        df = get_data(pair, limit=1000)
        df = calculate_features(df)
        # Целевая переменная: 1 если следующий разворот вниз (close < ema200 и цена упала на 1%)
        df['target'] = np.where((df['close'].shift(-1) < df['ema200'].shift(-1)) & (df['price_change'].shift(-1) < -0.01), 1, 0)
        all_df = pd.concat([all_df, df.dropna()])
    
    # Фичи и таргет
    features = ['ema200', 'rsi', 'macd', 'bb_lower', 'price_change']
    X = all_df[features]
    y = all_df['target']
    
    # Разделяем и обучаем
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.05, verbose=0)
    model.fit(X_train, y_train)
    
    # Точность
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    print(f"Модель обучена! Accuracy: {acc:.2f}")
    
    # Сохраняем
    model.save_model(MODEL_FILE)
    return model, acc

# Загружаем или обучаем модель
if not os.path.exists(MODEL_FILE):
    model, acc = train_model()
else:
    model = CatBoostClassifier()
    model.load_model(MODEL_FILE)
    print("Модель загружена")

def send_signal(pair, price, prob, score, target1, target2):
    df = get_data(pair)
    df = calculate_features(df)
    
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close']))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['ema200'], line=dict(color='orange'), name='EMA200'))
    fig.update_layout(title=f'{pair} — Разворот ВНИЗ', template='plotly_dark')
    fig.write_image('chart.png')
    
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

    with open('chart.png', 'rb') as photo:
        bot.send_photo(chat_id=CHAT_ID, photo=photo, caption=text)

def trading_loop():
    global model
    while True:
        # Авто-обновление пар (топ-150 по объёму)
        markets = exchange.load_markets()
        top_pairs = sorted([p for p in markets if p.endswith('/USDT')], key=lambda p: markets[p].get('quoteVolume', 0), reverse=True)[:150]
        PAIRS[:] = top_pairs  # обновляем список
        
        for pair in PAIRS:
            try:
                df = get_data(pair)
                df = calculate_features(df)
                price = df['close'].iloc[-1]
                
                # Предсказание от модели
                last_features = df.iloc[-1][['ema200', 'rsi', 'macd', 'bb_lower', 'price_change']].values.reshape(1, -1)
                prob = model.predict_proba(last_features)[0][1]  # вероятность разворота вниз
                
                if prob > 0.75:  # порог для сигнала
                    score = int(prob * 100 - 38)  # динамическая сила
                    target1 = round(price * (1 - 0.05), 6)  # -5%
                    target2 = round(price * (1 - 0.10), 6)  # -10%
                    send_signal(pair, price, prob, score, target1, target2)
                    print(f"✅ Сигнал отправлен по {pair} с уверенностью {int(prob*100)}%")
            except Exception as e:
                print(f"Ошибка по {pair}: {e}")
        
        time.sleep(INTERVAL)
        # Переобучение модели раз в 24 часа
        if datetime.now().hour == 0:
            train_model()

# === ЗАПУСК ===
if __name__ == '__main__':
    keep_alive()
    print("🚀 Полный ИИ-бот запущен на Railway!")
    trading_loop()
