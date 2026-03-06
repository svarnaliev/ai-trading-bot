import os
import time
import ccxt
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from datetime import datetime
from telegram import Bot
from keep_alive import keep_alive
import threading

# === КЛЮЧИ ===
TELEGRAM_TOKEN = os.getenv('7603082014:AAFYIowDNZBZGzahnkHsfPjkm-cEkc5Jmak')
CHAT_ID = os.getenv('754858892')
MEXC_API_KEY = os.getenv('MEXC_API_KEY')
MEXC_API_SECRET = os.getenv('MEXC_API_SECRET')

bot = Bot(token=TELEGRAM_TOKEN)
exchange = ccxt.mexc({
    'apiKey': MEXC_API_KEY,
    'secret': MEXC_API_SECRET,
    'enableRateLimit': True,
})

PAIRS = ['ENA/USDT', 'BTC/USDT', 'ETH/USDT']  # добавляй свои
TIMEFRAME = '1h'
INTERVAL = 900  # 15 минут

def get_data(symbol):
    bars = exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=500)
    df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

def calculate_features(df):
    df['ema200'] = ta.ema(df['close'], length=200)
    df['rsi'] = ta.rsi(df['close'])
    df['macd'] = ta.macd(df['close'])['MACD_12_26_9']
    return df

def send_signal(pair, price, confidence, score, target1, target2):
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

Уверенность: {confidence}%
Сила сигнала: {score}/100
Общий Score: {int(confidence + score)}"""

    with open('chart.png', 'rb') as photo:
        bot.send_photo(chat_id=CHAT_ID, photo=photo, caption=text)

def trading_loop():
    while True:
        for pair in PAIRS:
            try:
                df = get_data(pair)
                df = calculate_features(df)
                price = df['close'].iloc[-1]
                
                if price < df['ema200'].iloc[-1] and df['rsi'].iloc[-1] < 40:
                    confidence = 86
                    score = 62
                    target1 = round(price * 0.88, 6)
                    target2 = round(price * 0.84, 6)
                    send_signal(pair, price, confidence, score, target1, target2)
                    print(f"✅ Сигнал отправлен по {pair}")
            except Exception as e:
                print(f"Ошибка по {pair}: {e}")
        time.sleep(INTERVAL)

# === ЗАПУСК ===
if __name__ == '__main__':
    keep_alive()
    print("🚀 Бот запущен на Railway!")
    trading_loop()
