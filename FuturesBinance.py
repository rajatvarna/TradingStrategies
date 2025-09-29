import requests
import pandas as pd
import os
import time
from datetime import datetime, timedelta

BASE_URL = "https://fapi.binance.com"
KLINE_ENDPOINT = "/fapi/v1/klines"
INFO_ENDPOINT = "/fapi/v1/exchangeInfo"
SAVE_DIR = "binance_futures_data"
START_DATE = datetime(2020, 1, 1)  # Binance futures launch
INTERVAL = "1d"
LIMIT = 1500  # max per request

def get_all_symbols(quote_asset="USDT"):
    r = requests.get(BASE_URL + INFO_ENDPOINT)
    data = r.json()
    return [
        s["symbol"] for s in data["symbols"]
        if s["contractType"] == "PERPETUAL" and s["quoteAsset"] == quote_asset
    ]

def fetch_all_klines(symbol, interval="1d", start_time=START_DATE):
    ms_since_epoch = lambda dt: int(dt.timestamp() * 1000)
    start = ms_since_epoch(start_time)
    klines = []

    while True:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start,
            "limit": LIMIT
        }
        try:
            r = requests.get(BASE_URL + KLINE_ENDPOINT, params=params, timeout=10)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"❌ {symbol} failed: {e}")
            return None

        if not data:
            break

        klines.extend(data)
        last_open_time = data[-1][0]
        start = last_open_time + 1

        if len(data) < LIMIT:
            break

        time.sleep(0.2)  # avoid getting rate-limited

    return klines

def klines_to_df(klines, symbol):
    df = pd.DataFrame(klines, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "num_trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])
    df["Date"] = pd.to_datetime(df["open_time"], unit="ms").dt.strftime("%Y-%m-%d")
    df["Symbol"] = symbol
    df = df[["Date", "Symbol", "open", "high", "low", "close", "volume"]]
    df.columns = ["Date", "Symbol", "Open", "High", "Low", "Close", "Volume"]
    return df

def save_df_to_csv(df, symbol):
    os.makedirs(SAVE_DIR, exist_ok=True)
    path = os.path.join(SAVE_DIR, f"{symbol}.csv")
    df.to_csv(path, index=False)

def run():
    all_symbols = get_all_symbols()
    print(f"🔍 Found {len(all_symbols)} futures symbols...")

    for idx, symbol in enumerate(all_symbols):
        print(f"[{idx+1}/{len(all_symbols)}] ⏬ Downloading {symbol}...")
        klines = fetch_all_klines(symbol)
        if klines:
            df = klines_to_df(klines, symbol)
            save_df_to_csv(df, symbol)
            print(f"✅ Saved {symbol} with {len(df)} rows.")
        else:
            print(f"⚠️ No data for {symbol}")
        time.sleep(0.2)

if __name__ == "__main__":
    run()
