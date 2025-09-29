import ccxt
import pandas as pd
import time
import os

exchange = ccxt.binance()  # Change to ccxt.bybit() or ccxt.okx() if needed
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "all_binance")

def get_usdt_pairs(exchange):
    """Fetch all symbols ending with USDT"""
    markets = exchange.load_markets()
    return [symbol for symbol in markets if symbol.endswith("/USDT")]

def fetch_ohlcv(symbol, timeframe='1d'):
    """Fetch historical OHLCV data for a symbol"""
    since = 0  # Start from the beginning
    all_data = []
    
    while True:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit=1000)
        if not ohlcv:
            break  # No more data

        all_data.extend(ohlcv)
        since = ohlcv[-1][0] + 1  # Continue from the last timestamp

        time.sleep(exchange.rateLimit / 1000)  # Respect API limits
    
    return all_data

def save_to_csv(data, filename, symbol):
    """Save OHLCV data to a CSV file with custom modifications"""
    df = pd.DataFrame(data, columns=["time", "open", "high", "low", "close", "volume"])
    df["time"] = pd.to_datetime(df["time"], unit="ms").dt.strftime("%Y-%m-%d")  # Keep only YYYY-MM-DD
    df.insert(1, "symbol", symbol)  # Insert symbol column after time
    df.to_csv(filename, index=False)
    print(f"Saved: {filename}")

# Main process
os.makedirs(OUTPUT_DIR, exist_ok=True)
usdt_pairs = get_usdt_pairs(exchange)

for symbol in usdt_pairs:
    print(f"Fetching data for {symbol}...")
    data = fetch_ohlcv(symbol)
    filename = os.path.join(OUTPUT_DIR, f"{symbol.replace('/', '')}_daily.csv")
    save_to_csv(data, filename, symbol)

print("✅ All data downloaded and formatted!")
