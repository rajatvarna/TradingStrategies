# ================================
# Global Parameters & Imports
# ================================
import os
import seaborn as sns
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import pytz
import math
import time
from datetime import datetime, timedelta
from IPython.display import display, Markdown

# ===== Configuration =====
# Please set your Polygon.io API key below:
API_KEY = "RZAXDMQqIv9B1_IRwC7Ejog1GgP7WFX0"

# Set to True if you have a paid Polygon subscription; otherwise, set to False.
PAID_POLYGON_SUBSCRIPTION = False

# Trading parameters (feel free to modify these)
TICKER = "TQQQ"
START_DATE = "2023-09-27"
END_DATE = "2025-08-25"
OUTPUT_FILE = f"{TICKER}_intraday_data.csv"

utc_tz = pytz.timezone('UTC')
nyc_tz = pytz.timezone('America/New_York')

# ================================
# Data Download Functions
# ================================
def get_polygon_data(url=None, ticker=TICKER, multiplier=1, timespan="minute",
                       from_date=START_DATE, to_date=END_DATE, adjusted=False):
    """Retrieve intraday aggregate data from Polygon.io."""
    if url is None:
        url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
        params = {
            "adjusted": "true" if adjusted else "false",
            "sort": "asc",
            "limit": 50000,
            "apiKey": API_KEY
        }
        response = requests.get(url, params=params)
    else:
        if "apiKey" not in url:
            url = f"{url}&apiKey={API_KEY}" if "?" in url else f"{url}?apiKey={API_KEY}"
        response = requests.get(url)

    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None, None

    data = response.json()
    next_url = data.get("next_url")
    if next_url and "apiKey" not in next_url:
        next_url = f"{next_url}&apiKey={API_KEY}" if "?" in next_url else f"{next_url}?apiKey={API_KEY}"

    return data.get("results", []), next_url

def get_daily_adjusted_data():
    """Fetch daily adjusted data for calculating the opening price and ATR.

    Uses ATR_START_DATE = 30 days before START_DATE.
    """
    ATR_START_DATE = (datetime.strptime(START_DATE, '%Y-%m-%d') - timedelta(days=30)).strftime('%Y-%m-%d')

    print("Fetching daily adjusted data for ATR calculation...")
    url = f"https://api.polygon.io/v2/aggs/ticker/{TICKER}/range/1/day/{ATR_START_DATE}/{END_DATE}"
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 50000,
        "apiKey": API_KEY
    }
    response = requests.get(url, params=params)

    if response.status_code != 200:
        print(f"Error getting daily data: {response.status_code}")
        print(response.text)
        return pd.DataFrame()

    data = response.json().get("results", [])
    if not data:
        print("No daily data found.")
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df['datetime_utc'] = pd.to_datetime(df['t'], unit='ms')
    df['datetime_et'] = df['datetime_utc'].dt.tz_localize(utc_tz).dt.tz_convert(nyc_tz)
    df['day'] = df['datetime_et'].dt.date.astype(str)
    df = df.rename(columns={'o': 'dOpen', 'h': 'dHigh', 'l': 'dLow', 'c': 'dClose', 'v': 'dVolume'})
    df = calculate_atr(df, period=14)
    df = df[df['datetime_et'] >= pd.Timestamp(START_DATE, tz=nyc_tz)].copy()
    return df[['day', 'dOpen', 'ATR']]

# ================================
# ATR Calculation & Data Processing
# ================================
def calculate_atr(df, period=14):
    """Calculate the Average True Range (ATR) over a given period."""
    if 'dHigh' not in df.columns or 'dLow' not in df.columns or 'dClose' not in df.columns:
        print("Missing required columns for ATR calculation")
        return df

    df = df.copy()
    df['prev_close'] = df['dClose'].shift(1)
    df['HC'] = abs(df['dHigh'] - df['prev_close'])
    df['LC'] = abs(df['prev_close'] - df['dLow'])
    df['HL'] = abs(df['dHigh'] - df['dLow'])
    df['TR'] = df[['HC', 'LC', 'HL']].max(axis=1)
    df['ATR'] = df['TR'].rolling(window=period, min_periods=1).mean()
    df['ATR'] = df['ATR'].shift(1)
    df.drop(['prev_close', 'HC', 'LC', 'HL', 'TR'], axis=1, inplace=True)
    return df

def process_data(results):
    """Process raw intraday data from Polygon.io into a clean DataFrame."""
    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df['datetime_utc'] = pd.to_datetime(df['t'], unit='ms')
    df['datetime_et'] = df['datetime_utc'].dt.tz_localize(utc_tz).dt.tz_convert(nyc_tz)
    df['caldt'] = df['datetime_et'].dt.tz_localize(None)

    # Filter to regular market hours (9:30-15:59 ET)
    df = df.set_index('datetime_et')
    market_data = df.between_time('09:30', '15:59').reset_index()

    market_data['date'] = market_data['datetime_et'].dt.date
    market_data = market_data.rename(columns={
        'v': 'volume',
        'vw': 'vwap',
        'o': 'open',
        'c': 'close',
        'h': 'high',
        'l': 'low',
        't': 'timestamp_ms',
        'n': 'transactions'
    })
    market_data['day'] = market_data['date'].astype(str)

    return market_data

def download_and_merge_data():
    """
    Download intraday and daily adjusted data, merge them, and save to CSV.

    If PAID_POLYGON_SUBSCRIPTION is False, simply warn or stop if START_DATE is older than 2 years.
    """
    if not PAID_POLYGON_SUBSCRIPTION:
        two_years_ago = datetime.now() - timedelta(days=730)
        start_dt = datetime.strptime(START_DATE, '%Y-%m-%d')
        if start_dt < two_years_ago:
            print("ERROR: For free Polygon subscriptions, START_DATE must be within the past 2 years.")
            return None

    daily_adjusted_data = get_daily_adjusted_data()
    if daily_adjusted_data.empty:
        print("Unable to get adjusted daily data. Exiting.")
        return None

    # Download all data first, then process it all at once
    all_raw_data = []
    next_url = None
    batch_count = 0
    print(f"Downloading intraday data for {TICKER} from {START_DATE}...")

    while True:
        batch_count += 1
        print(f"Batch {batch_count}...")
        results, next_url = get_polygon_data(url=next_url, adjusted=False)
        if not results:
            print("No more data.")
            break

        # Just add raw results to our collection, don't process yet
        all_raw_data.extend(results)
        print(f"Batch {batch_count}: Retrieved {len(results)} records")

        if not next_url:
            print("Download complete.")
            break

        if not PAID_POLYGON_SUBSCRIPTION:
            # Enforce a rate limit: 5 requests per minute (sleep for 12 seconds)
            time.sleep(12)

    if all_raw_data:
        # Now process all data at once
        print(f"Processing {len(all_raw_data)} total records...")
        final_df = process_data(all_raw_data)

        if not final_df.empty:
            final_df = pd.merge(final_df, daily_adjusted_data, on='day', how='left')
            final_df['caldt'] += pd.Timedelta(minutes=1)
            cols = ['caldt', 'open', 'high', 'low', 'close', 'volume', 'vwap', 'transactions', 'day', 'dOpen', 'ATR']
            available_cols = [col for col in cols if col in final_df.columns]
            final_df[available_cols].to_csv(OUTPUT_FILE, index=False)
            print(f"Data saved to {OUTPUT_FILE}")
            print(f"Total records: {len(final_df)}")
            return final_df
        else:
            print("No data after processing.")
            return None
    else:
        print("No data collected.")
        return None


# ================================
# Performance Analysis & Backtesting Functions
# ================================
def price2return(price, n=1):
    """Convert a series of prices into returns."""
    price = np.array(price)
    T = len(price)
    y = np.full_like(price, np.nan, dtype=float)
    if T > n:
        y[n:] = price[n:] / price[:T-n] - 1
    return y

def summary_statistics(dailyReturns):
    """Calculate performance metrics and return a summary table."""
    riskFreeRate = 0
    tradingDays = 252
    dailyReturns = np.array(dailyReturns)
    dailyReturns = dailyReturns[~np.isnan(dailyReturns)]
    totalReturn = np.prod(1 + dailyReturns) - 1
    numYears = len(dailyReturns) / tradingDays
    CAGR = (1 + totalReturn)**(1/numYears) - 1
    volatility = np.std(dailyReturns, ddof=0) * np.sqrt(tradingDays)
    sharpeRatio = (np.mean(dailyReturns) - riskFreeRate/tradingDays) / np.std(dailyReturns, ddof=0) * np.sqrt(tradingDays)
    nav = np.cumprod(1 + dailyReturns)
    peak = np.maximum.accumulate(nav)
    drawdown = (nav - peak) / peak
    MDD = np.min(drawdown)
    metrics = ["Total Return (%)", "CAGR (%)", "Volatility (%)", "Sharpe Ratio", "Max Drawdown (%)"]
    values = [totalReturn*100, CAGR*100, volatility*100, sharpeRatio, MDD*100]
    formatted_values = [f"{v:.4f}" if i < 3 or i == 4 else f"{v:.6f}" for i,v in enumerate(values)]
    performance_table = pd.DataFrame({'Metric': metrics, 'Value': formatted_values})
    return performance_table

def monthly_performance_table(returns, dates):
    """Create a table of monthly returns."""
    returns_series = pd.Series(returns, index=pd.DatetimeIndex(dates))
    returns_series = returns_series[~np.isnan(returns_series)]
    df = pd.DataFrame({'return': returns_series,
                       'year': returns_series.index.year,
                       'month': returns_series.index.month})
    monthly_returns = df.groupby(['year', 'month'])['return'].apply(lambda x: np.prod(1 + x) - 1).reset_index()
    pivot_table = monthly_returns.pivot(index='year', columns='month', values='return')
    pivot_table['Year Total'] = pivot_table.apply(lambda row: np.prod(1 + row.dropna()) - 1
                                                   if not row.dropna().empty else np.nan, axis=1)
    formatted_table = pivot_table.apply(lambda col: col.map(lambda x: f"{x*100:.2f}%" if not pd.isna(x) else ""))
    month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                   7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
    formatted_table = formatted_table.rename(columns=month_names)
    return formatted_table

def backtest(days, p, orb_m, target_R, risk, max_Lev, AUM_0, commission):
    """Perform an optimized backtest for the ORB strategy."""
    start_time = time.time()
    str_df = pd.DataFrame()
    str_df['Date'] = days
    str_df['AUM'] = np.nan
    str_df.loc[0, 'AUM'] = AUM_0
    str_df['pnl_R'] = np.nan
    or_candles = orb_m
    day_groups = dict(tuple(p.groupby(p['day'].dt.date)))

    for t in range(1, len(days)):
        current_day = days[t].date()
        if current_day not in day_groups:
            str_df.loc[t, 'pnl_R'] = 0
            str_df.loc[t, 'AUM'] = str_df.loc[t-1, 'AUM']
            continue

        day_data = day_groups[current_day]
        if len(day_data) <= or_candles:
            str_df.loc[t, 'pnl_R'] = 0
            str_df.loc[t, 'AUM'] = str_df.loc[t-1, 'AUM']
            continue

        OHLC = day_data[['open', 'high', 'low', 'close']].values
        split_adj = OHLC[0, 0] / day_data['dOpen'].iloc[0]
        atr_raw = day_data['ATR'].iloc[0] * split_adj
        side = np.sign(OHLC[or_candles-1, 3] - OHLC[0, 0])
        entry = OHLC[or_candles, 0] if len(OHLC) > or_candles else np.nan


        if side == 1:
            stop = abs(np.min(OHLC[:or_candles, 2]) / entry - 1)
        elif side == -1:
            stop = abs(np.max(OHLC[:or_candles, 1]) / entry - 1)
        else:
            stop = np.nan

        if side == 0 or math.isnan(stop) or math.isnan(entry):
            str_df.loc[t, 'pnl_R'] = 0
            str_df.loc[t, 'AUM'] = str_df.loc[t-1, 'AUM']
            continue

        if entry == 0 or stop == 0:
            shares = 0
        else:
            shares = math.floor(min(str_df.loc[t-1, 'AUM'] * risk / (entry * stop),
                                    max_Lev * str_df.loc[t-1, 'AUM'] / entry))

        if shares == 0:
            str_df.loc[t, 'pnl_R'] = 0
            str_df.loc[t, 'AUM'] = str_df.loc[t-1, 'AUM']
            continue

        OHLC_post_entry = OHLC[or_candles:, :]

        if side == 1:  # Long trade
            stop_price = entry * (1 - stop)
            target_price = entry * (1 + target_R * stop) if np.isfinite(target_R) else float('inf')
            stop_hits = OHLC_post_entry[:, 2] <= stop_price
            target_hits = OHLC_post_entry[:, 1] > target_price

            if np.any(stop_hits) and np.any(target_hits):
                idx_stop = np.argmax(stop_hits)
                idx_target = np.argmax(target_hits)
                if idx_target < idx_stop:
                    PnL_T = max(target_price, OHLC_post_entry[idx_target, 0]) - entry
                else:
                    PnL_T = min(stop_price, OHLC_post_entry[idx_stop, 0]) - entry
            elif np.any(stop_hits):
                idx_stop = np.argmax(stop_hits)
                PnL_T = min(stop_price, OHLC_post_entry[idx_stop, 0]) - entry
            elif np.any(target_hits):
                idx_target = np.argmax(target_hits)
                PnL_T = max(target_price, OHLC_post_entry[idx_target, 0]) - entry
            else:
                PnL_T = OHLC_post_entry[-1, 3] - entry
        elif side == -1:  # Short trade
            stop_price = entry * (1 + stop)
            target_price = entry * (1 - target_R * stop) if np.isfinite(target_R) else 0
            stop_hits = OHLC_post_entry[:, 1] >= stop_price
            target_hits = OHLC_post_entry[:, 2] < target_price

            if np.any(stop_hits) and np.any(target_hits):
                idx_stop = np.argmax(stop_hits)
                idx_target = np.argmax(target_hits)
                if idx_target < idx_stop:
                    PnL_T = entry - min(target_price, OHLC_post_entry[idx_target, 0])
                else:
                    PnL_T = entry - max(stop_price, OHLC_post_entry[idx_stop, 0])
            elif np.any(stop_hits):
                idx_stop = np.argmax(stop_hits)
                PnL_T = entry - max(stop_price, OHLC_post_entry[idx_stop, 0])
            elif np.any(target_hits):
                idx_target = np.argmax(target_hits)
                PnL_T = entry - min(target_price, OHLC_post_entry[idx_target, 0])
            else:
                PnL_T = entry - OHLC_post_entry[-1, 3]

        str_df.loc[t, 'AUM'] = str_df.loc[t-1, 'AUM'] + shares * PnL_T - shares * commission * 2
        str_df.loc[t, 'pnl_R'] = (str_df.loc[t, 'AUM'] - str_df.loc[t-1, 'AUM']) / (risk * str_df.loc[t-1, 'AUM'])

    end_time = time.time()
    print(f"******** Optimized Backtest Completed in {round(end_time - start_time, 2)} seconds! ********")
    print(f"Starting AUM: ${AUM_0:,.2f}")
    print(f"Final AUM: ${str_df['AUM'].iloc[-1]:,.2f}")
    print(f"Total Return: {(str_df['AUM'].iloc[-1]/AUM_0 - 1)*100:.4f}%")
    return str_df

def plot_equity_curve(str_df, AUM_0, orb_m, target_R, ticker):
    """Plot the equity curve with weekly resampling and highlight out-of-sample period."""
    fig, ax = plt.subplots(figsize=(12, 7))
    df_plot = str_df.copy()
    if 'Date' in df_plot.columns:
        df_plot = df_plot.set_index('Date')
    try:
        weekly_data = df_plot['AUM'].resample('W').last().dropna()
    except Exception as e:
        print("Resampling failed, using original data.", e)
        weekly_data = df_plot['AUM'].dropna()

    p1, = ax.plot(weekly_data.index, weekly_data.values, 'r-', linewidth=2, label='Equity')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=90)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax.grid(True, linestyle=':')

    min_val = weekly_data.min() if not weekly_data.empty else AUM_0
    max_val = weekly_data.max() if not weekly_data.empty else AUM_0
    ax.set_ylim([0.9 * min_val, 1.25 * max_val])

    target_str = f"Target {target_R}R" if np.isfinite(target_R) else "No Target"
    ax.set_title(f"{orb_m}m-ORB - Stop @ OR High/Low - {target_str}\nFull Period - Ticker = {ticker}", fontsize=12)

    # Highlight out-of-sample period starting from a specific date
    start_date = datetime(2023, 2, 17)
    if not weekly_data.empty and start_date >= weekly_data.index[0] and start_date <= weekly_data.index[-1]:
        p2 = ax.axvspan(start_date, weekly_data.index[-1], alpha=0.1, color='green', label='Out-of-Sample')
        ax.legend(handles=[p1, p2], loc='upper left')
    else:
        ax.legend(loc='upper left')

    ax.set_yscale('log')
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0, subs=None))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    return fig, ax


# ================================
# Downloading and Loading into Memory
# ================================
# Download & merge data
data = download_and_merge_data()
if data is None:
    raise Exception("Data download failed.")

# Load the exported intraday data
p = pd.read_csv(OUTPUT_FILE, parse_dates=['caldt', 'day'])
days = pd.to_datetime(p['day'].unique())
days = pd.DatetimeIndex(sorted(days))

# ----------------------
# Backtest & Strategy Parameters (Edit as needed)
# ----------------------
orb_m       = 5             # Opening Range (minutes)
target_R    = float('inf')  # Profit target (use inf for no target)
commission  = 0.0005        # Commission per share
risk        = 0.01          # Equity risk per trade (1% of AUM)
max_Lev     = 1            # Maximum leverage
AUM_0       = 25000         # Starting capital

# ----------------------
# Run the Backtest
# ----------------------
str_df = backtest(days, p, orb_m, target_R, risk, max_Lev, AUM_0, commission)

# ----------------------
# Performance Analysis
# ----------------------
returns = price2return(str_df['AUM'].values)

display(Markdown("### Performance Summary"))
summary_stats = summary_statistics(returns)
display(summary_stats)

display(Markdown("### Monthly Performance"))
monthly_table = monthly_performance_table(returns, str_df['Date'])
display(monthly_table)

# ----------------------
# Equity Curve Visualization
# ----------------------
fig, ax = plot_equity_curve(str_df, AUM_0, orb_m, target_R, TICKER)
plt.tight_layout()
plt.show()

# --- Monthly Heatmap ---
def plot_monthly_heatmap(returns, dates, title="Monthly Return Heatmap"):
    returns_series = pd.Series(returns, index=pd.DatetimeIndex(dates))
    df = pd.DataFrame({'return': returns_series})
    df['Year'] = df.index.year
    df['Month'] = df.index.month
    pivot = df.pivot_table(index='Year', columns='Month', values='return', aggfunc=lambda x: np.prod(1 + x) - 1)
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot * 100, annot=True, fmt=".1f", cmap="RdYlGn", center=0)
    plt.title(title)
    plt.ylabel("Year")
    plt.xlabel("Month")
    plt.show()

plot_monthly_heatmap(returns, str_df['Date'], title=f"{TICKER} Monthly Return Heatmap")

# --- Enhanced Equity & Drawdown Plot ---
def plot_equity_and_drawdown(str_df):
    nav = str_df['AUM'].values
    dates = str_df['Date']
    peak = np.maximum.accumulate(nav)
    drawdown = (nav - peak) / peak
    fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax[0].plot(dates, nav, label="Equity Curve", color='blue')
    ax[0].set_ylabel("Equity ($)")
    ax[0].legend()
    ax[0].grid(True)
    ax[1].plot(dates, drawdown * 100, label="Drawdown (%)", color='red')
    ax[1].set_ylabel("Drawdown (%)")
    ax[1].legend()
    ax[1].grid(True)
    plt.xlabel("Date")
    plt.tight_layout()
    plt.show()

plot_equity_and_drawdown(str_df)

# --- Parameter Optimization ---
def optimize_parameters(days, p, orb_m_list, lev_list, target_R, risk, AUM_0, commission):
    results = []
    for orb_m in orb_m_list:
        for lev in lev_list:
            str_df = backtest(days, p, orb_m, target_R, risk, lev, AUM_0, commission)
            returns = price2return(str_df['AUM'].values)
            stats = summary_statistics(returns)
            total_return = float(stats.loc[stats['Metric'] == "Total Return (%)", "Value"].values[0])
            sharpe = float(stats.loc[stats['Metric'] == "Sharpe Ratio", "Value"].values[0])
            mdd = float(stats.loc[stats['Metric'] == "Max Drawdown (%)", "Value"].values[0])
            results.append({
                "ORB (min)": orb_m,
                "Leverage": lev,
                "Total Return (%)": total_return,
                "Sharpe Ratio": sharpe,
                "Max Drawdown (%)": mdd
            })
    return pd.DataFrame(results)

def plot_optimization_results(opt_df):
    pivot = opt_df.pivot(index="ORB (min)", columns="Leverage", values="Total Return (%)")
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlGnBu")
    plt.title("Total Return (%) by ORB Timeframe and Leverage")
    plt.ylabel("ORB (min)")
    plt.xlabel("Leverage")
    plt.show()

# Example: Try different ORB windows and leverage levels
orb_m_list = [5, 10, 15, 30]
lev_list = [1, 2, 3, 4, 5]
opt_df = optimize_parameters(days, p, orb_m_list, lev_list, target_R, risk, AUM_0, commission)
print(opt_df)
plot_optimization_results(opt_df)
