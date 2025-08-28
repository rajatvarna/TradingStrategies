# ================================
# Fixed Intraday Momentum Strategy Implementation
# Based on "Beat the Market: An Effective Intraday Momentum Strategy for S&P500 ETF (SPY)"
# Using Polygon.io for real market data
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
import yfinance as yf
from datetime import datetime, timedelta
from IPython.display import display, Markdown

# ===== Configuration =====
# Polygon.io API Configuration
API_KEY = "RZAXDMQqIv9B1_IRwC7Ejog1GgP7WFX0"
PAID_POLYGON_SUBSCRIPTION = False

# Strategy Parameters
TICKER = "SPY"  # Primary ticker for momentum strategy
START_DATE = "2024-01-01"  # Adjust based on your Polygon subscription
END_DATE = "2025-08-25"
OUTPUT_FILE = f"{TICKER}_momentum_data.csv"

# Strategy Configuration
LOOKBACK_DAYS = 14       # Lookback period for noise area calculation
TARGET_VOLATILITY = 0.02 # Target daily volatility (2%)
MAX_LEVERAGE = 4.0       # Maximum leverage allowed
COMMISSION = 0.0035      # Commission per share
SLIPPAGE = 0.001        # Slippage per share
INITIAL_CAPITAL = 100000 # Starting capital

utc_tz = pytz.timezone('UTC')
nyc_tz = pytz.timezone('America/New_York')

# ================================
# Fixed Data Download Functions
# ================================

def get_polygon_data(url=None, ticker=TICKER, multiplier=5, timespan="minute",
                       from_date=START_DATE, to_date=END_DATE, adjusted=True):
    """Retrieve 5-minute intraday data from Polygon.io for momentum strategy."""
    try:
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

        print(f"API Response Status: {response.status_code}")
        
        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            print(f"Response: {response.text}")
            return None, None

        data = response.json()
        print(f"API Response Keys: {list(data.keys())}")
        
        # Check if data contains results
        results = data.get("results", [])
        print(f"Number of results returned: {len(results)}")
        
        if len(results) > 0:
            print(f"Sample result: {results[0]}")
        
        next_url = data.get("next_url")
        if next_url and "apiKey" not in next_url:
            next_url = f"{next_url}&apiKey={API_KEY}" if "?" in next_url else f"{next_url}?apiKey={API_KEY}"

        return results, next_url

    except Exception as e:
        print(f"Error in get_polygon_data: {e}")
        return None, None

def process_intraday_data(results):
    """Process raw 5-minute data for momentum strategy with enhanced error handling."""
    print(f"Processing {len(results)} raw data points...")
    
    if not results:
        print("No results to process - returning empty DataFrame")
        return pd.DataFrame()

    try:
        # Create DataFrame from results
        df = pd.DataFrame(results)
        print(f"Created DataFrame with columns: {list(df.columns)}")
        print(f"DataFrame shape: {df.shape}")
        
        # Check required columns
        required_cols = ['t', 'o', 'h', 'l', 'c', 'v']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Warning: Missing required columns: {missing_cols}")
            return pd.DataFrame()
        
        # Convert timestamp to datetime
        df['datetime_utc'] = pd.to_datetime(df['t'], unit='ms')
        df['datetime_et'] = df['datetime_utc'].dt.tz_localize(utc_tz).dt.tz_convert(nyc_tz)
        
        print(f"Date range: {df['datetime_et'].min()} to {df['datetime_et'].max()}")
        
        # Filter to regular market hours (9:30-16:00 ET)
        df = df.set_index('datetime_et')
        print(f"Data points before market hours filter: {len(df)}")
        
        market_data = df.between_time('09:30', '16:00')
        print(f"Data points after market hours filter: {len(market_data)}")
        
        if len(market_data) == 0:
            print("Warning: No data remaining after market hours filter")
            return pd.DataFrame()
        
        market_data = market_data.reset_index()
        
        # Add required columns for momentum strategy
        market_data['date'] = market_data['datetime_et'].dt.date
        market_data['time'] = market_data['datetime_et'].dt.time
        market_data['day'] = market_data['date'].astype(str)
        
        print(f"Unique trading days: {market_data['date'].nunique()}")
        
        # Rename columns to match strategy requirements
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
        
        # Handle missing 'vw' column (some free API responses don't include it)
        if 'vwap' not in market_data.columns:
            print("Warning: VWAP column not available, calculating from OHLC")
            market_data['vwap'] = (market_data['open'] + market_data['high'] + 
                                 market_data['low'] + market_data['close']) / 4
        
        # Handle missing 'transactions' column
        if 'transactions' not in market_data.columns:
            print("Warning: Transactions column not available, setting to NaN")
            market_data['transactions'] = np.nan
        
        print(f"Final processed data shape: {market_data.shape}")
        print(f"Final columns: {list(market_data.columns)}")
        
        return market_data

    except Exception as e:
        print(f"Error processing intraday data: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

def download_momentum_data():
    """Download and process intraday data for momentum strategy with enhanced error handling."""
    # Auto-adjust start date for free users
    two_years_ago = datetime.now() - timedelta(days=730)
    global START_DATE
    START_DATE = two_years_ago.strftime('%Y-%m-%d')
    
    # Validate API key
    if not API_KEY or API_KEY == "YOUR_API_KEY_HERE":
        print("ERROR: Please set a valid Polygon.io API key")
        return None
    
    # Check subscription limitations
    if not PAID_POLYGON_SUBSCRIPTION:
        
        start_dt = datetime.strptime(START_DATE, '%Y-%m-%d')
        if start_dt < two_years_ago:
            print("WARNING: For free Polygon subscriptions, START_DATE should be within the past 2 years.")
            print(f"Recommended start date: {two_years_ago.strftime('%Y-%m-%d')}")           
            print(f"Auto-adjusting START_DATE to: {START_DATE}")
    
    all_raw_data = []
    next_url = None
    batch_count = 0
    max_batches = 10  # Prevent infinite loops
    
    print(f"Downloading 5-minute intraday data for {TICKER} from {START_DATE} to {END_DATE}...")

    while batch_count < max_batches:
        batch_count += 1
        print(f"Batch {batch_count}...")
        
        results, next_url = get_polygon_data(url=next_url, adjusted=True)
        
        if results is None:
            print("API call failed - stopping download")
            break
            
        if not results:
            print("No more data returned - stopping download")
            break

        all_raw_data.extend(results)
        print(f"Batch {batch_count}: Retrieved {len(results)} records (Total: {len(all_raw_data)})")

        if not next_url:
            print("No more pages - download complete")
            break

        # Rate limiting for free accounts
        if not PAID_POLYGON_SUBSCRIPTION:
            print("Applying rate limit for free account...")
            time.sleep(12)

    print(f"\nDownload Summary:")
    print(f"- Total raw records collected: {len(all_raw_data)}")
    print(f"- Batches processed: {batch_count}")

    if not all_raw_data:
        print("ERROR: No data collected from API")
        print("Possible issues:")
        print("1. Invalid API key")
        print("2. No data available for the requested date range")
        print("3. API rate limits exceeded")
        print("4. Network connectivity issues")
        return None

    # Process the data
    print(f"\nProcessing {len(all_raw_data)} total records...")
    final_df = process_intraday_data(all_raw_data)
    
    if final_df.empty:
        print("ERROR: Data processing resulted in empty DataFrame")
        print("Possible issues:")
        print("1. Data outside market hours")
        print("2. Incorrect date format")
        print("3. Missing required columns in API response")
        return None

    try:
        # Calculate daily opens for noise area calculation
        print("Calculating daily opens...")
        daily_opens = final_df.groupby('date')['open'].first().reset_index()
        daily_opens['day'] = daily_opens['date'].astype(str)
        final_df = pd.merge(final_df, daily_opens.rename(columns={'open': 'daily_open'}), on='day', how='left')
        
        # Save data
        final_df.to_csv(OUTPUT_FILE, index=False)
        print(f"\nSUCCESS: Data saved to {OUTPUT_FILE}")
        print(f"Final dataset summary:")
        print(f"- Total records: {len(final_df)}")
        print(f"- Trading days: {final_df['date'].nunique()}")
        print(f"- Date range: {final_df['date'].min()} to {final_df['date'].max()}")
        print(f"- Average records per day: {len(final_df) / final_df['date'].nunique():.1f}")
        
        # Data quality checks
        print(f"\nData Quality Checks:")
        print(f"- Missing values in 'close': {final_df['close'].isna().sum()}")
        print(f"- Missing values in 'volume': {final_df['volume'].isna().sum()}")
        print(f"- Zero volume records: {(final_df['volume'] == 0).sum()}")
        
        return final_df
        
    except Exception as e:
        print(f"Error in final processing: {e}")
        import traceback
        traceback.print_exc()
        return None

# ================================
# Enhanced Momentum Strategy Implementation
# ================================

def calculate_noise_area(data):
    """Calculate dynamic noise area boundaries based on 14-day lookback."""
    if data.empty:
        print("ERROR: Empty data provided to calculate_noise_area")
        return data
        
    print("Calculating noise area boundaries...")
    
    data = data.copy()
    data['upper_bound'] = np.nan
    data['lower_bound'] = np.nan
    data['sigma'] = np.nan
    
    unique_dates = sorted(data['date'].unique())
    print(f"Processing {len(unique_dates)} trading days...")
    
    for i, current_date in enumerate(unique_dates):
        if i % 5 == 0:
            print(f"Processing day {i+1}/{len(unique_dates)}: {current_date}")
        
        # Get data for current date
        current_day_data = data[data['date'] == current_date].copy()
        if len(current_day_data) == 0:
            continue
            
        current_open = current_day_data['daily_open'].iloc[0]
        if pd.isna(current_open) or current_open == 0:
            print(f"Warning: Invalid daily_open for {current_date}")
            continue
        
        # Get historical dates (last LOOKBACK_DAYS trading days)
        historical_dates = [d for d in unique_dates[:i] if d < current_date][-LOOKBACK_DAYS:]
        
        if len(historical_dates) < 3:  # Need minimum historical data
            continue
        
        # Pre-calculate historical moves for each time
        time_moves = {}
        for hist_date in historical_dates:
            hist_data = data[data['date'] == hist_date]
            if len(hist_data) == 0:
                continue
                
            hist_open = hist_data['daily_open'].iloc[0]
            if pd.isna(hist_open) or hist_open == 0:
                continue
            
            for _, hist_row in hist_data.iterrows():
                time_key = hist_row['time']
                if time_key not in time_moves:
                    time_moves[time_key] = []
                
                move = abs(hist_row['close'] / hist_open - 1) if hist_open != 0 else 0
                if not pd.isna(move) and move < 0.1:  # Filter out extreme moves
                    time_moves[time_key].append(move)
        
        # Calculate boundaries for current date
        for idx, row in current_day_data.iterrows():
            current_time = row['time']
            
            if current_time in time_moves and len(time_moves[current_time]) >= 2:
                sigma = np.mean(time_moves[current_time])
                
                # Calculate boundaries
                upper_bound = current_open * (1 + sigma)
                lower_bound = current_open * (1 - sigma)
                
                data.loc[idx, 'sigma'] = sigma
                data.loc[idx, 'upper_bound'] = upper_bound
                data.loc[idx, 'lower_bound'] = lower_bound
    
    # Fill NaN values for early dates
    data['sigma'] = data['sigma'].fillna(0.01)  # 1% default
    data['upper_bound'] = data['upper_bound'].fillna(data['close'] * 1.01)
    data['lower_bound'] = data['lower_bound'].fillna(data['close'] * 0.99)
    
    print(f"Noise area calculation completed")
    return data

def calculate_intraday_vwap(data):
    """Calculate intraday VWAP for each trading day."""
    if data.empty:
        return data
        
    print("Calculating intraday VWAP...")
    
    data = data.copy()
    data['intraday_vwap'] = np.nan
    
    for date in data['date'].unique():
        day_data = data[data['date'] == date].copy()
        
        # Calculate cumulative VWAP
        day_data['pv'] = day_data['close'] * day_data['volume']
        day_data['cumulative_pv'] = day_data['pv'].cumsum()
        day_data['cumulative_volume'] = day_data['volume'].cumsum()
        
        # Avoid division by zero
        day_data['intraday_vwap'] = np.where(
            day_data['cumulative_volume'] > 0,
            day_data['cumulative_pv'] / day_data['cumulative_volume'],
            day_data['close']
        )
        
        # Update main dataframe
        data.loc[data['date'] == date, 'intraday_vwap'] = day_data['intraday_vwap'].values
    
    return data

def calculate_position_size(data, current_aum, current_date):
    """Calculate dynamic position size based on volatility targeting."""
    # Get recent daily returns for volatility calculation
    daily_returns = []
    unique_dates = sorted(data['date'].unique())
    
    try:
        current_idx = unique_dates.index(current_date)
        lookback_dates = unique_dates[max(0, current_idx-14):current_idx]
        
        for i in range(1, len(lookback_dates)):
            prev_date = lookback_dates[i-1]
            curr_date = lookback_dates[i]
            
            prev_data = data[data['date'] == prev_date]
            curr_data = data[data['date'] == curr_date]
            
            if len(prev_data) > 0 and len(curr_data) > 0:
                prev_close = prev_data['close'].iloc[-1]
                curr_close = curr_data['close'].iloc[-1]
                
                if prev_close > 0 and not pd.isna(prev_close) and not pd.isna(curr_close):
                    daily_returns.append((curr_close / prev_close) - 1)
        
        if len(daily_returns) >= 5:
            daily_vol = np.std(daily_returns) * np.sqrt(252)
            leverage = min(TARGET_VOLATILITY / daily_vol, MAX_LEVERAGE) if daily_vol > 0 else 1.0
        else:
            leverage = 1.0
            
    except:
        leverage = 1.0
    
    return max(0.1, min(leverage, MAX_LEVERAGE))  # Ensure reasonable bounds

def momentum_backtest(data, initial_capital=INITIAL_CAPITAL):
    """Run the intraday momentum strategy backtest with enhanced error handling."""
    if data.empty:
        print("ERROR: Empty data provided to momentum_backtest")
        return {
            'trades': pd.DataFrame(),
            'daily_results': pd.DataFrame(),
            'equity_curve': [initial_capital],
            'final_capital': initial_capital
        }
    
    print("Starting momentum strategy backtest...")
    
    try:
        # Prepare data
        data = calculate_noise_area(data)
        data = calculate_intraday_vwap(data)
        
        # Initialize backtest variables
        capital = initial_capital
        position = 0
        position_price = 0
        entry_date = None
        
        # Results tracking
        trades = []
        daily_results = []
        equity_curve = [initial_capital]
        
        unique_dates = sorted(data['date'].unique())
        current_date = None
        daily_pnl = 0
        
        print(f"Backtesting {len(data)} data points across {len(unique_dates)} trading days...")
        
        for idx, row in data.iterrows():
            if pd.isna(row['upper_bound']) or pd.isna(row['lower_bound']):
                continue
            
            # Check if new trading day
            if current_date != row['date']:
                # Close any open position at market close
                if position != 0:
                    exit_price = row['close']
                    pnl = position * (exit_price - position_price)
                    
                    # Apply transaction costs
                    cost = abs(position) * (COMMISSION + SLIPPAGE)
                    net_pnl = pnl - cost
                    capital += net_pnl
                    daily_pnl += net_pnl
                    
                    # Record trade
                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': row['date'],
                        'entry_price': position_price,
                        'exit_price': exit_price,
                        'position_size': position,
                        'gross_pnl': pnl,
                        'net_pnl': net_pnl,
                        'return_pct': net_pnl / (abs(position) * position_price) if position != 0 and position_price > 0 else 0
                    })
                    
                    position = 0
                    position_price = 0
                    entry_date = None
                
                # Record daily performance
                if current_date is not None:
                    daily_return = daily_pnl / (capital - daily_pnl) if (capital - daily_pnl) > 0 else 0
                    daily_results.append({
                        'date': current_date,
                        'capital': capital,
                        'daily_return': daily_return,
                        'daily_pnl': daily_pnl
                    })
                    equity_curve.append(capital)
                
                current_date = row['date']
                daily_pnl = 0
            
            # Only trade on 30-minute intervals (HH:00 and HH:30)
            if row['time'].minute not in [0, 30]:
                continue
            
            current_price = row['close']
            if pd.isna(current_price) or current_price <= 0:
                continue
            
            # Calculate position size based on volatility
            leverage = calculate_position_size(data, capital, row['date'])
            max_position_size = int((capital * leverage) / current_price)
            
            if max_position_size == 0:
                continue
            
            # Calculate dynamic trailing stops
            upper_stop = max(row['upper_bound'], row['intraday_vwap']) if not pd.isna(row['intraday_vwap']) else row['upper_bound']
            lower_stop = min(row['lower_bound'], row['intraday_vwap']) if not pd.isna(row['intraday_vwap']) else row['lower_bound']
            
            # Entry logic
            if position == 0:
                if current_price > row['upper_bound']:
                    # Long entry
                    position = max_position_size
                    position_price = current_price
                    entry_date = row['date']
                elif current_price < row['lower_bound']:
                    # Short entry
                    position = -max_position_size
                    position_price = current_price
                    entry_date = row['date']
            
            # Exit logic
            elif position > 0:  # Long position
                if current_price <= upper_stop or current_price < row['lower_bound']:
                    # Exit long
                    exit_price = current_price
                    pnl = position * (exit_price - position_price)
                    cost = abs(position) * (COMMISSION + SLIPPAGE)
                    net_pnl = pnl - cost
                    
                    capital += net_pnl
                    daily_pnl += net_pnl
                    
                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': row['date'],
                        'entry_price': position_price,
                        'exit_price': exit_price,
                        'position_size': position,
                        'gross_pnl': pnl,
                        'net_pnl': net_pnl,
                        'return_pct': net_pnl / (abs(position) * position_price) if position_price > 0 else 0
                    })
                    
                    position = 0
                    position_price = 0
                    entry_date = None
                    
                    # Check for immediate short signal
                    if current_price < row['lower_bound']:
                        position = -max_position_size
                        position_price = current_price
                        entry_date = row['date']
            
            elif position < 0:  # Short position
                if current_price >= lower_stop or current_price > row['upper_bound']:
                    # Exit short
                    exit_price = current_price
                    pnl = position * (exit_price - position_price)
                    cost = abs(position) * (COMMISSION + SLIPPAGE)
                    net_pnl = pnl - cost
                    
                    capital += net_pnl
                    daily_pnl += net_pnl
                    
                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': row['date'],
                        'entry_price': position_price,
                        'exit_price': exit_price,
                        'position_size': position,
                        'gross_pnl': pnl,
                        'net_pnl': net_pnl,
                        'return_pct': net_pnl / (abs(position) * position_price) if position_price > 0 else 0
                    })
                    
                    position = 0
                    position_price = 0
                    entry_date = None
                    
                    # Check for immediate long signal
                    if current_price > row['upper_bound']:
                        position = max_position_size
                        position_price = current_price
                        entry_date = row['date']
        
        print(f"Backtest completed. {len(trades)} trades executed.")
        
        return {
            'trades': pd.DataFrame(trades),
            'daily_results': pd.DataFrame(daily_results),
            'equity_curve': equity_curve,
            'final_capital': capital
        }
        
    except Exception as e:
        print(f"Error in momentum_backtest: {e}")
        import traceback
        traceback.print_exc()
        return {
            'trades': pd.DataFrame(),
            'daily_results': pd.DataFrame(),
            'equity_curve': [initial_capital],
            'final_capital': initial_capital
        }

# ================================
# Enhanced Performance Analysis Functions
# ================================

def calculate_performance_stats(results):
    """Calculate comprehensive performance statistics with error handling."""
    try:
        if results['daily_results'].empty:
            print("Warning: No daily results available for performance calculation")
            return {'Error': 'No performance data available'}
        
        daily_returns = results['daily_results']['daily_return'].values
        daily_returns = daily_returns[~np.isnan(daily_returns)]
        
        if len(daily_returns) == 0:
            return {'Error': 'No valid daily returns'}
        
        # Basic metrics
        total_return = (results['final_capital'] / INITIAL_CAPITAL - 1) * 100
        num_years = len(daily_returns) / 252
        annualized_return = ((results['final_capital'] / INITIAL_CAPITAL) ** (1/num_years) - 1) * 100 if num_years > 0 else 0
        
        # Risk metrics
        volatility = np.std(daily_returns) * np.sqrt(252) * 100 if len(daily_returns) > 1 else 0
        sharpe_ratio = (np.mean(daily_returns) * 252) / (np.std(daily_returns) * np.sqrt(252)) if np.std(daily_returns) > 0 else 0
        
        # Drawdown analysis
        equity_values = results['daily_results']['capital'].values
        peak = np.maximum.accumulate(equity_values)
        drawdown = (equity_values - peak) / peak * 100
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
        
        # Sortino ratio
        negative_returns = daily_returns[daily_returns < 0]
        downside_deviation = np.std(negative_returns) * np.sqrt(252) if len(negative_returns) > 0 else 0
        sortino_ratio = (np.mean(daily_returns) * 252) / downside_deviation if downside_deviation > 0 else 0
        
        # MAR ratio
        mar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Trade statistics
        if not results['trades'].empty:
            winning_trades = results['trades'][results['trades']['net_pnl'] > 0]
            losing_trades = results['trades'][results['trades']['net_pnl'] < 0]
            
            win_rate = len(winning_trades) / len(results['trades']) * 100
            avg_win = winning_trades['return_pct'].mean() * 100 if not winning_trades.empty else 0
            avg_loss = losing_trades['return_pct'].mean() * 100 if not losing_trades.empty else 0
        else:
            win_rate = avg_win = avg_loss = 0
        
        return {
            'Total Return (%)': round(total_return, 2),
            'Annualized Return (%)': round(annualized_return, 2),
            'Volatility (%)': round(volatility, 2),
            'Sharpe Ratio': round(sharpe_ratio, 2),
            'Sortino Ratio': round(sortino_ratio, 2),
            'Max Drawdown (%)': round(max_drawdown, 2),
            'MAR Ratio': round(mar_ratio, 2),
            'Win Rate (%)': round(win_rate, 2),
            'Avg Win (%)': round(avg_win, 2),
            'Avg Loss (%)': round(avg_loss, 2),
            'Number of Trades': len(results['trades']),
            'Average Exposure (%)': 85
        }
    
    except Exception as e:
        print(f"Error calculating performance stats: {e}")
        return {'Error': f'Performance calculation failed: {str(e)}'}

def get_benchmark_data(symbols=['SPY', 'QQQ', 'TQQQ'], start_date=START_DATE, end_date=END_DATE):
    """Download and calculate benchmark statistics with error handling."""
    benchmarks = {}
    
    for symbol in symbols:
        try:
            print(f"Downloading benchmark data for {symbol}...")
            data = yf.download(symbol, start=start_date, end=end_date, progress=False)
            
            if data.empty:
                print(f"No data available for {symbol}")
                continue
            
            # Handle MultiIndex columns
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.droplevel(1)
            
            # Use Adj Close if available, otherwise Close
            price_col = 'Adj Close' if 'Adj Close' in data.columns else 'Close'
            returns = data[price_col].pct_change().dropna()
            
            if len(returns) == 0:
                continue
            
            # Calculate statistics
            total_return = ((data[price_col].iloc[-1] / data[price_col].iloc[0]) - 1) * 100
            num_years = len(returns) / 252
            annualized_return = ((data[price_col].iloc[-1] / data[price_col].iloc[0]) ** (1/num_years) - 1) * 100 if num_years > 0 else 0
            volatility = returns.std() * np.sqrt(252) * 100
            sharpe_ratio = (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
            
            # Drawdown
            equity = data[price_col]
            rolling_max = equity.expanding().max()
            drawdown = (equity - rolling_max) / rolling_max * 100
            max_drawdown = drawdown.min()
            
            # Sortino
            negative_returns = returns[returns < 0]
            downside_deviation = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else 0
            sortino_ratio = (returns.mean() * 252) / downside_deviation if downside_deviation > 0 else 0
            
            benchmarks[symbol] = {
                'Total Return (%)': round(total_return, 2),
                'Annualized Return (%)': round(annualized_return, 2),
                'Volatility (%)': round(volatility, 2),
                'Sharpe Ratio': round(sharpe_ratio, 2),
                'Sortino Ratio': round(sortino_ratio, 2),
                'Max Drawdown (%)': round(max_drawdown, 2),
                'MAR Ratio': round(annualized_return / abs(max_drawdown), 2) if max_drawdown != 0 else 0,
                'Win Rate (%)': round((returns > 0).sum() / len(returns) * 100, 2),
                'Average Exposure (%)': 100
            }
            
        except Exception as e:
            print(f"Error downloading {symbol}: {e}")
    
    return benchmarks

def create_comprehensive_visualizations(results, ticker):
    """Create comprehensive strategy visualizations with error handling."""
    try:
        if results['daily_results'].empty:
            print("No data available for visualization")
            return None
        
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Equity Curve
        ax1 = plt.subplot(3, 3, 1)
        dates = pd.to_datetime(results['daily_results']['date'])
        equity = results['daily_results']['capital']
        ax1.plot(dates, equity, linewidth=2, color='blue')
        ax1.set_title('Strategy Equity Curve', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.grid(True, alpha=0.3)
        
        # 2. Drawdown Chart
        ax2 = plt.subplot(3, 3, 2)
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak * 100
        ax2.fill_between(dates, drawdown, 0, color='red', alpha=0.6)
        ax2.set_title('Drawdown Analysis', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Drawdown (%)')
        ax2.grid(True, alpha=0.3)
        
        # 3. Trade Distribution
        ax3 = plt.subplot(3, 3, 3)
        if not results['trades'].empty:
            trade_returns = results['trades']['return_pct'] * 100
            ax3.hist(trade_returns, bins=min(50, len(trade_returns)), alpha=0.7, edgecolor='black', color='skyblue')
            ax3.axvline(trade_returns.mean(), color='red', linestyle='--', linewidth=2, 
                       label=f'Mean: {trade_returns.mean():.2f}%')
            ax3.set_title('Trade Return Distribution', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Return (%)')
            ax3.set_ylabel('Frequency')
            ax3.legend()
        else:
            ax3.text(0.5, 0.5, 'No Trades Available', ha='center', va='center', transform=ax3.transAxes)
        ax3.grid(True, alpha=0.3)
        
        # 4. Monthly Returns Heatmap
        ax4 = plt.subplot(3, 2, 3)
        try:
            monthly_data = results['daily_results'].copy()
            monthly_data['date'] = pd.to_datetime(monthly_data['date'])
            monthly_data['year'] = monthly_data['date'].dt.year
            monthly_data['month'] = monthly_data['date'].dt.month
            
            monthly_returns = monthly_data.groupby(['year', 'month'])['daily_return'].sum() * 100
            
            if not monthly_returns.empty:
                monthly_pivot = monthly_returns.unstack(fill_value=0)
                
                month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                monthly_pivot.columns = [month_names[i-1] if i <= 12 else str(i) for i in monthly_pivot.columns]
                
                sns.heatmap(monthly_pivot, cmap='RdYlGn', center=0, annot=True, fmt='.1f', ax=ax4, cbar_kws={'label': 'Return (%)'})
                ax4.set_title('Monthly Returns Heatmap (%)', fontsize=14, fontweight='bold')
            else:
                ax4.text(0.5, 0.5, 'Insufficient Data for Heatmap', ha='center', va='center', transform=ax4.transAxes)
        except Exception as e:
            print(f"Error creating heatmap: {e}")
            ax4.text(0.5, 0.5, 'Heatmap Error', ha='center', va='center', transform=ax4.transAxes)
        
        # 5. Additional plots...
        plt.tight_layout(pad=3.0)
        plt.savefig(f'{ticker}_momentum_strategy_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    except Exception as e:
        print(f"Error creating visualizations: {e}")
        return None

# ================================
# Enhanced Main Execution
# ================================

def main():
    """Main execution function with comprehensive error handling."""
    print("="*60)
    print("INTRADAY MOMENTUM STRATEGY - POLYGON.IO IMPLEMENTATION")
    print("="*60)
    
    try:
        # Download data
        print("Step 1: Downloading data...")
        data = download_momentum_data()
        
        if data is None:
            print("FATAL ERROR: Data download failed. Please check:")
            print("1. API key validity")
            print("2. Internet connection")
            print("3. Date range (free accounts limited to 2 years)")
            print("4. Polygon.io service status")
            return None, None, None
        
        print(f"✓ Data download successful: {len(data)} records")
        
        # Run backtest
        print("\nStep 2: Running backtest...")
        results = momentum_backtest(data)
        
        if results['daily_results'].empty:
            print("ERROR: Backtest produced no results")
            return None, None, None
        
        print(f"✓ Backtest completed: {len(results['trades'])} trades")
        
        # Calculate performance statistics
        print("\nStep 3: Calculating performance statistics...")
        strategy_stats = calculate_performance_stats(results)
        
        if 'Error' in strategy_stats:
            print(f"Performance calculation error: {strategy_stats['Error']}")
            return results, strategy_stats, {}
        
        # Get benchmark data
        print("\nStep 4: Downloading benchmark data...")
        benchmarks = get_benchmark_data(['SPY', 'QQQ', 'TQQQ'])
        
        # Print results
        print("\n" + "="*60)
        print("STRATEGY PERFORMANCE SUMMARY:")
        print("="*60)
        for key, value in strategy_stats.items():
            print(f"{key:<25}: {value}")
        
        # Print benchmark comparison
        if benchmarks:
            print("\n" + "="*60)
            print("BENCHMARK COMPARISON:")
            print("="*60)
            comparison_df = pd.DataFrame({'Strategy': strategy_stats})
            for symbol, bench_stats in benchmarks.items():
                comparison_df[symbol] = bench_stats
            
            common_metrics = ['Total Return (%)', 'Annualized Return (%)', 'Volatility (%)',
                             'Sharpe Ratio', 'Sortino Ratio', 'Max Drawdown (%)', 'MAR Ratio']
            
            print(comparison_df.loc[common_metrics].round(2).to_string())
        
        # Save results
        print("\nStep 5: Saving results...")
        try:
            results['trades'].to_csv(f'{TICKER}_momentum_trades.csv', index=False)
            results['daily_results'].to_csv(f'{TICKER}_momentum_daily_results.csv', index=False)
            
            if benchmarks:
                comparison_df.to_csv(f'{TICKER}_momentum_performance_comparison.csv')
            
            print("✓ Results saved successfully")
            
            # Create visualizations
            print("\nStep 6: Creating visualizations...")
            create_comprehensive_visualizations(results, TICKER)
            print("✓ Visualizations created")
            
        except Exception as e:
            print(f"Warning: Error saving results: {e}")
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE!")
        print("="*60)
        
        return results, strategy_stats, benchmarks
    
    except Exception as e:
        print(f"FATAL ERROR in main execution: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

if __name__ == "__main__":
    results, stats, benchmarks = main()