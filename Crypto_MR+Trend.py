import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import os
import glob
import math
warnings.filterwarnings('ignore')

# Create charts folder
CHARTS_FOLDER = 'backtest_charts'
if not os.path.exists(CHARTS_FOLDER):
    os.makedirs(CHARTS_FOLDER)

# ============================================================================
# CONFIGURATION
# ============================================================================
INITIAL_CAPITAL = 100000
COMMISSION = 0.002  # 0.2% taker fee
SLIPPAGE = 0.001    # 0.1%
MONTE_CARLO_SIMS = 1000
MAX_POSITIONS = 10  # For Mean Reversion

# Trend following specific
TREND_SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'DOGEUSDT', 'XRPUSDT', 'SOLUSDT']
MAX_TREND_POSITIONS = 5
TREND_POSITION_SIZE = 0.20  # 20% per position

# ============================================================================
# 1. LOAD BINANCE DATA
# ============================================================================
def load_binance_data(folder_path='all_binance'):
    """Load all crypto data from Binance CSV files"""
    csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
    
    if not csv_files:
        print(f"No CSV files found in {folder_path}. Please add CSV files.")
        return None
    
    print(f"Found {len(csv_files)} CSV files")
    
    all_data = {}
    for file in csv_files:
        symbol = os.path.basename(file).replace('_daily.csv', '').replace('.csv', '')
        try:
            df = pd.read_csv(file)
            
            # Handle different possible column names
            time_col = None
            for col in ['time', 'timestamp', 'date', 'Date', 'Time']:
                if col in df.columns:
                    time_col = col
                    break
            
            if time_col is None:
                print(f"  Skipping {symbol}: No time column found")
                continue
            
            # Parse dates
            df[time_col] = pd.to_datetime(df[time_col], utc=True)
            df = df.set_index(time_col)
            df.index = df.index.tz_localize(None)  # Remove timezone
            
            # Standardize column names
            df.columns = df.columns.str.lower()
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            
            if not all(col in df.columns for col in required_cols):
                print(f"  Skipping {symbol}: Missing required columns")
                continue
            
            df = df[required_cols].copy()
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            
            # Remove duplicates and sort
            df = df[~df.index.duplicated(keep='first')]
            df = df.sort_index()
            
            # Keep only if we have at least 100 days of data
            if len(df) < 100:
                print(f"  Skipping {symbol}: Only {len(df)} days")
                continue
            
            all_data[symbol] = df
            print(f"  Loaded {symbol}: {len(df)} days from {df.index[0].date()} to {df.index[-1].date()}")
            
        except Exception as e:
            print(f"  Error loading {file}: {e}")
    
    return all_data

# ============================================================================
# 2. MEAN REVERSION STRATEGY (IBS with Volatility Bands)
# ============================================================================
def backtest_mean_reversion_portfolio(all_data, initial_capital=INITIAL_CAPITAL, max_positions=5):
    """
    Mean Reversion portfolio using IBS (Internal Bar Strength):
    - Only trade top 5 cryptos: BTC, ETH, DOGE, XRP, SOL
    - Calculate rolling mean of (High - Low) over 25 days
    - Calculate IBS: (Close - Low) / (High - Low)
    - Calculate lower band: 10-day rolling High - 2.5 × mean(High - Low)
    - Entry: Close < Lower Band AND IBS < 0.3
    - Exit: 2 × ATR(10) stop loss OR 10 days time stop
    - Max concurrent positions: max_positions
    """
    # Select only top 5 cryptos
    symbols = ['BTCUSDT', 'ETHUSDT', 'DOGEUSDT', 'XRPUSDT', 'SOLUSDT']
    selected_data = {s: all_data[s] for s in symbols if s in all_data}
    
    if len(selected_data) < 1:
        print("No valid symbols for MR backtest.")
        return pd.DataFrame(), pd.Series(), pd.Series()

    # Prepare indicators per symbol
    prepared = {}
    for sym, df in selected_data.items():
        df = df.copy().sort_index()
        if 'Close' not in df.columns or df['Close'].dropna().empty:
            continue
        
        # Calculate High - Low range
        df['HL_Range'] = df['High'] - df['Low']
        
        # Rolling mean of High - Low over 25 days
        df['Mean_HL_25'] = df['HL_Range'].rolling(window=25).mean()
        
        # IBS indicator: (Close - Low) / (High - Low)
        df['IBS'] = (df['Close'] - df['Low']) / df['HL_Range']
        
        # Rolling High over 10 days
        df['Rolling_High_10'] = df['High'].rolling(window=10).max()
        
        # Lower Band: 10-day rolling High - 2.5 × mean(High - Low)
        df['Lower_Band'] = df['Rolling_High_10'] - 2.5 * df['Mean_HL_25']
        
        # ATR(10) for stop loss
        df['TR'] = df[['High', 'Low', 'Close']].apply(
            lambda row: max(
                row['High'] - row['Low'],
                abs(row['High'] - df['Close'].shift(1).loc[row.name]) if pd.notna(df['Close'].shift(1).loc[row.name]) else row['High'] - row['Low'],
                abs(row['Low'] - df['Close'].shift(1).loc[row.name]) if pd.notna(df['Close'].shift(1).loc[row.name]) else row['High'] - row['Low']
            ),
            axis=1
        )
        df['ATR_10'] = df['TR'].rolling(window=10).mean()
        
        prepared[sym] = df

    if not prepared:
        print("No valid symbols for MR backtest.")
        return pd.DataFrame(), pd.Series(), pd.Series()

    # union of all dates
    all_dates = sorted({d for df in prepared.values() for d in df.index})

    capital = initial_capital
    positions = {}  # sym -> dict with shares, entry_price, entry_date, entry_atr
    trades = []
    equity_ts = []
    exposure_ts = []

    for current_date in all_dates:
        # First handle exits: 2 × ATR(10) stop OR 10 days time stop
        to_exit = []
        for sym, pos in list(positions.items()):
            df = prepared.get(sym)
            if current_date not in df.index:
                continue
            
            close = df.at[current_date, 'Close']
            
            if pd.isna(close):
                continue
            
            holding_days = (current_date - pos['entry_date']).days
            
            # Calculate drawdown from entry
            drawdown = pos['entry_price'] - float(close)
            stop_loss_threshold = 2 * pos['entry_atr']  # 2x ATR stop
            
            # Exit if stop loss hit OR time stop hit
            if drawdown > stop_loss_threshold or holding_days >= 10:
                exit_price = float(close) * (1 - SLIPPAGE) * (1 - COMMISSION)
                pnl = (exit_price - pos['entry_price']) * pos['shares']
                capital += pos['shares'] * exit_price
                
                exit_reason = 'Stop_Loss' if drawdown > stop_loss_threshold else 'Time_Stop'
                
                trades.append({
                    'Strategy': 'MR_IBS',
                    'Symbol': sym,
                    'Entry_Date': pos['entry_date'],
                    'Exit_Date': current_date,
                    'Entry_Price': pos['entry_price'],
                    'Exit_Price': exit_price,
                    'Shares': pos['shares'],
                    'PnL': pnl,
                    'Return_%': (exit_price / pos['entry_price'] - 1) * 100,
                    'Holding_Days': holding_days,
                    'Exit_Reason': exit_reason
                })
                to_exit.append(sym)
        
        for sym in to_exit:
            positions.pop(sym, None)

        # Entry logic: Close < Lower Band AND IBS < 0.3
        candidates = []
        for sym, df in prepared.items():
            if sym in positions:
                continue
            if current_date not in df.index:
                continue
            
            row = df.loc[current_date]
            
            # Check if all required indicators are available
            if pd.isna(row['Close']) or pd.isna(row['Lower_Band']) or pd.isna(row['IBS']) or pd.isna(row['ATR_10']):
                continue
            
            close = float(row['Close'])
            lower_band = float(row['Lower_Band'])
            ibs = float(row['IBS'])
            
            # Entry condition: Close < Lower Band AND IBS < 0.3
            if close < lower_band and ibs < 0.3:
                candidates.append((sym, float(row['ATR_10'])))
        
        # Enter positions (up to max_positions)
        slots = max_positions - len(positions)
        if slots > 0 and candidates:
            for sym, atr in candidates[:slots]:
                df = prepared[sym]
                entry_price = float(df.at[current_date, 'Close']) * (1 + SLIPPAGE) * (1 + COMMISSION)
                
                # Equal position sizing
                position_capital = initial_capital / max_positions
                shares = math.floor((position_capital / entry_price) * 1e8) / 1e8
                cost = shares * entry_price
                
                if shares <= 0 or cost > capital:
                    continue
                
                capital -= cost
                positions[sym] = {
                    'shares': shares,
                    'entry_price': entry_price,
                    'entry_date': current_date,
                    'entry_atr': atr
                }

        # compute portfolio value using current day's close
        portfolio_value = capital
        for sym, pos in positions.items():
            df = prepared[sym]
            if current_date in df.index and not pd.isna(df.at[current_date, 'Close']):
                mark = float(df.at[current_date, 'Close'])
            else:
                try:
                    last_loc = df.index.get_indexer([current_date], method='ffill')[0]
                    mark = float(df['Close'].iat[last_loc])
                except Exception:
                    mark = pos['entry_price']
            portfolio_value += pos['shares'] * mark
        
        equity_ts.append(portfolio_value)
        exposure_ts.append(len(positions))

    # close remaining positions at last available price
    for sym, pos in list(positions.items()):
        df = prepared[sym]
        last_idx = df['Close'].last_valid_index()
        if last_idx is not None:
            exit_price = float(df.at[last_idx, 'Close']) * (1 - SLIPPAGE) * (1 - COMMISSION)
            pnl = (exit_price - pos['entry_price']) * pos['shares']
            holding_days = (last_idx - pos['entry_date']).days
            trades.append({
                'Strategy': 'MR_IBS',
                'Symbol': sym,
                'Entry_Date': pos['entry_date'],
                'Exit_Date': last_idx,
                'Entry_Price': pos['entry_price'],
                'Exit_Price': exit_price,
                'Shares': pos['shares'],
                'PnL': pnl,
                'Return_%': (exit_price / pos['entry_price'] - 1) * 100,
                'Holding_Days': holding_days,
                'Exit_Reason': 'End_of_Data'
            })

    trades_df = pd.DataFrame(trades)
    equity_series = pd.Series(equity_ts, index=all_dates)
    exposure_series = pd.Series(exposure_ts, index=all_dates)
    return trades_df, equity_series, exposure_series

# ============================================================================
# 3. TREND FOLLOWING STRATEGY (50 DMA)
# ============================================================================
def backtest_trend_portfolio(all_data, initial_capital=INITIAL_CAPITAL):
    """Simple Trend Following: Only BTC, ETH, DOGE, XRP, SOL. Max 5 positions, 20% weight each."""
    symbols = ['BTCUSDT', 'ETHUSDT', 'DOGEUSDT', 'XRPUSDT', 'SOLUSDT']
    selected_data = {s: all_data[s] for s in symbols if s in all_data}
    if len(selected_data) < 1:
        print("No valid symbols for trend following.")
        return pd.DataFrame(), pd.Series(), pd.Series()

    all_dates = sorted({d for df in selected_data.values() for d in df.index})

    prepared_data = {}
    for symbol, df in selected_data.items():
        df = df.copy()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df = df.reindex(all_dates, method='ffill')
        prepared_data[symbol] = df

    capital = initial_capital
    positions = {}
    trades = []
    equity_curve = []
    daily_exposure = []

    for date in all_dates:
        portfolio_value = capital
        for symbol, pos in list(positions.items()):
            if pd.notna(prepared_data[symbol].loc[date, 'Close']):
                current_price = prepared_data[symbol].loc[date, 'Close']
                portfolio_value += pos['shares'] * current_price
            else:
                del positions[symbol]

        positions_to_close = []
        for symbol, pos in list(positions.items()):
            df = prepared_data[symbol]
            if pd.isna(df.loc[date, 'Close']):
                positions_to_close.append(symbol)
                continue
            close = df.loc[date, 'Close']
            sma_50 = df.loc[date, 'SMA_50']
            if not np.isnan(sma_50) and close < sma_50:
                exit_price = close * (1 - SLIPPAGE) * (1 - COMMISSION)
                shares = pos['shares']
                entry_price = pos['entry_price']
                pnl = (exit_price - entry_price) * shares
                capital += shares * exit_price
                trades.append({
                    'Strategy': 'Trend_Following',
                    'Symbol': symbol,
                    'Entry_Date': pos['entry_date'],
                    'Exit_Date': date,
                    'Entry_Price': entry_price,
                    'Exit_Price': exit_price,
                    'Shares': shares,
                    'PnL': pnl,
                    'Return_%': (exit_price / entry_price - 1) * 100
                })
                positions_to_close.append(symbol)
        for symbol in positions_to_close:
            if symbol in positions:
                del positions[symbol]

        slots_available = 5 - len(positions)
        for symbol, df in prepared_data.items():
            if symbol in positions or slots_available == 0:
                continue
            if pd.isna(df.loc[date, 'Close']):
                continue
            close = df.loc[date, 'Close']
            sma_50 = df.loc[date, 'SMA_50']
            if not np.isnan(sma_50) and close > sma_50:
                entry_price = close * (1 + SLIPPAGE) * (1 + COMMISSION)
                position_capital = initial_capital * TREND_POSITION_SIZE
                shares = position_capital / entry_price
                if shares * entry_price <= capital:
                    capital -= shares * entry_price
                    positions[symbol] = {
                        'shares': shares,
                        'entry_price': entry_price,
                        'entry_date': date
                    }
                    slots_available -= 1

        final_portfolio_value = capital
        for symbol, pos in positions.items():
            if pd.notna(prepared_data[symbol].loc[date, 'Close']):
                current_price = prepared_data[symbol].loc[date, 'Close']
                final_portfolio_value += pos['shares'] * current_price
        equity_curve.append(final_portfolio_value)
        daily_exposure.append(len(positions) / 5)

    if positions:
        for symbol, pos in list(positions.items()):
            last_valid_idx = prepared_data[symbol]['Close'].last_valid_index()
            if last_valid_idx is not None:
                exit_price = prepared_data[symbol].loc[last_valid_idx, 'Close'] * (1 - SLIPPAGE) * (1 - COMMISSION)
                shares = pos['shares']
                entry_price = pos['entry_price']
                pnl = (exit_price - entry_price) * shares
                trades.append({
                    'Strategy': 'Trend_Following',
                    'Symbol': symbol,
                    'Entry_Date': pos['entry_date'],
                    'Exit_Date': last_valid_idx,
                    'Entry_Price': entry_price,
                    'Exit_Price': exit_price,
                    'Shares': shares,
                    'PnL': pnl,
                    'Return_%': (exit_price / entry_price - 1) * 100
                })

    return pd.DataFrame(trades), pd.Series(equity_curve, index=all_dates), pd.Series(daily_exposure, index=all_dates)

# ============================================================================
# 4. PERFORMANCE METRICS
# ============================================================================
def calculate_performance_metrics(equity_series, trades_df, initial_capital, exposure_series):
    """Calculate comprehensive performance metrics"""
    
    returns = equity_series.pct_change().dropna()
    
    # Basic metrics
    total_return = (equity_series.iloc[-1] / initial_capital - 1) * 100
    cagr = (((equity_series.iloc[-1] / initial_capital) ** (252 / len(equity_series))) - 1) * 100
    
    # Trade metrics
    num_trades = len(trades_df)
    
    if num_trades > 0:
        winning_trades = trades_df[trades_df['Return_%'] > 0]
        losing_trades = trades_df[trades_df['Return_%'] < 0]
        
        win_rate = len(winning_trades) / num_trades * 100
        avg_win = winning_trades['Return_%'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['Return_%'].mean() if len(losing_trades) > 0 else 0
        
        gross_profit = winning_trades['PnL'].sum() if len(winning_trades) > 0 else 0
        gross_loss = abs(losing_trades['PnL'].sum()) if len(losing_trades) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else np.inf
    else:
        win_rate = 0
        avg_win = 0
        avg_loss = 0
        profit_factor = 0
    
    # Risk metrics
    volatility = returns.std() * np.sqrt(252) * 100 if len(returns) > 0 else 0
    sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if len(returns) > 0 and returns.std() != 0 else 0
    
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
    sortino = (returns.mean() * 252 / downside_std) if downside_std != 0 else 0
    
    # Drawdown
    cumulative = equity_series / equity_series.expanding().max()
    drawdown = (cumulative - 1) * 100
    max_drawdown = drawdown.min()
    
    mar = abs(cagr / max_drawdown) if max_drawdown != 0 else 0
    exposure_pct = exposure_series.mean() * 100 if len(exposure_series) > 0 else 0
    calmar = abs(cagr / max_drawdown) if max_drawdown != 0 else 0
    
    metrics = {
        'Total_Return_%': total_return,
        'CAGR_%': cagr,
        'Num_Trades': num_trades,
        'Win_Rate_%': win_rate,
        'Avg_Win_%': avg_win,
        'Avg_Loss_%': avg_loss,
        'Profit_Factor': profit_factor,
        'Sharpe_Ratio': sharpe,
        'Sortino_Ratio': sortino,
        'Max_Drawdown_%': max_drawdown,
        'MAR_Ratio': mar,
        'Calmar_Ratio': calmar,
        'Volatility_%': volatility,
        'Avg_Exposure_%': exposure_pct,
        'Final_Equity': equity_series.iloc[-1]
    }
    
    return metrics

# ============================================================================
# 5. MONTE CARLO SIMULATION
# ============================================================================
def monte_carlo_portfolio_simulation(equity_series, initial_capital, n_simulations=1000):
    """
    Monte Carlo that resamples daily portfolio returns derived from equity_series
    """
    daily_ret = equity_series.pct_change().dropna().values
    if len(daily_ret) == 0:
        return {}, []

    final_equities = []
    max_drawdowns = []
    sharpes = []

    for _ in range(n_simulations):
        sampled = np.random.choice(daily_ret, size=len(daily_ret), replace=True)
        equity = initial_capital
        series = [equity]
        for r in sampled:
            equity = equity * (1 + r)
            series.append(equity)
        series = np.array(series)
        final_equities.append(series[-1])
        # drawdown
        running_max = np.maximum.accumulate(series)
        dd = (series / running_max - 1) * 100
        max_drawdowns.append(dd.min())
        # sharpe
        rets = pd.Series(series).pct_change().dropna()
        if rets.std() > 0:
            sharpes.append((rets.mean() / rets.std()) * np.sqrt(252))

    mc_results = {
        'Mean_Final_Equity': float(np.mean(final_equities)),
        'Median_Final_Equity': float(np.median(final_equities)),
        'Std_Final_Equity': float(np.std(final_equities)),
        'Min_Final_Equity': float(np.min(final_equities)),
        'Max_Final_Equity': float(np.max(final_equities)),
        'Mean_Max_DD_%': float(np.mean(max_drawdowns)),
        'Worst_Max_DD_%': float(np.min(max_drawdowns)),
        'Best_Max_DD_%': float(np.max(max_drawdowns)),
        '5th_Percentile': float(np.percentile(final_equities, 5)),
        '95th_Percentile': float(np.percentile(final_equities, 95)),
        'Mean_Sharpe': float(np.mean(sharpes)) if sharpes else 0.0
    }
    return mc_results, final_equities

# ============================================================================
# 6. RUN BACKTESTS
# ============================================================================
print("Loading Binance data...")
crypto_data = load_binance_data()

if not crypto_data or len(crypto_data) < 2:
    print(f"\nError: Need at least 2 valid CSV files. Found {len(crypto_data) if crypto_data else 0}")
    print("Please ensure CSV files are in 'all_binance' folder with columns: time, open, high, low, close, volume")
    exit()

print("\n" + "="*80)
print("RUNNING BACKTESTS")
print("="*80)

# Mean Reversion (IBS-based)
print("\n1. Backtesting Mean Reversion Strategy (IBS with Volatility Bands)...")
mr_trades, mr_equity, mr_exposure = backtest_mean_reversion_portfolio(crypto_data, INITIAL_CAPITAL, 5)
mr_metrics = calculate_performance_metrics(mr_equity, mr_trades, INITIAL_CAPITAL, mr_exposure)

print(f"\nMean Reversion Quick Stats:")
print(f"  - Total Trades: {len(mr_trades)}")
print(f"  - Unique Symbols Traded: {mr_trades['Symbol'].nunique() if len(mr_trades) > 0 else 0}")
if len(mr_trades) > 0:
    print(f"  - Average Holding Days: {mr_trades['Holding_Days'].mean():.1f}")
    print(f"  - Exit Reasons:")
    if 'Exit_Reason' in mr_trades.columns:
        for reason, count in mr_trades['Exit_Reason'].value_counts().items():
            print(f"    * {reason}: {count} trades ({count/len(mr_trades)*100:.1f}%)")

# Trend Following
print("\n2. Backtesting Trend Following Strategy...")
trend_trades, trend_equity, trend_exposure = backtest_trend_portfolio(crypto_data, INITIAL_CAPITAL)
trend_metrics = calculate_performance_metrics(trend_equity, trend_trades, INITIAL_CAPITAL, trend_exposure)

print(f"\nTrend Following Quick Stats:")
print(f"  - Total Trades: {len(trend_trades)}")
print(f"  - Symbols Traded: {trend_trades['Symbol'].unique().tolist() if len(trend_trades) > 0 else []}")

# Monte Carlo
print("\n3. Running Monte Carlo Simulations...")
mc_mr, mc_mr_dist = monte_carlo_portfolio_simulation(mr_equity, INITIAL_CAPITAL, MONTE_CARLO_SIMS)
mc_trend, mc_trend_dist = monte_carlo_portfolio_simulation(trend_equity, INITIAL_CAPITAL, MONTE_CARLO_SIMS)

# ============================================================================
# 7. PRINT RESULTS
# ============================================================================
def print_metrics(name, metrics):
    print(f"\n{'='*80}")
    print(f"{name}")
    print(f"{'='*80}")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key:.<40} {value:>15,.2f}")
        else:
            print(f"{key:.<40} {value:>15}")

print_metrics("MEAN REVERSION (IBS + 2×ATR Stop)", mr_metrics)
print_metrics("TREND FOLLOWING", trend_metrics)

if mc_mr:
    print_metrics("MONTE CARLO - MEAN REVERSION", mc_mr)
if mc_trend:
    print_metrics("MONTE CARLO - TREND FOLLOWING", mc_trend)

# ============================================================================
# 8. VISUALIZATIONS
# ============================================================================
print("\n4. Creating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Equity Curves
ax1 = axes[0, 0]
ax1.plot(mr_equity.index, mr_equity.values, label='Mean Reversion', linewidth=2)
ax1.plot(trend_equity.index, trend_equity.values, label='Trend Following', linewidth=2)
ax1.set_title('Equity Curves Comparison', fontsize=14, fontweight='bold')
ax1.set_xlabel('Date')
ax1.set_ylabel('Equity ($)')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))

# Plot 2: Drawdown
ax2 = axes[0, 1]
mr_dd = (mr_equity / mr_equity.expanding().max() - 1) * 100
trend_dd = (trend_equity / trend_equity.expanding().max() - 1) * 100

ax2.fill_between(mr_equity.index, mr_dd.values, 0, label='Mean Reversion', alpha=0.5)
ax2.fill_between(trend_equity.index, trend_dd.values, 0, label='Trend Following', alpha=0.5)
ax2.set_title('Drawdown Comparison', fontsize=14, fontweight='bold')
ax2.set_xlabel('Date')
ax2.set_ylabel('Drawdown (%)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Monte Carlo - Mean Reversion
ax3 = axes[1, 0]
if mc_mr_dist:
    ax3.hist(mc_mr_dist, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax3.axvline(INITIAL_CAPITAL, color='red', linestyle='--', linewidth=2, label='Initial')
    ax3.axvline(mr_equity.iloc[-1], color='green', linestyle='--', linewidth=2, label='Actual')
    ax3.set_title('Monte Carlo - Mean Reversion', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Final Equity ($)')
    ax3.set_ylabel('Frequency')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

# Plot 4: Monte Carlo - Trend
ax4 = axes[1, 1]
if mc_trend_dist:
    ax4.hist(mc_trend_dist, bins=50, alpha=0.7, color='green', edgecolor='black')
    ax4.axvline(INITIAL_CAPITAL, color='red', linestyle='--', linewidth=2, label='Initial')
    ax4.axvline(trend_equity.iloc[-1], color='blue', linestyle='--', linewidth=2, label='Actual')
    ax4.set_title('Monte Carlo - Trend Following', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Final Equity ($)')
    ax4.set_ylabel('Frequency')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

plt.tight_layout()
chart_path = os.path.join(CHARTS_FOLDER, 'equity_curves.png')
plt.savefig(chart_path, dpi=300, bbox_inches='tight')
print(f"   - Saved {chart_path}")
plt.close()

# ============================================================================
# 9. SAVE TO EXCEL
# ============================================================================
print("\n5. Saving results to Excel...")

with pd.ExcelWriter('Crypto_Backtest_Results.xlsx', engine='openpyxl') as writer:
    # Summary
    summary = pd.DataFrame({
        'Mean_Reversion': mr_metrics,
        'Trend_Following': trend_metrics
    }).T
    summary.to_excel(writer, sheet_name='Summary_Metrics')
    
    # Trades
    mr_trades.to_excel(writer, sheet_name='MR_Trades', index=False)
    trend_trades.to_excel(writer, sheet_name='Trend_Trades', index=False)
    
    # Equity Curves
    equity_df = pd.DataFrame({
        'Date': mr_equity.index,
        'MR_Equity': mr_equity.values,
        'MR_Exposure': mr_exposure.values,
        'Trend_Equity': trend_equity.reindex(mr_equity.index, method='ffill').values,
        'Trend_Exposure': trend_exposure.reindex(mr_equity.index, method='ffill').values
    })
    equity_df.to_excel(writer, sheet_name='Equity_Curves', index=False)
    
    # Monte Carlo
    if mc_mr:
        pd.DataFrame([mc_mr]).to_excel(writer, sheet_name='MC_MeanReversion', index=False)
        pd.DataFrame({'Final_Equity': mc_mr_dist}).to_excel(writer, sheet_name='MC_MR_Distribution', index=False)
    if mc_trend:
        pd.DataFrame([mc_trend]).to_excel(writer, sheet_name='MC_Trend', index=False)
        pd.DataFrame({'Final_Equity': mc_trend_dist}).to_excel(writer, sheet_name='MC_Trend_Distribution', index=False)

print("   - Saved Crypto_Backtest_Results.xlsx")

print("\n" + "="*80)
print("BACKTEST COMPLETE!")
print("="*80)
print(f"\nMean Reversion Strategy Details (IBS with Volatility Bands):")
print(f"  - Symbols: BTC, ETH, DOGE, XRP, SOL (same as Trend Following)")
print(f"  - Entry: Close < Lower Band AND IBS < 0.3")
print(f"    * Lower Band = 10-day Rolling High - 2.5 × 25-day Mean(High-Low)")
print(f"    * IBS = (Close - Low) / (High - Low)")
print(f"  - Exit: 2 × ATR(10) stop loss OR 10-day time stop")
print(f"  - Max Positions: 5")
print(f"  - Commission: {COMMISSION*100}% (taker fee)")
print(f"  - Slippage: {SLIPPAGE*100}%")
print(f"\nTrend Following Strategy Details:")
print(f"  - Symbols: BTC, ETH, DOGE, XRP, SOL")
print(f"  - Entry: Close > 50 DMA")
print(f"  - Exit: Close < 50 DMA")
print(f"  - Max Positions: 5 (20% each)")
print(f"\nTotal symbols loaded: {len(crypto_data)}")
print(f"\nFiles created:")
print(f"  1. Crypto_Backtest_Results.xlsx")
print(f"  2. {CHARTS_FOLDER}/equity_curves.png")
print("="*80)