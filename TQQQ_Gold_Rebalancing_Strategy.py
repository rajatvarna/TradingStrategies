"""
Leveraged Rebalancing Strategy with Crash Protection & Gold Regime Filter
==========================================================================
Strategy: 50/50 TQQQ/UGL with regime filter and crash protection

Gold Regime Filter (based on GLD 200 MA with 5% hysteresis band):
- Uses GLD (non-leveraged) for cleaner regime signals
- Exit UGL → IEF when GLD drops below 200 MA * 0.95
- Re-enter UGL when GLD rises above 200 MA * 1.05
- Neutral zone (0.95 to 1.05 MA) reduces whipsaw trades

Positions based on regime:
- Gold Bullish: 50% TQQQ / 50% UGL
- Gold Bearish: 50% TQQQ / 50% IEF

Crash Protection:
- TQQQ 20% single-day drop → Full exit to 100% IEF
- Recovery above pre-crash price → Return to regime-based allocation

Author: Quant Developer
Date: October 2025
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.styles import PatternFill, Font, Alignment
import scipy.stats as stats
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

INITIAL_CAPITAL = 100000
TARGET_TQQQ_WEIGHT = 0.50
TARGET_UGL_WEIGHT = 0.50
REBALANCE_MONTHS = 2
REBALANCE_BUFFER = 0.10  # 10% buffer (rebalance only if outside 40-60%)
CRASH_THRESHOLD = -0.20  # 20% drop in a single day
COMMISSION_PER_SHARE = 0.005
SLIPPAGE_PCT = 0.002  # 0.2%
MONTE_CARLO_RUNS = 10000
REGIME_MA_PERIOD = 200
REGIME_BUFFER = 0.05

# Tickers
TQQQ = 'TQQQ'
UGL = 'UGL'
IEF = 'IEF'
BENCHMARKS = ['SPY', 'QQQ', 'BTC-USD', 'GLD']

# Create output folder
output_folder = 'backtest_results'
os.makedirs(output_folder, exist_ok=True)

print("=" * 80)
print("LEVERAGED REBALANCING STRATEGY BACKTEST")
print("=" * 80)
print(f"\nConfiguration:")
print(f"  Initial Capital: ${INITIAL_CAPITAL:,.0f}")
print(f"  Target Allocation: {TARGET_TQQQ_WEIGHT*100:.0f}% TQQQ / {TARGET_UGL_WEIGHT*100:.0f}% UGL")
print(f"  Rebalance Frequency: Every {REBALANCE_MONTHS} months")
print(f"  Rebalance Buffer: ±{REBALANCE_BUFFER*100:.0f}%")
print(f"  Crash Filter: Exit to IEF if TQQQ drops {CRASH_THRESHOLD*100:.0f}% in one day")
print(f"  Transaction Costs: ${COMMISSION_PER_SHARE}/share + {SLIPPAGE_PCT*100:.1f}% slippage")
print(f"  Monte Carlo Runs: {MONTE_CARLO_RUNS:,}")
print(f"  Output Folder: {output_folder}/")

# ============================================================================
# DATA DOWNLOAD
# ============================================================================

print("\n" + "=" * 80)
print("DOWNLOADING DATA FROM YAHOO FINANCE")
print("=" * 80)

def download_data(tickers, start_date='2010-01-01'):
    """Download historical data for multiple tickers"""
    data = {}
    for ticker in tickers:
        try:
            print(f"Downloading {ticker}...", end=' ')
            df = yf.download(ticker, start=start_date, progress=False, auto_adjust=True)
            
            if df is not None and len(df) > 0:
                # Get Close prices - handle different yfinance return formats
                if isinstance(df, pd.Series):
                    price_data = df
                elif 'Close' in df.columns:
                    price_data = df['Close']
                    # If Close is still a DataFrame, get the first column
                    if isinstance(price_data, pd.DataFrame):
                        price_data = price_data.iloc[:, 0]
                else:
                    # Get first column if no Close
                    price_data = df.iloc[:, 0]
                
                # At this point price_data should be a Series, but let's be sure
                # Extract the actual numpy array and create a clean Series
                values = np.array(price_data).flatten()
                index = df.index
                
                data[ticker] = pd.Series(values, index=index, name=ticker)
                    
                print(f"✓ ({len(data[ticker])} rows, {data[ticker].index[0].date()} to {data[ticker].index[-1].date()})")
            else:
                print(f"✗ No data")
        except Exception as e:
            print(f"✗ Error: {e}")
    
    if len(data) == 0:
        raise ValueError("No data downloaded for any ticker. Please check your internet connection and ticker symbols.")
    
    # Create DataFrame from dictionary of Series
    df_combined = pd.DataFrame(data)
    
    # Forward fill for assets that start later (like BTC-USD)
    df_combined = df_combined.ffill().bfill()
    
    # Drop rows where primary assets (TQQQ, UGL, IEF) are missing
    df_combined = df_combined.dropna(subset=[tickers[0], tickers[1], tickers[2]], how='any')
    
    return df_combined

# Download all data
all_tickers = [TQQQ, UGL, IEF] + BENCHMARKS
df_prices = download_data(all_tickers)

print(f"\nData Summary:")
print(f"  Date Range: {df_prices.index[0].date()} to {df_prices.index[-1].date()}")
print(f"  Trading Days: {len(df_prices)}")

# Calculate 200-day moving average for GLD (Gold regime filter)
# Use GLD (non-leveraged) to avoid volatility decay in UGL
df_prices['GLD_MA200'] = df_prices['GLD'].rolling(window=REGIME_MA_PERIOD).mean()

print(f"\n  GLD 200-day MA calculated for regime filter (controls UGL positions)")

# ============================================================================
# STRATEGY IMPLEMENTATION
# ============================================================================

print("\n" + "=" * 80)
print("RUNNING STRATEGY BACKTEST")
print("=" * 80)

class PortfolioBacktest:
    def __init__(self, prices, initial_capital):
        self.prices = prices
        self.capital = initial_capital
        self.cash = initial_capital
        self.positions = {TQQQ: 0, UGL: 0, IEF: 0}  # shares held
        self.in_crash_mode = False
        self.gold_bearish = False  # Track if gold is below 200 DMA
        self.crash_recovery_price = None
        self.last_rebalance_date = None
        self.trades = []
        self.portfolio_values = []
        self.dates = []
        self.regime_changes = []  # Track regime changes for analysis
        
    def calculate_transaction_cost(self, shares, price):
        """Calculate commission + slippage"""
        commission = abs(shares) * COMMISSION_PER_SHARE
        slippage = abs(shares) * price * SLIPPAGE_PCT
        return commission + slippage
    
    def execute_trade(self, date, ticker, shares, price, reason):
        """Execute a trade with transaction costs"""
        if shares == 0:
            return
        
        cost = self.calculate_transaction_cost(shares, price)
        trade_value = shares * price
        
        # Update positions
        self.positions[ticker] += shares
        self.cash -= trade_value + cost
        
        # Record trade
        self.trades.append({
            'Date': date,
            'Ticker': ticker,
            'Shares': shares,
            'Price': price,
            'Value': trade_value,
            'Cost': cost,
            'Reason': reason
        })
    
    def get_portfolio_value(self, date):
        """Calculate current portfolio value"""
        holdings_value = sum(
            self.positions[ticker] * self.prices.loc[date, ticker]
            for ticker in self.positions
        )
        return self.cash + holdings_value
    
    def check_gold_regime(self, date):
        """Check if GLD is above or below its 200-day MA with 5% buffer to reduce whipsaws
        
        Uses GLD (non-leveraged) instead of UGL for cleaner regime signals
        """
        if 'GLD_MA200' not in self.prices.columns:
            return True  # Default to bullish if MA not available
        
        gld_price = self.prices.loc[date, 'GLD']
        gld_ma = self.prices.loc[date, 'GLD_MA200']
        
        # Check if we have valid MA (not NaN)
        if pd.isna(gld_ma):
            return True  # Default to bullish if MA not ready
        
        was_bearish = self.gold_bearish
        
        # Hysteresis logic with 5% buffer to reduce whipsaws
        # If currently bearish: need price > MA * 1.05 to turn bullish
        # If currently bullish: need price < MA * 0.95 to turn bearish
        
        if self.gold_bearish:
            # Currently bearish - need strong signal to turn bullish
            if gld_price > gld_ma * (1 + REGIME_BUFFER):
                self.gold_bearish = False
        else:
            # Currently bullish - need strong signal to turn bearish
            if gld_price < gld_ma * (1 - REGIME_BUFFER):
                self.gold_bearish = True
        
        # Log regime change
        if was_bearish != self.gold_bearish:
            regime = "BEARISH (GLD < 200 MA * 0.95)" if self.gold_bearish else "BULLISH (GLD > 200 MA * 1.05)"
            self.regime_changes.append({
                'Date': date,
                'Regime': regime,
                'GLD_Price': gld_price,
                'GLD_MA200': gld_ma,
                'Lower_Band': gld_ma * (1 - REGIME_BUFFER),
                'Upper_Band': gld_ma * (1 + REGIME_BUFFER)
            })
        
        return not self.gold_bearish  # Return True if bullish
    
    def check_crash(self, date):
        """Check if TQQQ crashed 20% in a single day"""
        if date == self.prices.index[0]:
            return False
        
        prev_date = self.prices.index[self.prices.index < date][-1]
        prev_close = self.prices.loc[prev_date, TQQQ]
        curr_close = self.prices.loc[date, TQQQ]
        daily_return = (curr_close / prev_close) - 1
        
        if daily_return <= CRASH_THRESHOLD:
            self.crash_recovery_price = prev_close
            return True
        return False
    
    def check_recovery(self, date):
        """Check if TQQQ has recovered above pre-crash price"""
        curr_price = self.prices.loc[date, TQQQ]
        return curr_price > self.crash_recovery_price
    
    def should_rebalance(self, date):
        """Check if it's time to rebalance (every 2 months on fixed schedule)"""
        if self.last_rebalance_date is None:
            return True
        
        # Check if 2 months have passed
        months_diff = (date.year - self.last_rebalance_date.year) * 12 + \
                      (date.month - self.last_rebalance_date.month)
        
        return months_diff >= REBALANCE_MONTHS
    
    def rebalance_to_tqqq_ugl(self, date):
        """Rebalance to 50/50 TQQQ/UGL or 50/50 TQQQ/IEF based on gold regime"""
        port_value = self.get_portfolio_value(date)
        target_tqqq_value = port_value * TARGET_TQQQ_WEIGHT
        
        # Check gold regime to determine second asset
        gold_bullish = self.check_gold_regime(date)
        
        if gold_bullish:
            # Normal regime: 50% TQQQ / 50% UGL
            target_ugl_value = port_value * TARGET_UGL_WEIGHT
            target_ief_value = 0
            second_asset = UGL
            second_asset_price = self.prices.loc[date, UGL]
            target_second_shares = target_ugl_value / second_asset_price
        else:
            # Gold bearish: 50% TQQQ / 50% IEF
            target_ugl_value = 0
            target_ief_value = port_value * TARGET_UGL_WEIGHT
            second_asset = IEF
            second_asset_price = self.prices.loc[date, IEF]
            target_second_shares = target_ief_value / second_asset_price
        
        # Calculate target shares
        tqqq_price = self.prices.loc[date, TQQQ]
        target_tqqq_shares = target_tqqq_value / tqqq_price
        
        # Exit the asset we're not using
        if gold_bullish and self.positions[IEF] > 0:
            # Switching from IEF to UGL
            ief_price = self.prices.loc[date, IEF]
            self.execute_trade(date, IEF, -self.positions[IEF], ief_price, 'Exit IEF (Gold Bullish)')
        elif not gold_bullish and self.positions[UGL] > 0:
            # Switching from UGL to IEF
            ugl_price = self.prices.loc[date, UGL]
            self.execute_trade(date, UGL, -self.positions[UGL], ugl_price, 'Exit UGL (Gold Bearish)')
        
        # Rebalance TQQQ
        tqqq_trade = target_tqqq_shares - self.positions[TQQQ]
        if abs(tqqq_trade) > 0.01:  # Minimum trade threshold
            self.execute_trade(date, TQQQ, tqqq_trade, tqqq_price, 'Rebalance')
        
        # Rebalance second asset (UGL or IEF)
        second_trade = target_second_shares - self.positions[second_asset]
        if abs(second_trade) > 0.01:
            reason = 'Rebalance (Gold Bullish)' if gold_bullish else 'Rebalance (Gold Bearish)'
            self.execute_trade(date, second_asset, second_trade, second_asset_price, reason)
        
        self.last_rebalance_date = date
    
    def exit_to_ief(self, date):
        """Exit all positions and move to IEF"""
        tqqq_price = self.prices.loc[date, TQQQ]
        ugl_price = self.prices.loc[date, UGL]
        ief_price = self.prices.loc[date, IEF]
        
        # Sell all TQQQ
        if self.positions[TQQQ] > 0:
            self.execute_trade(date, TQQQ, -self.positions[TQQQ], tqqq_price, 'Crash Exit')
        
        # Sell all UGL
        if self.positions[UGL] > 0:
            self.execute_trade(date, UGL, -self.positions[UGL], ugl_price, 'Crash Exit')
        
        # Buy IEF with all remaining cash (minus small buffer)
        available_cash = self.cash * 0.99  # Keep 1% as buffer
        ief_shares = available_cash / ief_price
        self.execute_trade(date, IEF, ief_shares, ief_price, 'Enter IEF')
        
        self.in_crash_mode = True
    
    def run(self):
        """Run the complete backtest"""
        print("\nInitializing strategy...")
        
        for i, date in enumerate(self.prices.index):
            # Skip first REGIME_MA_PERIOD days to ensure MA is ready
            if i < REGIME_MA_PERIOD:
                port_value = self.capital  # Keep initial capital
                self.portfolio_values.append(port_value)
                self.dates.append(date)
                continue
            
            # Initial purchase after MA is ready
            if i == REGIME_MA_PERIOD:
                self.rebalance_to_tqqq_ugl(date)
            
            # Check gold regime (also updates self.gold_bearish)
            was_bearish = self.gold_bearish
            gold_bullish = self.check_gold_regime(date)
            regime_changed = was_bearish != self.gold_bearish
            
            # Check for crash
            if not self.in_crash_mode and self.check_crash(date):
                print(f"\n⚠️  CRASH DETECTED on {date.date()}: TQQQ dropped {CRASH_THRESHOLD*100:.0f}%+")
                print(f"    Exiting to IEF. Recovery target: ${self.crash_recovery_price:.2f}")
                self.exit_to_ief(date)
            
            # Check for recovery from crash
            elif self.in_crash_mode and self.check_recovery(date):
                print(f"\n✓  RECOVERY on {date.date()}: TQQQ above ${self.crash_recovery_price:.2f}")
                regime_status = "bearish (GLD < 200 MA)" if self.gold_bearish else "bullish (GLD > 200 MA)"
                print(f"    Gold regime: {regime_status}")
                print(f"    Returning to appropriate allocation")
                self.in_crash_mode = False
                self.crash_recovery_price = None
                self.rebalance_to_tqqq_ugl(date)
            
            # Check for gold regime change (only when not in crash mode)
            elif not self.in_crash_mode and regime_changed:
                gld_ma = self.prices.loc[date, 'GLD_MA200']
                lower_band = gld_ma * (1 - REGIME_BUFFER)
                upper_band = gld_ma * (1 + REGIME_BUFFER)
                
                if self.gold_bearish:
                    print(f"\n📉 GOLD BEARISH on {date.date()}: GLD dropped below ${lower_band:.2f} (MA*0.95)")
                    print(f"    GLD: ${self.prices.loc[date, 'GLD']:.2f} | 200 MA: ${gld_ma:.2f}")
                    print(f"    Exiting UGL → IEF (50% TQQQ / 50% IEF)")
                else:
                    print(f"\n📈 GOLD BULLISH on {date.date()}: GLD rose above ${upper_band:.2f} (MA*1.05)")
                    print(f"    GLD: ${self.prices.loc[date, 'GLD']:.2f} | 200 MA: ${gld_ma:.2f}")
                    print(f"    Entering UGL (50% TQQQ / 50% UGL)")
                self.rebalance_to_tqqq_ugl(date)
            
            # Regular rebalancing (only when not in crash mode and no regime change)
            elif not self.in_crash_mode and not regime_changed and self.should_rebalance(date):
                self.rebalance_to_tqqq_ugl(date)
            
            # Record daily portfolio value
            port_value = self.get_portfolio_value(date)
            self.portfolio_values.append(port_value)
            self.dates.append(date)
            
            # Progress update
            if (i + 1) % 500 == 0:
                pct_complete = (i + 1) / len(self.prices) * 100
                regime = "Gold Bearish" if self.gold_bearish else "Gold Bullish"
                crash = " [IN CRASH MODE]" if self.in_crash_mode else ""
                print(f"  Progress: {pct_complete:.1f}% - {date.date()} - ${port_value:,.0f} - {regime}{crash}")
        
        # Create results DataFrame
        self.equity_curve = pd.DataFrame({
            'Date': self.dates,
            'Portfolio_Value': self.portfolio_values
        }).set_index('Date')
        
        self.trades_df = pd.DataFrame(self.trades)
        self.regime_df = pd.DataFrame(self.regime_changes)
        
        print(f"\n✓ Backtest complete!")
        print(f"  Total Trades: {len(self.trades)}")
        print(f"  Regime Changes: {len(self.regime_changes)}")
        print(f"  Final Portfolio Value: ${self.portfolio_values[-1]:,.2f}")
        
        return self.equity_curve, self.trades_df, self.regime_df

# Run backtest
backtest = PortfolioBacktest(df_prices, INITIAL_CAPITAL)
equity_curve, trades_df, regime_df = backtest.run()

# ============================================================================
# CALCULATE STATISTICS
# ============================================================================

print("\n" + "=" * 80)
print("CALCULATING PERFORMANCE STATISTICS")
print("=" * 80)

def calculate_statistics(equity_curve, trades_df, name="Strategy"):
    """Calculate comprehensive performance statistics"""
    
    # Returns
    equity_curve['Returns'] = equity_curve['Portfolio_Value'].pct_change()
    equity_curve['Cumulative_Returns'] = (1 + equity_curve['Returns']).cumprod() - 1
    
    # Total return
    total_return = (equity_curve['Portfolio_Value'].iloc[-1] / equity_curve['Portfolio_Value'].iloc[0]) - 1
    
    # Annualized return
    years = (equity_curve.index[-1] - equity_curve.index[0]).days / 365.25
    annualized_return = (1 + total_return) ** (1 / years) - 1
    
    # Volatility
    daily_vol = equity_curve['Returns'].std()
    annual_vol = daily_vol * np.sqrt(252)
    
    # Sharpe Ratio (assuming 0% risk-free rate)
    sharpe = annualized_return / annual_vol if annual_vol > 0 else 0
    
    # Sortino Ratio
    downside_returns = equity_curve['Returns'][equity_curve['Returns'] < 0]
    downside_vol = downside_returns.std() * np.sqrt(252)
    sortino = annualized_return / downside_vol if downside_vol > 0 else 0
    
    # Maximum Drawdown
    cumulative = (1 + equity_curve['Returns']).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # MAR Ratio (Return/MaxDD)
    mar = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    # Win Rate and Trade Statistics
    if len(trades_df) > 0:
        # Calculate trade P&L
        trade_pnl = []
        for ticker in trades_df['Ticker'].unique():
            ticker_trades = trades_df[trades_df['Ticker'] == ticker].copy()
            position = 0
            entry_price = 0
            
            for idx, trade in ticker_trades.iterrows():
                if position == 0 and trade['Shares'] > 0:
                    entry_price = trade['Price']
                    position = trade['Shares']
                elif position > 0 and trade['Shares'] < 0:
                    exit_price = trade['Price']
                    pnl = (exit_price - entry_price) / entry_price
                    trade_pnl.append(pnl)
                    position += trade['Shares']
                    if position == 0:
                        entry_price = 0
        
        if len(trade_pnl) > 0:
            trade_pnl = np.array(trade_pnl)
            wins = trade_pnl[trade_pnl > 0]
            losses = trade_pnl[trade_pnl < 0]
            
            win_rate = len(wins) / len(trade_pnl) if len(trade_pnl) > 0 else 0
            avg_win = wins.mean() if len(wins) > 0 else 0
            avg_loss = losses.mean() if len(losses) > 0 else 0
            profit_factor = abs(wins.sum() / losses.sum()) if len(losses) > 0 and losses.sum() != 0 else 0
        else:
            win_rate = avg_win = avg_loss = profit_factor = 0
    else:
        win_rate = avg_win = avg_loss = profit_factor = 0
    
    # Exposure (% of time in market)
    # For this strategy, we're always invested
    exposure = 1.0
    
    # Calmar Ratio
    calmar = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    stats = {
        'Total Return': total_return,
        'Annualized Return': annualized_return,
        'Annual Volatility': annual_vol,
        'Sharpe Ratio': sharpe,
        'Sortino Ratio': sortino,
        'Max Drawdown': max_drawdown,
        'MAR Ratio': mar,
        'Calmar Ratio': calmar,
        'Win Rate': win_rate,
        'Avg Win %': avg_win,
        'Avg Loss %': avg_loss,
        'Profit Factor': profit_factor,
        'Exposure %': exposure,
        'Total Trades': len(trades_df),
        'Final Value': equity_curve['Portfolio_Value'].iloc[-1]
    }
    
    return stats

# Calculate strategy statistics
strategy_stats = calculate_statistics(equity_curve, trades_df, "Strategy")

print("\nStrategy Performance:")
for key, value in strategy_stats.items():
    if 'Rate' in key or '%' in key or 'Exposure' in key:
        print(f"  {key}: {value*100:.2f}%")
    elif 'Ratio' in key or 'Factor' in key:
        print(f"  {key}: {value:.2f}")
    elif 'Trades' in key:
        print(f"  {key}: {int(value)}")
    elif 'Return' in key or 'Volatility' in key or 'Drawdown' in key:
        print(f"  {key}: {value*100:.2f}%")
    else:
        print(f"  {key}: ${value:,.2f}")

# ============================================================================
# BENCHMARK COMPARISONS
# ============================================================================

print("\n" + "=" * 80)
print("CALCULATING BENCHMARK COMPARISONS")
print("=" * 80)

benchmark_stats = {}

for benchmark in BENCHMARKS:
    if benchmark in df_prices.columns:
        print(f"\nCalculating {benchmark}...")
        
        # Create buy and hold equity curve
        bm_equity = pd.DataFrame(index=equity_curve.index)
        bm_prices = df_prices[benchmark].loc[equity_curve.index]
        bm_equity['Portfolio_Value'] = INITIAL_CAPITAL * (bm_prices / bm_prices.iloc[0])
        
        # Calculate statistics (no trades for buy & hold)
        bm_trades = pd.DataFrame()
        stats = calculate_statistics(bm_equity, bm_trades, benchmark)
        benchmark_stats[benchmark] = stats
        
        print(f"  {benchmark} Return: {stats['Annualized Return']*100:.2f}%")
        print(f"  {benchmark} Sharpe: {stats['Sharpe Ratio']:.2f}")
        print(f"  {benchmark} Max DD: {stats['Max Drawdown']*100:.2f}%")

# ============================================================================
# CREATE COMPARISON TABLE
# ============================================================================

print("\n" + "=" * 80)
print("PERFORMANCE COMPARISON TABLE")
print("=" * 80)

comparison_df = pd.DataFrame({
    'Strategy': strategy_stats,
    **{bm: benchmark_stats[bm] for bm in benchmark_stats}
})

print("\n" + comparison_df.to_string())

# ============================================================================
# MONTHLY RETURNS HEATMAP
# ============================================================================

print("\n" + "=" * 80)
print("GENERATING MONTHLY RETURNS")
print("=" * 80)

def calculate_monthly_returns(equity_curve):
    """Calculate monthly and yearly returns"""
    monthly = equity_curve['Portfolio_Value'].resample('M').last()
    monthly_returns = monthly.pct_change()
    
    # Create pivot table for heatmap
    monthly_returns_df = pd.DataFrame({
        'Year': monthly_returns.index.year,
        'Month': monthly_returns.index.month,
        'Return': monthly_returns.values
    })
    
    pivot = monthly_returns_df.pivot(index='Year', columns='Month', values='Return')
    pivot.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # Calculate yearly returns
    yearly = equity_curve['Portfolio_Value'].resample('Y').last()
    yearly_returns = yearly.pct_change()
    
    # Add annual returns as a column to the pivot table
    annual_returns_series = []
    for year in pivot.index:
        # Get portfolio value at start and end of year
        year_data = equity_curve[equity_curve.index.year == year]
        if len(year_data) > 0:
            year_return = (year_data['Portfolio_Value'].iloc[-1] / year_data['Portfolio_Value'].iloc[0]) - 1
            annual_returns_series.append(year_return)
        else:
            annual_returns_series.append(np.nan)
    
    pivot['Annual'] = annual_returns_series
    
    return pivot, yearly_returns

monthly_pivot, yearly_returns = calculate_monthly_returns(equity_curve)

print("\nMonthly Returns (%):")
print((monthly_pivot * 100).round(2))

print("\nYearly Returns (%):")
for year, ret in yearly_returns.items():
    if not np.isnan(ret):
        print(f"  {year.year}: {ret*100:.2f}%")

# ============================================================================
# MONTE CARLO SIMULATION
# ============================================================================

print("\n" + "=" * 80)
print("RUNNING MONTE CARLO SIMULATION")
print("=" * 80)
print(f"Simulating {MONTE_CARLO_RUNS:,} scenarios...")

def monte_carlo_simulation(returns, num_simulations=10000, num_days=252):
    """Run Monte Carlo simulation on returns"""
    
    # Calculate return statistics
    mean_return = returns.mean()
    std_return = returns.std()
    
    # Run simulations
    simulation_results = []
    
    for i in range(num_simulations):
        # Generate random returns
        simulated_returns = np.random.normal(mean_return, std_return, num_days)
        
        # Calculate cumulative return
        cumulative_return = (1 + simulated_returns).prod() - 1
        simulation_results.append(cumulative_return)
        
        if (i + 1) % 1000 == 0:
            print(f"  Progress: {(i+1)/num_simulations*100:.0f}%")
    
    simulation_results = np.array(simulation_results)
    
    # Calculate statistics
    mc_stats = {
        'Mean Return': simulation_results.mean(),
        'Median Return': np.median(simulation_results),
        'Std Dev': simulation_results.std(),
        '5th Percentile': np.percentile(simulation_results, 5),
        '25th Percentile': np.percentile(simulation_results, 25),
        '75th Percentile': np.percentile(simulation_results, 75),
        '95th Percentile': np.percentile(simulation_results, 95),
        'Probability of Profit': (simulation_results > 0).sum() / len(simulation_results),
        'Probability of Loss > 20%': (simulation_results < -0.20).sum() / len(simulation_results),
        'Probability of Gain > 50%': (simulation_results > 0.50).sum() / len(simulation_results),
    }
    
    return simulation_results, mc_stats

# Run Monte Carlo with 1-year forward projection
mc_results, mc_stats = monte_carlo_simulation(
    equity_curve['Returns'].dropna(),
    num_simulations=MONTE_CARLO_RUNS,
    num_days=252
)

print("\nMonte Carlo Results (1-Year Projection):")
for key, value in mc_stats.items():
    if 'Probability' in key:
        print(f"  {key}: {value*100:.2f}%")
    else:
        print(f"  {key}: {value*100:.2f}%")

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print("\n" + "=" * 80)
print("GENERATING VISUALIZATIONS")
print("=" * 80)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# 1. Equity Curve Comparison
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[3, 1])

# Plot strategy
ax1.plot(equity_curve.index, equity_curve['Portfolio_Value'], 
        label='Strategy', linewidth=2, color='#2E86DE')

# Plot benchmarks
colors = ['#EE5A6F', '#FD79A8', '#FDCB6E', '#6C5CE7']
for i, benchmark in enumerate(BENCHMARKS):
    if benchmark in df_prices.columns:
        bm_prices = df_prices[benchmark].loc[equity_curve.index]
        bm_equity = INITIAL_CAPITAL * (bm_prices / bm_prices.iloc[0])
        ax1.plot(equity_curve.index, bm_equity, 
                label=f'{benchmark} B&H', linewidth=1.5, alpha=0.7, color=colors[i])

# Highlight regime periods
if len(regime_df) > 0:
    bearish_periods = []
    for idx, row in regime_df.iterrows():
        if 'BEARISH' in row['Regime']:
            start_date = row['Date']
            # Find end date (next bullish signal or end of data)
            next_bullish = regime_df[(regime_df['Date'] > start_date) & (regime_df['Regime'].str.contains('BULLISH'))]
            if len(next_bullish) > 0:
                end_date = next_bullish.iloc[0]['Date']
            else:
                end_date = equity_curve.index[-1]
            bearish_periods.append((start_date, end_date))
    
    for start, end in bearish_periods:
        ax1.axvspan(start, end, alpha=0.1, color='red', label='Gold Bearish' if start == bearish_periods[0][0] else '')

ax1.set_title('Strategy vs Buy & Hold Benchmarks (with Gold Regime)', fontsize=16, fontweight='bold')
ax1.set_xlabel('Date', fontsize=12)
ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
ax1.legend(loc='upper left', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))

# Plot GLD vs 200 MA on second subplot (regime indicator)
ax2.plot(df_prices.index, df_prices['GLD'], label='GLD Price', linewidth=1.5, color='#F39C12')
ax2.plot(df_prices.index, df_prices['GLD_MA200'], label='GLD 200 MA', linewidth=1.5, color='#E74C3C', linestyle='--')

# Plot buffer bands
ax2.plot(df_prices.index, df_prices['GLD_MA200'] * (1 - REGIME_BUFFER), 
         label=f'Lower Band (MA*{1-REGIME_BUFFER:.2f})', linewidth=1, color='#C0392B', linestyle=':', alpha=0.7)
ax2.plot(df_prices.index, df_prices['GLD_MA200'] * (1 + REGIME_BUFFER), 
         label=f'Upper Band (MA*{1+REGIME_BUFFER:.2f})', linewidth=1, color='#27AE60', linestyle=':', alpha=0.7)

# Fill between bands to show neutral zone
ax2.fill_between(df_prices.index, 
                  df_prices['GLD_MA200'] * (1 - REGIME_BUFFER), 
                  df_prices['GLD_MA200'] * (1 + REGIME_BUFFER), 
                  alpha=0.1, color='gray', label='Neutral Zone')

ax2.set_xlabel('Date', fontsize=12)
ax2.set_ylabel('GLD Price ($)', fontsize=12)
ax2.set_title('Gold Regime Filter: GLD vs 200-Day MA (with 5% Hysteresis Band)', fontsize=14, fontweight='bold')
ax2.legend(loc='upper left', fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'equity_curve.png'), dpi=300, bbox_inches='tight')
print("  ✓ Saved: equity_curve.png")
plt.close()

# 2. Monthly Returns Heatmap
fig, ax = plt.subplots(figsize=(14, 8))

# Create custom colormap for better visualization
cmap = sns.diverging_palette(10, 130, as_cmap=True)

# Plot heatmap with annual returns column
sns.heatmap(monthly_pivot * 100, annot=True, fmt='.1f', cmap=cmap, 
            center=0, cbar_kws={'label': 'Return (%)'}, ax=ax,
            linewidths=0.5, linecolor='gray')

ax.set_title('Monthly Returns Heatmap (%) with Annual Returns', fontsize=16, fontweight='bold')
ax.set_xlabel('Month', fontsize=12)
ax.set_ylabel('Year', fontsize=12)

# Bold the Annual column by adjusting the plot
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'monthly_heatmap.png'), dpi=300, bbox_inches='tight')
print("  ✓ Saved: monthly_heatmap.png")
plt.close()

# 3. Drawdown Chart
fig, ax = plt.subplots(figsize=(14, 6))
cumulative = (1 + equity_curve['Returns']).cumprod()
running_max = cumulative.expanding().max()
drawdown = (cumulative - running_max) / running_max
ax.fill_between(drawdown.index, drawdown * 100, 0, alpha=0.3, color='red')
ax.plot(drawdown.index, drawdown * 100, color='darkred', linewidth=1)
ax.set_title('Strategy Drawdown Over Time', fontsize=16, fontweight='bold')
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Drawdown (%)', fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'drawdown.png'), dpi=300, bbox_inches='tight')
print("  ✓ Saved: drawdown.png")
plt.close()

# 4. Monte Carlo Distribution
fig, ax = plt.subplots(figsize=(12, 6))
ax.hist(mc_results * 100, bins=100, edgecolor='black', alpha=0.7, color='#0984e3')
ax.axvline(np.percentile(mc_results, 5) * 100, color='red', 
           linestyle='--', linewidth=2, label=f'5th %ile: {np.percentile(mc_results, 5)*100:.1f}%')
ax.axvline(np.percentile(mc_results, 50) * 100, color='green', 
           linestyle='--', linewidth=2, label=f'Median: {np.percentile(mc_results, 50)*100:.1f}%')
ax.axvline(np.percentile(mc_results, 95) * 100, color='blue', 
           linestyle='--', linewidth=2, label=f'95th %ile: {np.percentile(mc_results, 95)*100:.1f}%')
ax.set_title(f'Monte Carlo Simulation ({MONTE_CARLO_RUNS:,} runs, 1-Year Projection)', 
             fontsize=16, fontweight='bold')
ax.set_xlabel('Return (%)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'monte_carlo.png'), dpi=300, bbox_inches='tight')
print("  ✓ Saved: monte_carlo.png")
plt.close()

# ============================================================================
# EXPORT TO EXCEL
# ============================================================================

print("\n" + "=" * 80)
print("EXPORTING TO EXCEL")
print("=" * 80)

excel_filename = os.path.join(output_folder, 'backtest_results.xlsx')

with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
    # 1. Summary Statistics
    summary_df = pd.DataFrame({
        'Metric': list(strategy_stats.keys()),
        'Value': list(strategy_stats.values())
    })
    summary_df.to_excel(writer, sheet_name='Summary', index=False)
    
    # 2. Comparison Table
    comparison_df.T.to_excel(writer, sheet_name='Comparison')
    
    # 3. All Trades
    trades_df.to_excel(writer, sheet_name='Trades', index=False)
    
    # 4. Daily Equity Curve
    equity_export = equity_curve.copy()
    equity_export['Daily_Return_%'] = equity_export['Returns'] * 100
    equity_export.to_excel(writer, sheet_name='Equity_Curve')
    
    # 5. Monthly Returns
    monthly_pivot.to_excel(writer, sheet_name='Monthly_Returns')
    
    # 6. Yearly Returns
    yearly_df = pd.DataFrame({
        'Year': [d.year for d in yearly_returns.index],
        'Return_%': yearly_returns.values * 100
    })
    yearly_df.to_excel(writer, sheet_name='Yearly_Returns', index=False)
    
    # 7. Monte Carlo Results
    mc_df = pd.DataFrame({
        'Simulation': range(1, len(mc_results) + 1),
        'Return_%': mc_results * 100
    })
    mc_df.to_excel(writer, sheet_name='Monte_Carlo', index=False)
    
    # 8. Monte Carlo Statistics
    mc_stats_df = pd.DataFrame({
        'Metric': list(mc_stats.keys()),
        'Value': list(mc_stats.values())
    })
    mc_stats_df.to_excel(writer, sheet_name='MC_Statistics', index=False)
    
    # 9. Risk Metrics
    risk_metrics = {
        'Value at Risk (95%)': np.percentile(equity_curve['Returns'].dropna(), 5),
        'Conditional VaR (95%)': equity_curve['Returns'][equity_curve['Returns'] <= np.percentile(equity_curve['Returns'], 5)].mean(),
        'Skewness': equity_curve['Returns'].skew(),
        'Kurtosis': equity_curve['Returns'].kurtosis(),
        'Best Day': equity_curve['Returns'].max(),
        'Worst Day': equity_curve['Returns'].min(),
        'Positive Days %': (equity_curve['Returns'] > 0).sum() / len(equity_curve['Returns']) * 100,
    }
    risk_df = pd.DataFrame({
        'Metric': list(risk_metrics.keys()),
        'Value': list(risk_metrics.values())
    })
    risk_df.to_excel(writer, sheet_name='Risk_Metrics', index=False)
    
    # 10. Regime Changes
    if len(regime_df) > 0:
        regime_df.to_excel(writer, sheet_name='Regime_Changes', index=False)

print(f"  ✓ Saved: {excel_filename}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("BACKTEST COMPLETE!")
print("=" * 80)

print(f"\n📊 STRATEGY PERFORMANCE SUMMARY")
print(f"{'='*80}")
print(f"  Total Return:        {strategy_stats['Total Return']*100:>10.2f}%")
print(f"  Annualized Return:   {strategy_stats['Annualized Return']*100:>10.2f}%")
print(f"  Sharpe Ratio:        {strategy_stats['Sharpe Ratio']:>10.2f}")
print(f"  Sortino Ratio:       {strategy_stats['Sortino Ratio']:>10.2f}")
print(f"  MAR Ratio:           {strategy_stats['MAR Ratio']:>10.2f}")
print(f"  Max Drawdown:        {strategy_stats['Max Drawdown']*100:>10.2f}%")
print(f"  Win Rate:            {strategy_stats['Win Rate']*100:>10.2f}%")
print(f"  Profit Factor:       {strategy_stats['Profit Factor']:>10.2f}")
print(f"  Total Trades:        {strategy_stats['Total Trades']:>10.0f}")
print(f"  Final Value:         ${strategy_stats['Final Value']:>10,.2f}")

if len(regime_df) > 0:
    bearish_count = len(regime_df[regime_df['Regime'].str.contains('BEARISH')])
    bullish_count = len(regime_df[regime_df['Regime'].str.contains('BULLISH')])
    print(f"\n🔄 REGIME FILTER ACTIVITY:")
    print(f"  Total Regime Changes: {len(regime_df)}")
    print(f"  Gold Bearish Periods: {bearish_count}")
    print(f"  Gold Bullish Periods: {bullish_count}")

print(f"\n📁 FILES GENERATED IN: {output_folder}/")
print(f"  • backtest_results.xlsx")
print(f"  • equity_curve.png")
print(f"  • monthly_heatmap.png")
print(f"  • drawdown.png")
print(f"  • monte_carlo.png")

print(f"\n✅ All analysis complete!\n")