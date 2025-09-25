import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

"""
One Percent Per Week v3 - Python Translation from WealthLab
Original by Dion, Modified by Guy Fleury

Strategy Logic (Faithful Translation):
1. Buy TQQQ on Monday with limit order at Monday open × 1.001
2. Set initial profit target at 7% (entry price × 1.07)
3. If current profit > 30%, adjust target to entry × 1.08 (assuming 1.011 was typo)
4. Always place Market-on-Close sell order for Friday
5. Only trade once per week
6. Use limit orders for all entries and exits
"""

class OnePercentPerWeekStrategy:
    def __init__(self, initial_capital=100000, commission=0.005, slippage=0.001):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.cash = initial_capital
        self.position = 0
        self.trades = []
        self.equity_curve = []
        self.debug_log = []
        
        # Strategy variables (matching WealthLab)
        self.monday_open = np.nan
        self.traded_this_week = False
        self.entry_price = 0
        
    def log_debug(self, message):
        """Debug logging"""
        self.debug_log.append(f"Trade {len([t for t in self.trades if 'Trade_PnL' in t]):3d}: {message}")
        
    def download_data(self, symbol, start_date='2010-01-01'):
        """Download data from Yahoo Finance"""
        print(f"Downloading {symbol} data from {start_date}...")
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start_date, end=datetime.now().strftime('%Y-%m-%d'))
        data = data.round(4)
        data.index = data.index.tz_localize(None)
        return data
    
    def calculate_transaction_costs(self, shares, price):
        """Calculate total transaction costs"""
        commission_cost = abs(shares) * self.commission
        slippage_cost = abs(shares) * price * self.slippage
        return commission_cost + slippage_cost
    
    def execute_trade(self, date, price, shares, trade_type, trade_pnl=None):
        """Execute a trade with transaction costs"""
        if shares == 0:
            return None
            
        transaction_cost = self.calculate_transaction_costs(shares, price)
        trade_value = shares * price
        
        # Check cash availability for buys
        if shares > 0:
            total_cost = trade_value + transaction_cost
            if total_cost > self.cash:
                self.log_debug(f"INSUFFICIENT CASH: Need ${total_cost:.2f}, have ${self.cash:.2f}")
                return None
        
        # Execute trade
        old_cash = self.cash
        old_position = self.position
        
        self.position += shares
        self.cash -= (trade_value + transaction_cost)
        
        # Record trade
        trade_record = {
            'Date': date,
            'Type': trade_type,
            'Shares': shares,
            'Price': price,
            'Value': trade_value,
            'Transaction_Cost': transaction_cost,
            'Cash_Before': old_cash,
            'Cash_After': self.cash,
            'Position_Before': old_position,
            'Position_After': self.position
        }
        
        if trade_pnl is not None:
            trade_record['Trade_PnL'] = trade_pnl
            
        self.trades.append(trade_record)
        
        self.log_debug(f"{trade_type}: {shares} shares @ ${price:.2f} | Cash: ${old_cash:.0f}→${self.cash:.0f}")
        
        return trade_record
    
    def backtest(self, data):
        """Backtest with faithful WealthLab translation"""
        print("Running One Percent Per Week Strategy (WealthLab Translation)...")
        
        # Add day of week column
        data['DayOfWeek'] = data.index.day_name()
        data['Week'] = data.index.to_period('W-MON')
        
        trade_count = 0
        weeks_processed = 0
        
        for week_period, week_data in data.groupby('Week'):
            weeks_processed += 1
            
            # Reset weekly variables
            self.monday_open = np.nan
            self.traded_this_week = False
            
            # Get Monday data (start of week)
            monday_data = week_data[week_data['DayOfWeek'] == 'Monday']
            if len(monday_data) > 0:
                self.monday_open = monday_data['Open'].iloc[0]
            
            # Track max intraweek loss
            max_intraweek_loss = 0.0
            entry_date = None
            exit_date = None
            exit_price = None
            exit_type = None
            
            # Process each day of the week
            for current_date, row in week_data.sort_index().iterrows():
                day_of_week = row['DayOfWeek']
                open_price = row['Open']
                high_price = row['High']
                low_price = row['Low']
                close_price = row['Close']
                
                # Check if it's start of week (Monday)
                is_start_of_week = (day_of_week == 'Monday')
                is_last_day_of_week = (day_of_week == 'Friday')
                
                # If we have an open position
                if self.position > 0:
                    # Track max intraweek loss
                    current_loss_pct = (low_price - self.entry_price) / self.entry_price * 100
                    if current_loss_pct < max_intraweek_loss:
                        max_intraweek_loss = current_loss_pct
                    
                    # Calculate current profit
                    current_profit_pct = (close_price - self.entry_price) / self.entry_price
                    
                    # Set profit target (faithful to WealthLab logic)
                    target_price = self.entry_price
                    if current_profit_pct > 0.0:
                        target_price = self.entry_price * 1.07  # 7% target
                        if current_profit_pct > 0.3:  # If profit > 30% (seems high, but faithful to original)
                            target_price = self.entry_price * 1.08  # Assuming 1.011 was typo for 1.08
                    
                    # Check if limit order target is hit during the day
                    if high_price >= target_price and exit_date is None:
                        exit_date = current_date
                        exit_price = target_price
                        exit_type = 'SELL_LIMIT_8%' if current_profit_pct > 0.3 else 'SELL_LIMIT_7%'
                    
                    # Friday Market-on-Close (always placed as per WealthLab code)
                    if is_last_day_of_week and exit_date is None:
                        exit_date = current_date
                        exit_price = close_price
                        exit_type = 'SELL_FRIDAY_MOC'
                
                else:  # No position - look to enter
                    if is_start_of_week and not np.isnan(self.monday_open) and not self.traded_this_week:
                        # Buy with limit order at Monday open * 1.001 (as per WealthLab)
                        entry_limit_price = self.monday_open * 1.001
                        
                        # Check if limit order would be filled (open <= limit price)
                        if open_price <= entry_limit_price:
                            # Calculate position size
                            available_cash = self.cash * 0.95  # Use 95% of cash
                            shares_to_buy = int(available_cash / (entry_limit_price * (1 + self.slippage)))
                            
                            if shares_to_buy > 0:
                                # Execute buy
                                buy_trade = self.execute_trade(current_date, entry_limit_price, shares_to_buy, 'BUY_MONDAY_LIMIT')
                                if buy_trade:
                                    self.entry_price = entry_limit_price
                                    self.traded_this_week = True
                                    entry_date = current_date
                                    
                                    # If it's also last day of week, prepare Friday exit
                                    if is_last_day_of_week:
                                        exit_date = current_date
                                        exit_price = close_price
                                        exit_type = 'SELL_FRIDAY_MOC'
            
            # Execute exit trade if we have a position
            if self.position > 0 and exit_date is not None:
                shares_to_sell = self.position
                trade_pnl = (exit_price - self.entry_price) * shares_to_sell
                trade_return_pct = (exit_price - self.entry_price) / self.entry_price * 100
                
                # Execute sell
                sell_trade = self.execute_trade(exit_date, exit_price, -shares_to_sell, exit_type, trade_pnl)
                
                if sell_trade:
                    # Add metrics
                    sell_trade['Max_Intraweek_Loss_Pct'] = round(max_intraweek_loss, 2)
                    sell_trade['Entry_Price'] = self.entry_price
                    sell_trade['Exit_Price'] = exit_price
                    sell_trade['Trade_Return_Pct'] = round(trade_return_pct, 2)
                    sell_trade['Shares_Traded'] = shares_to_sell
                    sell_trade['Entry_Date'] = entry_date
                    
                    trade_count += 1
                    
                    # Record equity
                    self.equity_curve.append({
                        'Date': exit_date,
                        'Equity': self.cash,
                        'Cash': self.cash,
                        'Position': 0,
                        'Max_Intraweek_Loss_Pct': max_intraweek_loss,
                        'Trade_Return_Pct': trade_return_pct
                    })
        
        print(f"Backtest completed:")
        print(f"  Weeks processed: {weeks_processed}")
        print(f"  Successful trades: {trade_count}")
        print(f"  Final cash: ${self.cash:.2f}")
        
        # Show sample debug messages
        if len(self.debug_log) > 0:
            print(f"\nFirst 10 debug messages:")
            for msg in self.debug_log[:10]:
                print(f"  {msg}")
            if len(self.debug_log) > 10:
                print(f"  ... and {len(self.debug_log)-10} more")
        
        return self.trades, self.equity_curve

def calculate_performance_metrics(equity_curve, trades):
    """Calculate comprehensive performance metrics"""
    if len(equity_curve) < 2:
        return {'Error': 'Insufficient data'}
    
    df = pd.DataFrame(equity_curve)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date').sort_index()
    
    # Calculate returns
    df['Returns'] = df['Equity'].pct_change().fillna(0)
    
    # Basic metrics
    initial_capital = df['Equity'].iloc[0]
    final_capital = df['Equity'].iloc[-1]
    total_return = (final_capital / initial_capital - 1) * 100
    
    years = max((df.index[-1] - df.index[0]).days / 365.25, 1/252)
    cagr = ((final_capital / initial_capital) ** (1/years) - 1) * 100 if final_capital > 0 else -100
    
    daily_returns = df['Returns'].dropna()
    volatility = daily_returns.std() * np.sqrt(252) * 100 if len(daily_returns) > 1 else 0
    
    # Risk-adjusted metrics
    risk_free_rate = 0.02
    sharpe_ratio = (cagr/100 - risk_free_rate) / (volatility/100) if volatility > 0 else 0
    
    negative_returns = daily_returns[daily_returns < 0]
    if len(negative_returns) > 0:
        downside_volatility = negative_returns.std() * np.sqrt(252) * 100
        sortino_ratio = (cagr/100 - risk_free_rate) / (downside_volatility/100) if downside_volatility > 0 else 0
    else:
        sortino_ratio = sharpe_ratio
    
    # Drawdown
    cumulative = (1 + daily_returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdown = (cumulative / rolling_max - 1) * 100
    max_drawdown = drawdown.min()
    
    mar_ratio = abs(cagr / max_drawdown) if max_drawdown != 0 else 0
    
    # Trade metrics
    trades_with_pnl = [t for t in trades if 'Trade_PnL' in t]
    total_trades = len(trades_with_pnl)
    
    max_losses = [t.get('Max_Intraweek_Loss_Pct', 0) for t in trades_with_pnl]
    avg_max_intraweek_loss = np.mean(max_losses) if max_losses else 0
    worst_max_intraweek_loss = min(max_losses) if max_losses else 0
    
    if total_trades > 0:
        winning_trades = [t for t in trades_with_pnl if t['Trade_PnL'] > 0]
        losing_trades = [t for t in trades_with_pnl if t['Trade_PnL'] <= 0]
        
        win_count = len(winning_trades)
        win_rate = (win_count / total_trades * 100)
        
        avg_win = np.mean([t['Trade_PnL'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['Trade_PnL'] for t in losing_trades]) if losing_trades else 0
        
        avg_win_pct = (avg_win / initial_capital * 100) if avg_win else 0
        avg_loss_pct = (avg_loss / initial_capital * 100) if avg_loss else 0
        
        total_profit = sum([t['Trade_PnL'] for t in winning_trades]) if winning_trades else 0
        total_loss = abs(sum([t['Trade_PnL'] for t in losing_trades])) if losing_trades else 1
        profit_factor = total_profit / total_loss if total_loss > 0 else 0
    else:
        win_rate = avg_win_pct = avg_loss_pct = profit_factor = 0
        win_count = 0
    
    return {
        'Total Return (%)': round(total_return, 2),
        'CAGR (%)': round(cagr, 2),
        'Volatility (%)': round(volatility, 2),
        'Sharpe Ratio': round(sharpe_ratio, 3),
        'Sortino Ratio': round(sortino_ratio, 3),
        'Max Drawdown (%)': round(max_drawdown, 2),
        'MAR Ratio': round(mar_ratio, 3),
        'Win Rate (%)': round(win_rate, 2),
        'Avg Win (%)': round(avg_win_pct, 2),
        'Avg Loss (%)': round(avg_loss_pct, 2),
        'Profit Factor': round(profit_factor, 3),
        'Avg Max Intraweek Loss (%)': round(avg_max_intraweek_loss, 2),
        'Worst Max Intraweek Loss (%)': round(worst_max_intraweek_loss, 2),
        'Exposure (%)': 100.0,
        'Total Trades': total_trades,
        'Winning Trades': win_count,
        'Losing Trades': total_trades - win_count
    }

def create_monthly_heatmap(equity_curve):
    """Create monthly returns heatmap with yearly totals"""
    if len(equity_curve) < 2:
        return plt.figure()
        
    df = pd.DataFrame(equity_curve)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date').sort_index()
    
    df['Returns'] = df['Equity'].pct_change().fillna(0)
    monthly_returns = df['Returns'].resample('M').apply(lambda x: (1 + x).prod() - 1 if len(x) > 0 else 0) * 100
    
    if len(monthly_returns) == 0:
        return plt.figure()
    
    # Create pivot table
    monthly_data = monthly_returns.reset_index()
    monthly_data['Year'] = monthly_data['Date'].dt.year
    monthly_data['Month'] = monthly_data['Date'].dt.month
    heatmap_data = monthly_data.pivot(index='Year', columns='Month', values='Returns')
    
    # Add yearly totals
    yearly_returns = df['Returns'].resample('Y').apply(lambda x: (1 + x).prod() - 1 if len(x) > 0 else 0) * 100
    yearly_data = yearly_returns.reset_index()
    yearly_data['Year'] = yearly_data['Date'].dt.year
    yearly_dict = dict(zip(yearly_data['Year'], yearly_data['Returns']))
    
    yearly_column = [yearly_dict.get(year, 0) for year in heatmap_data.index]
    heatmap_data['Year Total'] = yearly_column
    
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Year Total']
    
    plt.figure(figsize=(14, 8))
    if not heatmap_data.empty:
        sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
                    xticklabels=month_labels[:heatmap_data.shape[1]], 
                    cbar_kws={'label': 'Returns (%)'})
    plt.title('Monthly and Yearly Returns Heatmap - One Percent Per Week Strategy')
    plt.tight_layout()
    return plt.gcf()

def calculate_benchmark_metrics(data, start_date, end_date, initial_value):
    """Calculate benchmark metrics"""
    mask = (data.index >= start_date) & (data.index <= end_date)
    benchmark_data = data[mask]
    
    if len(benchmark_data) == 0:
        return {
            'Total Return (%)': 0, 'CAGR (%)': 0, 'Volatility (%)': 0,
            'Sharpe Ratio': 0, 'Sortino Ratio': 0, 'Max Drawdown (%)': 0,
            'MAR Ratio': 0, 'Exposure (%)': 100.0
        }
    
    returns = benchmark_data.pct_change().fillna(0)
    initial_price = float(benchmark_data.iloc[0])
    final_price = float(benchmark_data.iloc[-1])
    
    total_return = (final_price / initial_price - 1) * 100
    years = max((benchmark_data.index[-1] - benchmark_data.index[0]).days / 365.25, 1/252)
    cagr = ((final_price / initial_price) ** (1/years) - 1) * 100
    
    volatility = float(returns.std()) * np.sqrt(252) * 100 if len(returns) > 1 else 0.0
    
    risk_free_rate = 0.02
    sharpe_ratio = (cagr/100 - risk_free_rate) / (volatility/100) if volatility > 0 else 0
    
    negative_returns = returns[returns < 0]
    if len(negative_returns) > 0:
        downside_volatility = float(negative_returns.std()) * np.sqrt(252) * 100
        sortino_ratio = (cagr/100 - risk_free_rate) / (downside_volatility/100) if downside_volatility > 0 else 0
    else:
        sortino_ratio = sharpe_ratio
    
    if len(returns) > 0:
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative / rolling_max - 1) * 100
        max_drawdown = float(drawdown.min())
    else:
        max_drawdown = 0.0
    
    mar_ratio = abs(cagr / max_drawdown) if max_drawdown != 0 else 0
    
    return {
        'Total Return (%)': round(total_return, 2),
        'CAGR (%)': round(cagr, 2),
        'Volatility (%)': round(volatility, 2),
        'Sharpe Ratio': round(sharpe_ratio, 3),
        'Sortino Ratio': round(sortino_ratio, 3),
        'Max Drawdown (%)': round(max_drawdown, 2),
        'MAR Ratio': round(mar_ratio, 3),
        'Exposure (%)': 100.0
    }

def export_to_excel(trades, equity_curve, performance_metrics, benchmarks, filename='OnePercent_Strategy_Results.xlsx'):
    """Export comprehensive results to Excel"""
    
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        # Performance Summary
        summary_data = []
        for strategy_name, metrics in benchmarks.items():
            row = {'Strategy': strategy_name}
            row.update(metrics)
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Performance_Summary', index=False)
        
        # All Trades
        trades_df = pd.DataFrame(trades)
        if len(trades_df) > 0:
            trades_df['Date'] = pd.to_datetime(trades_df['Date'])
            trades_df = trades_df.sort_values('Date')
            trades_df.to_excel(writer, sheet_name='All_Trades', index=False)
        
        # Equity Curve
        equity_df = pd.DataFrame(equity_curve)
        if len(equity_df) > 0:
            equity_df['Date'] = pd.to_datetime(equity_df['Date'])
            equity_df.to_excel(writer, sheet_name='Equity_Curve', index=False)
        
        # Trade Analysis
        trades_with_pnl = [t for t in trades if 'Trade_PnL' in t]
        if len(trades_with_pnl) > 0:
            trades_pnl_df = pd.DataFrame(trades_with_pnl)
            max_losses = [t.get('Max_Intraweek_Loss_Pct', 0) for t in trades_with_pnl]
            
            trade_analysis = {
                'Total Trades': len(trades_pnl_df),
                'Winning Trades': len(trades_pnl_df[trades_pnl_df['Trade_PnL'] > 0]),
                'Losing Trades': len(trades_pnl_df[trades_pnl_df['Trade_PnL'] <= 0]),
                'Win Rate (%)': len(trades_pnl_df[trades_pnl_df['Trade_PnL'] > 0]) / len(trades_pnl_df) * 100,
                'Average Win ($)': trades_pnl_df[trades_pnl_df['Trade_PnL'] > 0]['Trade_PnL'].mean(),
                'Average Loss ($)': trades_pnl_df[trades_pnl_df['Trade_PnL'] <= 0]['Trade_PnL'].mean(),
                'Largest Win ($)': trades_pnl_df['Trade_PnL'].max(),
                'Largest Loss ($)': trades_pnl_df['Trade_PnL'].min(),
                'Total P&L ($)': trades_pnl_df['Trade_PnL'].sum(),
                'Average Max Intraweek Loss (%)': np.mean(max_losses),
                'Worst Max Intraweek Loss (%)': min(max_losses) if max_losses else 0,
                'Std Dev Max Intraweek Loss (%)': np.std(max_losses) if len(max_losses) > 1 else 0
            }
            
            trade_analysis_df = pd.DataFrame([trade_analysis]).T
            trade_analysis_df.columns = ['Value']
            trade_analysis_df.to_excel(writer, sheet_name='Trade_Analysis')
        
        # Monthly Returns
        if len(equity_curve) > 0:
            eq_temp_df = pd.DataFrame(equity_curve)
            eq_temp_df['Date'] = pd.to_datetime(eq_temp_df['Date'])
            eq_temp_df = eq_temp_df.set_index('Date')
            
            eq_temp_df['Returns'] = eq_temp_df['Equity'].pct_change().fillna(0)
            monthly_returns = eq_temp_df['Returns'].resample('M').apply(lambda x: (1 + x).prod() - 1 if len(x) > 0 else 0) * 100
            yearly_returns = eq_temp_df['Returns'].resample('Y').apply(lambda x: (1 + x).prod() - 1 if len(x) > 0 else 0) * 100
            
            if len(monthly_returns) > 0:
                monthly_data = monthly_returns.reset_index()
                monthly_data['Year'] = monthly_data['Date'].dt.year
                monthly_data['Month'] = monthly_data['Date'].dt.strftime('%b')
                monthly_pivot = monthly_data.pivot(index='Year', columns='Month', values='Returns')
                
                yearly_data = yearly_returns.reset_index()
                yearly_data['Year'] = yearly_data['Date'].dt.year
                yearly_dict = dict(zip(yearly_data['Year'], yearly_data['Returns']))
                
                yearly_column = [yearly_dict.get(year, 0) for year in monthly_pivot.index]
                monthly_pivot['Year_Total'] = yearly_column
                monthly_pivot.to_excel(writer, sheet_name='Monthly_Yearly_Returns')
        
        # Intraweek Loss Analysis
        if len(trades_with_pnl) > 0:
            trades_loss_df = pd.DataFrame(trades_with_pnl)
            if 'Max_Intraweek_Loss_Pct' in trades_loss_df.columns:
                loss_analysis = trades_loss_df[['Date', 'Entry_Date', 'Type', 'Entry_Price', 'Exit_Price', 
                                               'Trade_Return_Pct', 'Max_Intraweek_Loss_Pct', 'Trade_PnL']]
                loss_analysis['Date'] = pd.to_datetime(loss_analysis['Date'])
                loss_analysis = loss_analysis.sort_values('Date')
                loss_analysis.to_excel(writer, sheet_name='Intraweek_Loss_Analysis', index=False)
    
    print(f"Results exported to {filename}")
    return filename

# Initialize and run strategy
print("ONE PERCENT PER WEEK v3 - Faithful Python Translation from WealthLab")
print("Original by Dion, Modified by Guy Fleury")
print("=" * 70)

strategy = OnePercentPerWeekStrategy(initial_capital=100000, commission=0.005, slippage=0.001)

# Download TQQQ data
tqqq_data = strategy.download_data('TQQQ', '2010-01-01')
print(f"Downloaded {len(tqqq_data)} days of TQQQ data")
print(f"Date range: {tqqq_data.index[0].date()} to {tqqq_data.index[-1].date()}")

# Show 2022 validation data
sample_2022 = tqqq_data[tqqq_data.index.year == 2022]
if len(sample_2022) > 0:
    print(f"\nTQQQ 2022 Reality Check:")
    print(f"  Start: ${sample_2022['Close'].iloc[0]:.2f}")
    print(f"  End: ${sample_2022['Close'].iloc[-1]:.2f}")
    print(f"  Actual 2022 return: {(sample_2022['Close'].iloc[-1]/sample_2022['Close'].iloc[0]-1)*100:.1f}%")

# Run backtest
trades, equity_curve = strategy.backtest(tqqq_data)

# Calculate performance metrics
performance_metrics = calculate_performance_metrics(equity_curve, trades)

print("\nPERFORMANCE SUMMARY:")
print("=" * 50)
for metric, value in performance_metrics.items():
    print(f"{metric:<30}: {value}")

# Download benchmark data
print("\nDownloading benchmark data...")
spy_data = yf.download('SPY', start='2010-01-01', progress=False)['Close']
qqq_data = yf.download('QQQ', start='2010-01-01', progress=False)['Close'] 
tqqq_bh_data = yf.download('TQQQ', start='2010-01-01', progress=False)['Close']
tlt_data = yf.download('TLT', start='2010-01-01', progress=False)['Close']

# Remove timezone info
spy_data.index = spy_data.index.tz_localize(None)
qqq_data.index = qqq_data.index.tz_localize(None)
tqqq_bh_data.index = tqqq_bh_data.index.tz_localize(None)
tlt_data.index = tlt_data.index.tz_localize(None)

if len(equity_curve) > 0:
    eq_df = pd.DataFrame(equity_curve)
    eq_df['Date'] = pd.to_datetime(eq_df['Date'])
    eq_df = eq_df.set_index('Date').sort_index()
    
    start_date = eq_df.index[0]
    end_date = eq_df.index[-1]
    
    # Calculate benchmark metrics
    benchmarks = {
        'OnePercent Strategy': performance_metrics,
        'SPY B&H': calculate_benchmark_metrics(spy_data, start_date, end_date, strategy.initial_capital),
        'QQQ B&H': calculate_benchmark_metrics(qqq_data, start_date, end_date, strategy.initial_capital),
        'TQQQ B&H': calculate_benchmark_metrics(tqqq_bh_data, start_date, end_date, strategy.initial_capital),
        'TLT B&H': calculate_benchmark_metrics(tlt_data, start_date, end_date, strategy.initial_capital)
    }
    
    comparison_df = pd.DataFrame(benchmarks).T
    print("\nBENCHMARK COMPARISON:")
    print("=" * 80)
    print(comparison_df.to_string())
    
    # 2022 validation
    eq_2022 = eq_df[eq_df.index.year == 2022]
    if len(eq_2022) > 0:
        strategy_2022_return = (eq_2022['Equity'].iloc[-1] / eq_2022['Equity'].iloc[0] - 1) * 100
        
        tqqq_2022 = tqqq_bh_data[(tqqq_bh_data.index >= '2022-01-01') & (tqqq_bh_data.index <= '2022-12-31')]
        if len(tqqq_2022) > 0:
            tqqq_2022_return = float((tqqq_2022.iloc[-1] / tqqq_2022.iloc[0] - 1) * 100)
        
        print(f"\n2022 PERFORMANCE VALIDATION:")
        print(f"  Strategy 2022 return: {strategy_2022_return:.2f}%")
        print(f"  TQQQ B&H 2022 return: {tqqq_2022_return:.2f}%")
        
        trades_2022 = [t for t in trades if 'Date' in t and pd.to_datetime(t['Date']).year == 2022 and 'Trade_PnL' in t]
        win_2022 = sum(1 for t in trades_2022 if t['Trade_PnL'] > 0)
        print(f"  2022 trades: {len(trades_2022)} (wins: {win_2022}, losses: {len(trades_2022)-win_2022})")

    # ALWAYS show charts and export
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 1, 1)
    plt.plot(eq_df.index, eq_df['Equity'], label='OnePercent Strategy', linewidth=2, color='blue')
    
    # Add benchmarks
    for name, data in [('SPY B&H', spy_data), ('QQQ B&H', qqq_data), 
                       ('TQQQ B&H', tqqq_bh_data), ('TLT B&H', tlt_data)]:
        mask = (data.index >= start_date) & (data.index <= end_date)
        benchmark_data = data[mask]
        if len(benchmark_data) > 0:
            normalized_benchmark = (benchmark_data / benchmark_data.iloc[0]) * strategy.initial_capital
            plt.plot(benchmark_data.index, normalized_benchmark, 
                    label=name, alpha=0.7, linewidth=1.5)
    
    plt.title('One Percent Per Week Strategy vs Benchmarks - Equity Curves')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Drawdown chart
    plt.subplot(2, 1, 2)
    if len(eq_df) > 1:
        returns = eq_df['Equity'].pct_change().fillna(0)
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative / rolling_max - 1) * 100
        
        plt.fill_between(eq_df.index, drawdown, 0, alpha=0.3, color='red')
        plt.plot(eq_df.index, drawdown, color='red', linewidth=1)
    
    plt.title('Strategy Drawdown (%)')
    plt.xlabel('Date')
    plt.ylabel('Drawdown (%)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Monthly heatmap
    heatmap_fig = create_monthly_heatmap(equity_curve)
    plt.show()
    
    # Export to Excel
    excel_filename = export_to_excel(trades, equity_curve, performance_metrics, benchmarks)

    print(f"\nFINAL RESULTS:")
    print(f"Final portfolio value: ${eq_df['Equity'].iloc[-1]:,.2f}")
    print(f"Total return: {performance_metrics['Total Return (%)']:.2f}%")
    print(f"Average Max Intraweek Loss: {performance_metrics['Avg Max Intraweek Loss (%)']:.2f}%")
    print(f"Worst Max Intraweek Loss: {performance_metrics['Worst Max Intraweek Loss (%)']:.2f}%")
    print(f"All results saved to: {excel_filename}")

else:
    print("No equity curve data generated.")