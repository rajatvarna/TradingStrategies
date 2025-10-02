import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# For Excel export
try:
    import xlsxwriter
    EXCEL_AVAILABLE = True
except ImportError:
    EXCEL_AVAILABLE = False
    print("Warning: xlsxwriter not available. Please install: pip install xlsxwriter")

# For advanced metrics
from scipy import stats

class BitcoinMAStrategy:
    def __init__(self, short_window=50, long_window=55, initial_capital=10000):
        self.short_window = short_window
        self.long_window = long_window
        self.initial_capital = initial_capital
        self.trades = []
        self.results = {}
        
    def download_data(self, symbol, start_date=None):
        """Download data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            if start_date is None:
                # Get earliest available data
                data = ticker.history(period="max")
            else:
                data = ticker.history(start=start_date)
            
            if data.empty:
                print(f"No data found for {symbol}")
                return None
                
            print(f"Downloaded {symbol}: {len(data)} rows from {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
            return data
        except Exception as e:
            print(f"Error downloading {symbol}: {e}")
            return None
    
    def calculate_signals(self, data):
        """Calculate moving average crossover signals"""
        df = data.copy()
        
        # Calculate moving averages
        df[f'MA_{self.short_window}'] = df['Close'].rolling(window=self.short_window).mean()
        df[f'MA_{self.long_window}'] = df['Close'].rolling(window=self.long_window).mean()
        
        # Generate signals using boolean indexing
        df['Signal'] = 0
        
        # Create condition for signal generation
        condition = (df[f'MA_{self.short_window}'] > df[f'MA_{self.long_window}']) & \
                   (df[f'MA_{self.short_window}'].notna()) & \
                   (df[f'MA_{self.long_window}'].notna())
        
        df.loc[condition, 'Signal'] = 1
        
        # Generate trading positions
        df['Position'] = df['Signal'].diff()
        
        return df
    
    def backtest_strategy(self, data):
        """Run the backtest"""
        df = self.calculate_signals(data)
        
        # Initialize variables
        position = 0
        cash = self.initial_capital
        portfolio_value = []
        trades = []
        entry_price = 0
        entry_date = None
        
        for i, row in df.iterrows():
            current_price = row['Close']
            
            if pd.isna(current_price) or pd.isna(row['Signal']):
                portfolio_value.append(cash if position == 0 else cash + position * current_price)
                continue
            
            # Buy signal
            if row['Position'] == 1 and position == 0:
                shares = cash / current_price
                position = shares
                cash = 0
                entry_price = current_price
                entry_date = i
                
            # Sell signal
            elif row['Position'] == -1 and position > 0:
                cash = position * current_price
                
                # Record trade
                if entry_date is not None:
                    pnl = cash - self.initial_capital if len(trades) == 0 else cash - trades[-1]['Exit_Value']
                    pnl_pct = (current_price - entry_price) / entry_price * 100
                    
                    trades.append({
                        'Entry_Date': entry_date,
                        'Exit_Date': i,
                        'Entry_Price': entry_price,
                        'Exit_Price': current_price,
                        'Shares': position,
                        'PnL': pnl,
                        'PnL_Pct': pnl_pct,
                        'Duration': (i - entry_date).days,
                        'Entry_Value': position * entry_price,
                        'Exit_Value': cash
                    })
                
                position = 0
            
            # Calculate portfolio value
            current_value = cash if position == 0 else cash + position * current_price
            portfolio_value.append(current_value)
        
        # Handle final position
        if position > 0:
            final_price = df['Close'].iloc[-1]
            final_value = position * final_price
            
            if entry_date is not None:
                pnl = final_value - self.initial_capital if len(trades) == 0 else final_value - trades[-1]['Exit_Value']
                pnl_pct = (final_price - entry_price) / entry_price * 100
                
                trades.append({
                    'Entry_Date': entry_date,
                    'Exit_Date': df.index[-1],
                    'Entry_Price': entry_price,
                    'Exit_Price': final_price,
                    'Shares': position,
                    'PnL': pnl,
                    'PnL_Pct': pnl_pct,
                    'Duration': (df.index[-1] - entry_date).days,
                    'Entry_Value': position * entry_price,
                    'Exit_Value': final_value
                })
            portfolio_value[-1] = final_value
        
        df['Portfolio_Value'] = portfolio_value
        df['Returns'] = df['Portfolio_Value'].pct_change()
        df['Cumulative_Returns'] = (df['Portfolio_Value'] / self.initial_capital - 1) * 100
        
        self.trades = trades
        return df
    
    def calculate_sharpe_ratio(self, returns, risk_free_rate=0.0):
        """Calculate Sharpe ratio"""
        if len(returns) < 2 or returns.std() == 0:
            return 0
        excess_returns = returns - risk_free_rate/252
        return (excess_returns.mean() / returns.std()) * np.sqrt(252)
    
    def calculate_sortino_ratio(self, returns, risk_free_rate=0.0):
        """Calculate Sortino ratio"""
        if len(returns) < 2:
            return 0
        excess_returns = returns - risk_free_rate/252
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0:
            return float('inf')
        downside_std = downside_returns.std()
        if downside_std == 0:
            return 0
        return (excess_returns.mean() / downside_std) * np.sqrt(252)
    
    def calculate_max_drawdown(self, returns):
        """Calculate maximum drawdown"""
        if len(returns) < 2:
            return 0
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min() * 100
    
    def calculate_metrics(self, df, benchmark_data=None):
        """Calculate comprehensive performance metrics"""
        returns = df['Returns'].dropna()
        
        if len(returns) == 0:
            print("Warning: No returns data available")
            return {}
        
        # Basic metrics
        total_return = (df['Portfolio_Value'].iloc[-1] / self.initial_capital - 1) * 100
        
        # Calculate annualized return properly
        years = len(df) / 252
        if years > 0:
            ann_return = ((df['Portfolio_Value'].iloc[-1] / self.initial_capital) ** (1/years) - 1) * 100
        else:
            ann_return = 0
        
        # Risk metrics
        volatility = returns.std() * np.sqrt(252) * 100 if len(returns) > 1 else 0
        sharpe_ratio = self.calculate_sharpe_ratio(returns)
        sortino_ratio = self.calculate_sortino_ratio(returns)
        max_drawdown = self.calculate_max_drawdown(returns)
        
        # MAR ratio (Calmar ratio)
        mar_ratio = ann_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Trade metrics
        if self.trades:
            trades_df = pd.DataFrame(self.trades)
            winning_trades = trades_df[trades_df['PnL'] > 0]
            losing_trades = trades_df[trades_df['PnL'] < 0]
            
            win_rate = len(winning_trades) / len(trades_df) * 100
            avg_win = winning_trades['PnL_Pct'].mean() if len(winning_trades) > 0 else 0
            avg_loss = abs(losing_trades['PnL_Pct'].mean()) if len(losing_trades) > 0 else 0
            profit_factor = abs(winning_trades['PnL'].sum() / losing_trades['PnL'].sum()) if len(losing_trades) > 0 and losing_trades['PnL'].sum() != 0 else float('inf')
            avg_trade_duration = trades_df['Duration'].mean()
        else:
            win_rate = avg_win = avg_loss = profit_factor = avg_trade_duration = 0
        
        # Exposure
        signals = df['Signal'].dropna()
        exposure = (signals.sum() / len(signals)) * 100 if len(signals) > 0 else 0
        
        metrics = {
            'Total_Return_Pct': total_return,
            'Annualized_Return_Pct': ann_return,
            'Volatility_Pct': volatility,
            'Sharpe_Ratio': sharpe_ratio,
            'Sortino_Ratio': sortino_ratio,
            'Max_Drawdown_Pct': max_drawdown,
            'MAR_Ratio': mar_ratio,
            'Win_Rate_Pct': win_rate,
            'Avg_Win_Pct': avg_win,
            'Avg_Loss_Pct': avg_loss,
            'Profit_Factor': profit_factor,
            'Total_Trades': len(self.trades),
            'Avg_Trade_Duration_Days': avg_trade_duration,
            'Exposure_Pct': exposure,
            'Start_Date': df.index[0].strftime('%Y-%m-%d'),
            'End_Date': df.index[-1].strftime('%Y-%m-%d'),
            'Final_Portfolio_Value': df['Portfolio_Value'].iloc[-1]
        }
        
        return metrics
    
    def create_monthly_returns(self, df):
        """Create monthly returns heatmap data with yearly totals"""
        try:
            monthly_returns = df['Returns'].resample('M').apply(lambda x: (1 + x).prod() - 1) * 100
            monthly_returns.index = monthly_returns.index.to_period('M')
            
            # Create pivot table for heatmap
            monthly_pivot = monthly_returns.reset_index()
            monthly_pivot['Year'] = monthly_pivot['Date'].dt.year
            monthly_pivot['Month'] = monthly_pivot['Date'].dt.month
            
            pivot_table = monthly_pivot.pivot(index='Year', columns='Month', values='Returns')
            
            # Reindex to ensure all months are present
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            pivot_table = pivot_table.reindex(columns=range(1, 13))
            pivot_table.columns = month_names
            
            # Calculate yearly returns and add as a column
            yearly_returns_for_heatmap = []
            for year in pivot_table.index:
                year_data = pivot_table.loc[year].dropna()
                if len(year_data) > 0:
                    # Convert monthly percentages back to decimal, compound, then back to percentage
                    yearly_return = ((1 + year_data/100).prod() - 1) * 100
                    yearly_returns_for_heatmap.append(yearly_return)
                else:
                    yearly_returns_for_heatmap.append(np.nan)
            
            pivot_table['YEAR'] = yearly_returns_for_heatmap
            
            return pivot_table, monthly_returns
        except Exception as e:
            print(f"Error creating monthly returns: {e}")
            return pd.DataFrame(), pd.Series()
    
    def monte_carlo_simulation(self, df, num_simulations=1000):
        """Run Monte Carlo simulation"""
        returns = df['Returns'].dropna()
        
        if len(returns) < 2:
            print("Warning: Not enough data for Monte Carlo simulation")
            return None, None
        
        # Parameters for simulation
        mean_return = returns.mean()
        std_return = returns.std()
        num_days = len(returns)
        
        # Run simulations
        simulation_results = []
        
        for _ in range(num_simulations):
            # Generate random returns
            random_returns = np.random.normal(mean_return, std_return, num_days)
            
            # Calculate cumulative value
            cumulative_value = self.initial_capital
            for ret in random_returns:
                cumulative_value *= (1 + ret)
            
            total_return = (cumulative_value / self.initial_capital - 1) * 100
            simulation_results.append(total_return)
        
        simulation_results = np.array(simulation_results)
        
        # Calculate statistics
        mc_stats = {
            'Mean_Return_Pct': np.mean(simulation_results),
            'Median_Return_Pct': np.median(simulation_results),
            'Std_Return_Pct': np.std(simulation_results),
            'Min_Return_Pct': np.min(simulation_results),
            'Max_Return_Pct': np.max(simulation_results),
            'VaR_5_Pct': np.percentile(simulation_results, 5),
            'VaR_1_Pct': np.percentile(simulation_results, 1),
            'Prob_Positive_Return': np.sum(simulation_results > 0) / num_simulations * 100,
            'Prob_Outperform_BuyHold': 0  # Will be calculated when we have benchmark
        }

        return mc_stats
        
    def calculate_buy_hold_metrics(self, data, start_date, end_date):
        """Calculate buy and hold metrics for comparison"""
        try:
            # Align data to strategy period
            aligned_data = data[start_date:end_date].copy()
            
            if len(aligned_data) < 2:
                return {}
            
            # Calculate buy and hold returns
            aligned_data['BH_Returns'] = aligned_data['Close'].pct_change()
            aligned_data['BH_Cumulative'] = (aligned_data['Close'] / aligned_data['Close'].iloc[0] - 1) * 100
            aligned_data['BH_Portfolio_Value'] = aligned_data['Close'] / aligned_data['Close'].iloc[0] * self.initial_capital
            
            returns = aligned_data['BH_Returns'].dropna()
            
            if len(returns) == 0:
                return {}
            
            # Calculate metrics
            total_return = (aligned_data['Close'].iloc[-1] / aligned_data['Close'].iloc[0] - 1) * 100
            
            years = len(aligned_data) / 252
            ann_return = ((aligned_data['Close'].iloc[-1] / aligned_data['Close'].iloc[0]) ** (1/years) - 1) * 100 if years > 0 else 0
            
            volatility = returns.std() * np.sqrt(252) * 100 if len(returns) > 1 else 0
            sharpe_ratio = self.calculate_sharpe_ratio(returns)
            sortino_ratio = self.calculate_sortino_ratio(returns)
            max_drawdown = self.calculate_max_drawdown(returns)
            mar_ratio = ann_return / abs(max_drawdown) if max_drawdown != 0 else 0
            
            metrics = {
                'Total_Return_Pct': total_return,
                'Annualized_Return_Pct': ann_return,
                'Volatility_Pct': volatility,
                'Sharpe_Ratio': sharpe_ratio,
                'Sortino_Ratio': sortino_ratio,
                'Max_Drawdown_Pct': max_drawdown,
                'MAR_Ratio': mar_ratio,
                'Win_Rate_Pct': 0,  # N/A for buy and hold
                'Avg_Win_Pct': 0,   # N/A for buy and hold
                'Avg_Loss_Pct': 0,  # N/A for buy and hold
                'Profit_Factor': 0, # N/A for buy and hold
                'Total_Trades': 1,  # Single buy and hold
                'Avg_Trade_Duration_Days': len(aligned_data),
                'Exposure_Pct': 100,  # Always fully invested
                'Start_Date': aligned_data.index[0].strftime('%Y-%m-%d'),
                'End_Date': aligned_data.index[-1].strftime('%Y-%m-%d'),
                'Final_Portfolio_Value': aligned_data['BH_Portfolio_Value'].iloc[-1]
            }
            
            return metrics, aligned_data
            
        except Exception as e:
            print(f"Error calculating buy & hold metrics: {e}")
            return {}, pd.DataFrame()

def run_complete_backtest():
    """Run the complete backtesting analysis"""
    
    # Initialize strategy
    strategy = BitcoinMAStrategy(short_window=50, long_window=55, initial_capital=10000)
    
    print("="*60)
    print("BITCOIN MOVING AVERAGE CROSSOVER STRATEGY BACKTEST")
    print("Strategy: 50-day MA vs 55-day MA crossover")
    print("="*60)
    
    # Download Bitcoin data
    print("\n1. Downloading Bitcoin data...")
    btc_data = strategy.download_data('BTC-USD')
    
    if btc_data is None or btc_data.empty:
        print("Error: Could not download Bitcoin data")
        return
    
    # Download benchmark data
    print("\n2. Downloading benchmark data...")
    spy_data = strategy.download_data('SPY')
    qqq_data = strategy.download_data('QQQ')
    
    # Run backtest on Bitcoin
    print("\n3. Running Bitcoin strategy backtest...")
    btc_results = strategy.backtest_strategy(btc_data)
    btc_metrics = strategy.calculate_metrics(btc_results)
    
    if not btc_metrics:
        print("Error: Could not calculate metrics")
        return
    
    # Calculate Bitcoin buy & hold metrics for the same period
    print("\n4. Calculating Bitcoin buy & hold metrics...")
    start_date = btc_results.index[0]
    end_date = btc_results.index[-1]
    
    btc_bh_metrics, btc_bh_data = strategy.calculate_buy_hold_metrics(btc_data, start_date, end_date)
    
    # Calculate buy and hold for other benchmarks (aligned dates)
    spy_buy_hold = 0
    qqq_buy_hold = 0
    
    if spy_data is not None and len(spy_data) > 0:
        try:
            spy_aligned = spy_data[start_date:end_date]
            if len(spy_aligned) > 0:
                spy_buy_hold = (spy_aligned['Close'].iloc[-1] / spy_aligned['Close'].iloc[0] - 1) * 100
        except:
            spy_buy_hold = 0
    
    if qqq_data is not None and len(qqq_data) > 0:
        try:
            qqq_aligned = qqq_data[start_date:end_date]
            if len(qqq_aligned) > 0:
                qqq_buy_hold = (qqq_aligned['Close'].iloc[-1] / qqq_aligned['Close'].iloc[0] - 1) * 100
        except:
            qqq_buy_hold = 0
    
    # Create monthly returns
    print("\n5. Creating monthly returns analysis...")
    monthly_pivot, monthly_returns = strategy.create_monthly_returns(btc_results)
    
    # Calculate yearly returns
    try:
        if len(monthly_returns) > 0:
            yearly_returns = monthly_returns.resample('Y').apply(lambda x: (1 + x/100).prod() - 1) * 100
        else:
            yearly_returns = pd.Series()
    except:
        yearly_returns = pd.Series()
    
    # Monte Carlo simulation
    print("\n6. Running Monte Carlo simulation...")
    mc_result = strategy.monte_carlo_simulation(btc_results)
    
    if mc_result is not None and len(mc_result) == 2:
        mc_results, mc_stats = mc_result
    else:
        mc_results, mc_stats = None, None
        print("Monte Carlo simulation skipped due to insufficient data")
    
    # Display results
    print("\n" + "="*80)
    print("COMPREHENSIVE PERFORMANCE COMPARISON")
    print("="*80)
    
    # Create comparison table
    print(f"\n{'METRIC':<25} {'MA STRATEGY':<15} {'BUY & HOLD':<15} {'DIFFERENCE':<15}")
    print("-" * 80)
    
    if btc_bh_metrics:
        comparison_metrics = [
            ('Total Return (%)', 'Total_Return_Pct'),
            ('Annualized Return (%)', 'Annualized_Return_Pct'),
            ('Volatility (%)', 'Volatility_Pct'),
            ('Sharpe Ratio', 'Sharpe_Ratio'),
            ('Sortino Ratio', 'Sortino_Ratio'),
            ('Max Drawdown (%)', 'Max_Drawdown_Pct'),
            ('MAR Ratio', 'MAR_Ratio')
        ]
        
        for display_name, metric_key in comparison_metrics:
            strategy_val = btc_metrics.get(metric_key, 0)
            bh_val = btc_bh_metrics.get(metric_key, 0)
            difference = strategy_val - bh_val
            
            if 'Ratio' in display_name:
                print(f"{display_name:<25} {strategy_val:<15.3f} {bh_val:<15.3f} {difference:<+15.3f}")
            else:
                print(f"{display_name:<25} {strategy_val:<15.2f} {bh_val:<15.2f} {difference:<+15.2f}")
    
    print("\n" + "="*80)
    print("STRATEGY-SPECIFIC METRICS")
    print("="*80)
    
    print(f"\nTrading Statistics (MA Strategy Only):")
    print(f"Win Rate: {btc_metrics['Win_Rate_Pct']:.2f}%")
    print(f"Average Win: {btc_metrics['Avg_Win_Pct']:.2f}%")
    print(f"Average Loss: {btc_metrics['Avg_Loss_Pct']:.2f}%")
    print(f"Profit Factor: {btc_metrics['Profit_Factor']:.2f}")
    print(f"Total Trades: {btc_metrics['Total_Trades']}")
    print(f"Exposure: {btc_metrics['Exposure_Pct']:.2f}%")
    
    print(f"\nOther Benchmarks Comparison:")
    print(f"SPY Buy & Hold: {spy_buy_hold:.2f}%")
    print(f"QQQ Buy & Hold: {qqq_buy_hold:.2f}%")
    
    if mc_stats:
        print(f"\nMonte Carlo Results ({len(mc_results)} simulations):")
        print(f"Mean Return: {mc_stats['Mean_Return_Pct']:.2f}%")
        print(f"VaR (5%): {mc_stats['VaR_5_Pct']:.2f}%")
        print(f"Probability of Positive Return: {mc_stats['Prob_Positive_Return']:.1f}%")
    
    # Create visualizations
    print("\n7. Creating visualizations...")
    
    try:
        # Set up the plot style
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Bitcoin Moving Average Strategy vs Buy & Hold Analysis', fontsize=16, fontweight='bold')
        
        # Main equity curve comparison
        ax1 = axes[0, 0]
        ax1.plot(btc_results.index, btc_results['Cumulative_Returns'], label='MA Strategy', linewidth=2, color='blue')
        
        # Add buy and hold comparison using aligned data
        if not btc_bh_data.empty:
            ax1.plot(btc_bh_data.index, btc_bh_data['BH_Cumulative'], label='Buy & Hold BTC', linewidth=2, alpha=0.7, color='orange')
        
        ax1.set_title('Cumulative Returns Comparison')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Cumulative Returns (%)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Drawdown comparison
        ax2 = axes[0, 1]
        returns = btc_results['Returns'].dropna()
        if len(returns) > 1:
            rolling_max = (1 + returns).cumprod().expanding().max()
            drawdown = ((1 + returns).cumprod() / rolling_max - 1) * 100
            ax2.fill_between(btc_results.index[1:len(drawdown)+1], drawdown, 0, alpha=0.3, color='red', label='MA Strategy')
            ax2.plot(btc_results.index[1:len(drawdown)+1], drawdown, color='red', linewidth=1)
        
        # Add buy & hold drawdown
        if not btc_bh_data.empty:
            bh_returns = btc_bh_data['BH_Returns'].dropna()
            if len(bh_returns) > 1:
                bh_rolling_max = (1 + bh_returns).cumprod().expanding().max()
                bh_drawdown = ((1 + bh_returns).cumprod() / bh_rolling_max - 1) * 100
                ax2.plot(btc_bh_data.index[1:len(bh_drawdown)+1], bh_drawdown, color='orange', linewidth=1, alpha=0.7, label='Buy & Hold')
        
        ax2.set_title('Drawdown Comparison')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Drawdown (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Monthly returns heatmap with yearly totals
        ax3 = axes[1, 0]
        if not monthly_pivot.empty and monthly_pivot.shape[0] > 0:
            # Create heatmap
            im = ax3.imshow(monthly_pivot.values, cmap='RdYlGn', aspect='auto', interpolation='nearest')
            ax3.set_xticks(range(len(monthly_pivot.columns)))
            ax3.set_xticklabels(monthly_pivot.columns)
            ax3.set_yticks(range(len(monthly_pivot.index)))
            ax3.set_yticklabels(monthly_pivot.index)
            ax3.set_title('Monthly Returns Heatmap (%) with Yearly Totals')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax3)
            cbar.set_label('Monthly Return (%)')
            
            # Add text annotations
            for i in range(len(monthly_pivot.index)):
                for j in range(len(monthly_pivot.columns)):
                    if not pd.isna(monthly_pivot.iloc[i, j]):
                        text = ax3.text(j, i, f'{monthly_pivot.iloc[i, j]:.1f}', 
                                      ha="center", va="center", color="black", fontsize=8)
        else:
            ax3.text(0.5, 0.5, 'No monthly data available', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Monthly Returns Heatmap')
        
        # Performance comparison bar chart
        ax4 = axes[1, 1]
        if btc_bh_metrics:
            metrics_names = ['Total Return\n(%)', 'Ann. Return\n(%)', 'Sharpe\nRatio', 'Max DD\n(%)']
            strategy_values = [
                btc_metrics['Total_Return_Pct'],
                btc_metrics['Annualized_Return_Pct'],
                btc_metrics['Sharpe_Ratio'],
                abs(btc_metrics['Max_Drawdown_Pct'])
            ]
            bh_values = [
                btc_bh_metrics['Total_Return_Pct'],
                btc_bh_metrics['Annualized_Return_Pct'],
                btc_bh_metrics['Sharpe_Ratio'],
                abs(btc_bh_metrics['Max_Drawdown_Pct'])
            ]
            
            x = np.arange(len(metrics_names))
            width = 0.35
            
            ax4.bar(x - width/2, strategy_values, width, label='MA Strategy', color='blue', alpha=0.7)
            ax4.bar(x + width/2, bh_values, width, label='Buy & Hold', color='orange', alpha=0.7)
            
            ax4.set_xlabel('Metrics')
            ax4.set_ylabel('Values')
            ax4.set_title('Performance Metrics Comparison')
            ax4.set_xticks(x)
            ax4.set_xticklabels(metrics_names)
            ax4.legend()
            ax4.grid(True, alpha=0.3, axis='y')
        else:
            ax4.text(0.5, 0.5, 'Comparison data not available', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Performance Metrics Comparison')
        
        plt.tight_layout()
        plt.savefig('bitcoin_strategy_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    except Exception as e:
        print(f"Error creating visualizations: {e}")
    
    # Create Excel report
    print("\n8. Creating comprehensive Excel report...")
    try:
        create_excel_report(btc_results, btc_metrics, strategy.trades, monthly_pivot, 
                           yearly_returns, mc_stats, {
                               'BTC_Strategy': btc_metrics['Total_Return_Pct'],
                               'BTC_BuyHold': btc_bh_metrics.get('Total_Return_Pct', 0),
                               'SPY_BuyHold': spy_buy_hold,
                               'QQQ_BuyHold': qqq_buy_hold
                           }, btc_bh_metrics, btc_bh_data)
    except Exception as e:
        print(f"Error creating Excel report: {e}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("Files saved:")
    print("- bitcoin_strategy_analysis.png")
    if EXCEL_AVAILABLE:
        print("- bitcoin_strategy_report.xlsx")
    else:
        print("- Excel report skipped (xlsxwriter not available)")
    print("="*80)

def create_excel_report(results_df, metrics, trades, monthly_pivot, yearly_returns, mc_stats, benchmarks, btc_bh_metrics=None, btc_bh_data=None):
    """Create comprehensive Excel report using xlsxwriter"""
    
    if not EXCEL_AVAILABLE:
        print("Excel export not available. Please install xlsxwriter: pip install xlsxwriter")
        return
    
    try:
        # Create workbook and add formats
        workbook = xlsxwriter.Workbook('bitcoin_strategy_report.xlsx', {'remove_timezone': True})
        
        # Define formats
        header_format = workbook.add_format({
            'bold': True,
            'font_color': 'white',
            'bg_color': '#366092',
            'border': 1
        })
        
        good_format = workbook.add_format({
            'bg_color': '#90EE90',
            'border': 1
        })
        
        bad_format = workbook.add_format({
            'bg_color': '#FFB6C1',
            'border': 1
        })
        
        number_format = workbook.add_format({'num_format': '#,##0.00'})
        percent_format = workbook.add_format({'num_format': '0.00%'})
        date_format = workbook.add_format({'num_format': 'yyyy-mm-dd'})
        
        # 1. Executive Summary Sheet
        exec_ws = workbook.add_worksheet('Executive_Summary')
        exec_ws.write('A1', 'BITCOIN MOVING AVERAGE STRATEGY ANALYSIS', header_format)
        exec_ws.merge_range('A1:C1', 'BITCOIN MOVING AVERAGE STRATEGY ANALYSIS', header_format)
        
        exec_ws.write('A3', 'Strategy Overview', header_format)
        exec_ws.write('A4', 'Strategy Type: 50-day vs 55-day Moving Average Crossover')
        exec_ws.write('A5', f"Period: {metrics['Start_Date']} to {metrics['End_Date']}")
        exec_ws.write('A6', f"Initial Capital: ${metrics.get('Final_Portfolio_Value', 0) / (1 + metrics['Total_Return_Pct']/100):,.0f}")
        
        if btc_bh_metrics:
            exec_ws.write('A8', 'KEY PERFORMANCE COMPARISON', header_format)
            exec_ws.write('A9', 'Metric', header_format)
            exec_ws.write('B9', 'MA Strategy', header_format)
            exec_ws.write('C9', 'Buy & Hold', header_format)
            exec_ws.write('D9', 'Advantage', header_format)
            
            key_metrics = [
                ('Total Return (%)', 'Total_Return_Pct'),
                ('Annualized Return (%)', 'Annualized_Return_Pct'),
                ('Sharpe Ratio', 'Sharpe_Ratio'),
                ('Max Drawdown (%)', 'Max_Drawdown_Pct'),
                ('Volatility (%)', 'Volatility_Pct')
            ]
            
            for i, (name, key) in enumerate(key_metrics, 10):
                strategy_val = metrics.get(key, 0)
                bh_val = btc_bh_metrics.get(key, 0)
                
                exec_ws.write(i, 0, name)
                
                if '(%)' in name:
                    exec_ws.write(i, 1, strategy_val/100, percent_format)
                    exec_ws.write(i, 2, bh_val/100, percent_format)
                else:
                    exec_ws.write(i, 1, strategy_val, number_format)
                    exec_ws.write(i, 2, bh_val, number_format)
                
                # Determine advantage (higher is better except for drawdown and volatility)
                if 'Drawdown' in name or 'Volatility' in name:
                    advantage = 'Strategy' if strategy_val < bh_val else 'Buy & Hold'
                else:
                    advantage = 'Strategy' if strategy_val > bh_val else 'Buy & Hold'
                
                exec_ws.write(i, 3, advantage)
        
        exec_ws.set_column('A:A', 25)
        exec_ws.set_column('B:D', 15)
        
        # 2. Detailed Comparison Sheet
        if btc_bh_metrics:
            comp_ws = workbook.add_worksheet('Strategy_vs_BuyHold')
            comp_ws.write('A1', 'Metric', header_format)
            comp_ws.write('B1', 'MA Strategy', header_format)
            comp_ws.write('C1', 'Bitcoin Buy & Hold', header_format)
            comp_ws.write('D1', 'Difference', header_format)
            comp_ws.write('E1', 'Winner', header_format)
            
            comparison_data = [
                ('Total Return (%)', 'Total_Return_Pct', True),
                ('Annualized Return (%)', 'Annualized_Return_Pct', True),
                ('Volatility (%)', 'Volatility_Pct', False),
                ('Sharpe Ratio', 'Sharpe_Ratio', True),
                ('Sortino Ratio', 'Sortino_Ratio', True),
                ('Max Drawdown (%)', 'Max_Drawdown_Pct', False),
                ('MAR Ratio', 'MAR_Ratio', True),
                ('Exposure (%)', 'Exposure_Pct', None),
                ('Total Trades', 'Total_Trades', None)
            ]
            
            for i, (name, key, higher_better) in enumerate(comparison_data, 1):
                strategy_val = metrics.get(key, 0)
                bh_val = btc_bh_metrics.get(key, 0)
                difference = strategy_val - bh_val
                
                comp_ws.write(i, 0, name)
                
                # Format values
                if '(%)' in name:
                    comp_ws.write(i, 1, strategy_val/100, percent_format)
                    comp_ws.write(i, 2, bh_val/100, percent_format)
                    comp_ws.write(i, 3, difference/100, percent_format)
                else:
                    comp_ws.write(i, 1, strategy_val, number_format)
                    comp_ws.write(i, 2, bh_val, number_format)
                    comp_ws.write(i, 3, difference, number_format)
                
                # Determine winner
                if higher_better is not None:
                    if higher_better:
                        winner = 'Strategy' if strategy_val > bh_val else 'Buy & Hold'
                    else:
                        winner = 'Strategy' if strategy_val < bh_val else 'Buy & Hold'
                    comp_ws.write(i, 4, winner)
                
            comp_ws.set_column('A:A', 25)
            comp_ws.set_column('B:E', 15)
        
        # 3. Original Summary Sheet
        summary_ws = workbook.add_worksheet('Strategy_Metrics')
        summary_ws.write('A1', 'Performance Metrics', header_format)
        summary_ws.write('B1', 'Value', header_format)
        
        row = 1
        for metric, value in metrics.items():
            summary_ws.write(row, 0, metric.replace('_', ' ').title())
            if 'Pct' in metric or 'Rate' in metric:
                summary_ws.write(row, 1, value/100, percent_format)
            elif isinstance(value, (int, float)) and metric != 'Total_Trades':
                summary_ws.write(row, 1, value, number_format)
            else:
                summary_ws.write(row, 1, value)
            row += 1
        
        summary_ws.set_column('A:A', 25)
        summary_ws.set_column('B:B', 15)
        
        # 4. All Benchmarks Sheet
        benchmark_ws = workbook.add_worksheet('All_Benchmarks')
        benchmark_ws.write('A1', 'Strategy', header_format)
        benchmark_ws.write('B1', 'Total Return (%)', header_format)
        
        benchmark_data = [
            ['Bitcoin MA Strategy', benchmarks['BTC_Strategy']],
            ['Bitcoin Buy & Hold', benchmarks['BTC_BuyHold']],
            ['SPY Buy & Hold', benchmarks['SPY_BuyHold']],
            ['QQQ Buy & Hold', benchmarks['QQQ_BuyHold']]
        ]
        
        for i, (strategy, return_pct) in enumerate(benchmark_data, 1):
            benchmark_ws.write(i, 0, strategy)
            benchmark_ws.write(i, 1, return_pct/100, percent_format)
        
        benchmark_ws.set_column('A:A', 20)
        benchmark_ws.set_column('B:B', 15)
        
        # 5. Daily Data Sheet
        daily_ws = workbook.add_worksheet('Daily_Data')
        
        # Write headers
        headers = ['Date'] + list(results_df.columns)
        for col, header in enumerate(headers):
            daily_ws.write(0, col, header, header_format)
        
        # Write data
        for row, (date, data) in enumerate(results_df.iterrows(), 1):
            daily_ws.write(row, 0, date, date_format)
            for col, value in enumerate(data, 1):
                if pd.isna(value):
                    daily_ws.write(row, col, '')
                elif isinstance(value, (int, float)):
                    daily_ws.write(row, col, value, number_format)
                else:
                    daily_ws.write(row, col, value)
        
        daily_ws.set_column('A:A', 12)
        daily_ws.set_column('B:Z', 12)
        
        # 6. Trades Sheet
        if trades:
            trades_ws = workbook.add_worksheet('Trades')
            trades_df = pd.DataFrame(trades)
            
            # Write headers
            for col, header in enumerate(trades_df.columns):
                trades_ws.write(0, col, header.replace('_', ' ').title(), header_format)
            
            # Write data
            for row_idx, (_, row_data) in enumerate(trades_df.iterrows(), 1):
                for col_idx, value in enumerate(row_data):
                    if pd.isna(value):
                        trades_ws.write(row_idx, col_idx, '')
                    elif col_idx in [0, 1]:  # Date columns
                        trades_ws.write(row_idx, col_idx, value, date_format)
                    elif 'Pct' in trades_df.columns[col_idx]:
                        # Color code profitable vs losing trades
                        if value > 0:
                            cell_format = workbook.add_format({'num_format': '0.00%', 'bg_color': '#90EE90'})
                        else:
                            cell_format = workbook.add_format({'num_format': '0.00%', 'bg_color': '#FFB6C1'})
                        trades_ws.write(row_idx, col_idx, value/100, cell_format)
                    elif isinstance(value, (int, float)):
                        trades_ws.write(row_idx, col_idx, value, number_format)
                    else:
                        trades_ws.write(row_idx, col_idx, value)
            
            trades_ws.set_column('A:B', 12)  # Date columns
            trades_ws.set_column('C:Z', 10)
        
        # 7. Monthly Returns with Yearly Totals
        if not monthly_pivot.empty:
            monthly_ws = workbook.add_worksheet('Monthly_Returns')
            
            # Write year header
            monthly_ws.write(0, 0, 'Year', header_format)
            
            # Write month headers
            for col, month in enumerate(monthly_pivot.columns, 1):
                monthly_ws.write(0, col, month, header_format)
            
            # Write data with conditional formatting
            for row_idx, (year, row_data) in enumerate(monthly_pivot.iterrows(), 1):
                monthly_ws.write(row_idx, 0, year)
                for col_idx, value in enumerate(row_data, 1):
                    if pd.isna(value):
                        monthly_ws.write(row_idx, col_idx, '')
                    else:
                        # Color code positive/negative returns
                        if value > 0:
                            cell_format = workbook.add_format({'num_format': '0.00%', 'bg_color': '#90EE90'})
                        else:
                            cell_format = workbook.add_format({'num_format': '0.00%', 'bg_color': '#FFB6C1'})
                        monthly_ws.write(row_idx, col_idx, value/100, cell_format)
            
            monthly_ws.set_column('A:A', 8)
            monthly_ws.set_column('B:M', 8)
            monthly_ws.set_column('N:N', 10)  # YEAR column
        
        # 8. Monte Carlo Sheet
        if mc_stats:
            mc_ws = workbook.add_worksheet('Monte_Carlo')
            mc_ws.write('A1', 'Monte Carlo Statistics', header_format)
            mc_ws.write('B1', 'Value', header_format)
            
            mc_data = [
                ['Mean Return (%)', mc_stats['Mean_Return_Pct']],
                ['Median Return (%)', mc_stats['Median_Return_Pct']],
                ['Std Return (%)', mc_stats['Std_Return_Pct']],
                ['Min Return (%)', mc_stats['Min_Return_Pct']],
                ['Max Return (%)', mc_stats['Max_Return_Pct']],
                ['VaR 5% (%)', mc_stats['VaR_5_Pct']],
                ['VaR 1% (%)', mc_stats['VaR_1_Pct']],
                ['Prob Positive Return (%)', mc_stats['Prob_Positive_Return']]
            ]
            
            for i, (metric, value) in enumerate(mc_data, 1):
                mc_ws.write(i, 0, metric)
                if 'Prob' in metric:
                    mc_ws.write(i, 1, value/100, percent_format)
                else:
                    mc_ws.write(i, 1, value, number_format)
            
            mc_ws.set_column('A:A', 25)
            mc_ws.set_column('B:B', 15)
        
        workbook.close()
        print("Comprehensive Excel report 'bitcoin_strategy_report.xlsx' created successfully!")
        
    except Exception as e:
        print(f"Error creating Excel report: {e}")
        raise

if __name__ == "__main__":
    run_complete_backtest()