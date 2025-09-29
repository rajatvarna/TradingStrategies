"""
Complete Mean Reversion Strategy Backtesting Suite
===================================================

This script implements and backtests the mean reversion strategies from the Quantitativo blog post:
"A Mean Reversion Strategy with 2.11 Sharpe"

Features:
- Original strategy and 3 improvements
- Comprehensive performance metrics
- Monte Carlo simulation
- Enhanced risk analysis
- Parameter sensitivity testing
- Stress testing
- Excel export with all results

Author: AI Assistant
Date: 2025
"""

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import os
warnings.filterwarnings('ignore')

# Performance metrics and statistics
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Machine learning for enhanced analysis
from sklearn.linear_model import LinearRegression

# Excel export
import xlsxwriter
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows

class MeanReversionStrategy:
    """Main strategy implementation and backtesting class"""
    
    def __init__(self, save_charts=True, chart_folder='qqq_mr'):
        self.data = None
        self.results = {}
        self.trades = {}
        self.save_charts = save_charts
        self.chart_folder = chart_folder
        
        # Create charts folder if it doesn't exist
        if self.save_charts and not os.path.exists(self.chart_folder):
            os.makedirs(self.chart_folder)
            print(f"Created folder: {self.chart_folder}")
        
    def download_data(self, symbols=['QQQ', 'SPY', 'PSQ'], start_date='1999-01-01'):
        """Download data from Yahoo Finance"""
        print(f"Downloading data for {symbols} from {start_date}...")
        
        data = {}
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start_date, auto_adjust=True)
                if not df.empty:
                    data[symbol] = df
                    print(f"Downloaded {len(df)} rows for {symbol}")
                else:
                    print(f"No data found for {symbol}")
            except Exception as e:
                print(f"Error downloading {symbol}: {e}")
        
        self.data = data
        return data
    
    def calculate_indicators(self, df):
        """Calculate technical indicators"""
        # Rolling mean of High - Low over 25 days
        df['HL_Mean_25'] = (df['High'] - df['Low']).rolling(25).mean()
        
        # IBS indicator
        df['IBS'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
        
        # Rolling high over 10 days
        df['High_10'] = df['High'].rolling(10).max()
        
        # Lower band
        df['Lower_Band'] = df['High_10'] - 2.5 * df['HL_Mean_25']
        
        # Moving averages for regime filter
        df['SMA_200'] = df['Close'].rolling(200).mean()
        df['SMA_300'] = df['Close'].rolling(300).mean()
        
        # Previous day's high
        df['Prev_High'] = df['High'].shift(1)
        
        return df
    
    def original_strategy(self, df, symbol='QQQ'):
        """Original mean reversion strategy"""
        df = self.calculate_indicators(df.copy())
        
        # Entry conditions
        entry_condition = (df['Close'] < df['Lower_Band']) & (df['IBS'] < 0.3)
        
        # Exit condition 
        exit_condition = df['Close'] > df['Prev_High']
        
        return self.backtest_strategy(df, entry_condition, exit_condition, f'Original_{symbol}')
    
    def market_regime_filter_strategy(self, df, symbol='QQQ'):
        """Strategy with market regime filter"""
        df = self.calculate_indicators(df.copy())
        
        # Bull market condition
        bull_market = df['Close'] > df['SMA_300']
        
        # Entry conditions (only in bull market)
        entry_condition = (df['Close'] < df['Lower_Band']) & (df['IBS'] < 0.3) & bull_market
        
        # Exit condition
        exit_condition = df['Close'] > df['Prev_High']
        
        return self.backtest_strategy(df, entry_condition, exit_condition, f'MarketRegime_{symbol}')
    
    def dynamic_stop_strategy(self, df, symbol='QQQ'):
        """Strategy with dynamic stop loss"""
        df = self.calculate_indicators(df.copy())
        
        # Entry conditions
        entry_condition = (df['Close'] < df['Lower_Band']) & (df['IBS'] < 0.3)
        
        # Exit conditions (original OR dynamic stop)
        exit_condition = (df['Close'] > df['Prev_High']) | (df['Close'] < df['SMA_300'])
        
        return self.backtest_strategy(df, entry_condition, exit_condition, f'DynamicStop_{symbol}')
    
    def long_short_strategy(self, qqq_df, psq_df):
        """Long QQQ in bull market, long PSQ in bear market - CORRECTED VERSION"""
        qqq_df = self.calculate_indicators(qqq_df.copy())
        psq_df = self.calculate_indicators(psq_df.copy())

        # Align dates
        common_dates = qqq_df.index.intersection(psq_df.index)
        qqq_df = qqq_df.loc[common_dates]
        psq_df = psq_df.loc[common_dates]

        # Initialize tracking variables
        position = 0  # 0 = flat, 1 = long QQQ, 2 = long PSQ
        trades = []
        daily_returns = []
        positions = []
        entry_price = None
        entry_date = None
        
        for i in range(len(common_dates)):
            date = common_dates[i]
            qqq_price = qqq_df.loc[date, 'Close']
            psq_price = psq_df.loc[date, 'Close']
            
            # Determine market regime based on QQQ
            is_bull_market = qqq_df.loc[date, 'Close'] > qqq_df.loc[date, 'SMA_300']
            
            # Calculate daily return based on current position
            daily_return = 0.0
            if i > 0:  # Skip first day
                prev_date = common_dates[i-1]
                if position == 1:  # Long QQQ
                    daily_return = (qqq_price / qqq_df.loc[prev_date, 'Close']) - 1
                elif position == 2:  # Long PSQ
                    daily_return = (psq_price / psq_df.loc[prev_date, 'Close']) - 1
            
            daily_returns.append(daily_return)
            
            # Exit logic first
            if position == 1:  # Currently long QQQ
                qqq_exit_signal = qqq_df.loc[date, 'Close'] > qqq_df.loc[date, 'Prev_High']
                if qqq_exit_signal:
                    # Exit QQQ position
                    exit_price = qqq_price
                    trade_return = (exit_price / entry_price) - 1
                    duration = (date - entry_date).days
                    
                    trades.append({
                        'Entry_Date': entry_date,
                        'Exit_Date': date,
                        'Entry_Price': entry_price,
                        'Exit_Price': exit_price,
                        'Return': trade_return,
                        'Duration': duration,
                        'Symbol': 'QQQ'
                    })
                    
                    position = 0
                    entry_price = None
                    entry_date = None
                    
            elif position == 2:  # Currently long PSQ
                psq_exit_signal = psq_df.loc[date, 'Close'] > psq_df.loc[date, 'Prev_High']
                if psq_exit_signal:
                    # Exit PSQ position
                    exit_price = psq_price
                    trade_return = (exit_price / entry_price) - 1
                    duration = (date - entry_date).days
                    
                    trades.append({
                        'Entry_Date': entry_date,
                        'Exit_Date': date,
                        'Entry_Price': entry_price,
                        'Exit_Price': exit_price,
                        'Return': trade_return,
                        'Duration': duration,
                        'Symbol': 'PSQ'
                    })
                    
                    position = 0
                    entry_price = None
                    entry_date = None
            
            # Entry logic (only if flat)
            if position == 0:
                if is_bull_market:
                    # Check QQQ entry conditions
                    qqq_entry_signal = (qqq_df.loc[date, 'Close'] < qqq_df.loc[date, 'Lower_Band']) & \
                                      (qqq_df.loc[date, 'IBS'] < 0.3)
                    if qqq_entry_signal and not pd.isna(qqq_df.loc[date, 'Lower_Band']):
                        position = 1
                        entry_price = qqq_price
                        entry_date = date
                else:
                    # Check PSQ entry conditions
                    psq_entry_signal = (psq_df.loc[date, 'Close'] < psq_df.loc[date, 'Lower_Band']) & \
                                      (psq_df.loc[date, 'IBS'] < 0.3)
                    if psq_entry_signal and not pd.isna(psq_df.loc[date, 'Lower_Band']):
                        position = 2
                        entry_price = psq_price
                        entry_date = date
            
            positions.append(position)

        # Create strategy dataframe
        strategy_df = pd.DataFrame({
            'Close': [qqq_df.loc[d, 'Close'] for d in common_dates],  # Use QQQ for reference
            'Position': positions,
            'Returns': daily_returns
        }, index=common_dates)

        # Store trades
        self.trades['LongShort'] = pd.DataFrame(trades)
        
        return self.analyze_performance(strategy_df, pd.Series(daily_returns, index=common_dates), 'LongShort')
    
    def backtest_strategy(self, df, entry_condition, exit_condition, strategy_name):
        """Generic backtesting function"""
        position = 0
        trades = []
        positions = []
        entry_price = 0
        entry_date = None
        
        for i in range(len(df)):
            date = df.index[i]
            price = df['Close'].iloc[i]
            
            if position == 0 and entry_condition.iloc[i]:
                # Enter position
                position = 1
                entry_price = price
                entry_date = date
                
            elif position == 1 and exit_condition.iloc[i]:
                # Exit position
                position = 0
                exit_price = price
                trade_return = (exit_price / entry_price) - 1
                
                trades.append({
                    'Entry_Date': entry_date,
                    'Exit_Date': date,
                    'Entry_Price': entry_price,
                    'Exit_Price': exit_price,
                    'Return': trade_return,
                    'Duration': (date - entry_date).days
                })
                
            positions.append(position)
        
        # Calculate strategy returns
        df['Position'] = positions
        df['Returns'] = df['Close'].pct_change() * df['Position'].shift(1)
        df['Returns'] = df['Returns'].fillna(0)
        
        # Store trades
        self.trades[strategy_name] = pd.DataFrame(trades)
        
        return self.analyze_performance(df, df['Returns'], strategy_name)
    
    def analyze_performance(self, df, returns, strategy_name):
        """Calculate comprehensive performance metrics"""
        
        # Basic metrics
        total_return = (1 + returns).prod() - 1
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        # Downside metrics
        downside_returns = returns[returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = annual_return / downside_volatility if downside_volatility > 0 else 0
        
        # Drawdown analysis
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Find drawdown periods
        in_drawdown = drawdown < -0.01  # 1% threshold
        drawdown_periods = []
        start_dd = None
        
        for i, is_dd in enumerate(in_drawdown):
            if is_dd and start_dd is None:
                start_dd = i
            elif not is_dd and start_dd is not None:
                drawdown_periods.append(i - start_dd)
                start_dd = None
        
        avg_drawdown_duration = np.mean(drawdown_periods) if drawdown_periods else 0
        max_drawdown_duration = max(drawdown_periods) if drawdown_periods else 0
        
        # Exposure time
        if 'Position' in df.columns:
            exposure_time = (df['Position'] > 0).mean()
        else:
            exposure_time = (returns != 0).mean()
        
        # Win rate and trade metrics
        trades_df = self.trades.get(strategy_name, pd.DataFrame())
        if not trades_df.empty:
            win_rate = (trades_df['Return'] > 0).mean()
            avg_win = trades_df[trades_df['Return'] > 0]['Return'].mean()
            avg_loss = trades_df[trades_df['Return'] < 0]['Return'].mean()
            profit_factor = abs(avg_win / avg_loss) if avg_loss < 0 else np.inf
            num_trades = len(trades_df)
            avg_trade_duration = trades_df['Duration'].mean()
        else:
            win_rate = avg_win = avg_loss = profit_factor = num_trades = avg_trade_duration = 0
        
        # MAR ratio (return/max drawdown)
        mar_ratio = annual_return / abs(max_drawdown) if max_drawdown < 0 else np.inf
        
        # Calmar ratio
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown < 0 else np.inf
        
        results = {
            'Strategy': strategy_name,
            'Total_Return_%': total_return * 100,
            'Annual_Return_%': annual_return * 100,
            'Volatility_%': volatility * 100,
            'Sharpe_Ratio': sharpe_ratio,
            'Sortino_Ratio': sortino_ratio,
            'Max_Drawdown_%': max_drawdown * 100,
            'MAR_Ratio': mar_ratio,
            'Calmar_Ratio': calmar_ratio,
            'Exposure_Time_%': exposure_time * 100,
            'Win_Rate_%': win_rate * 100,
            'Avg_Win_%': avg_win * 100 if avg_win else 0,
            'Avg_Loss_%': avg_loss * 100 if avg_loss else 0,
            'Profit_Factor': profit_factor,
            'Num_Trades': num_trades,
            'Avg_Trade_Duration': avg_trade_duration,
            'Max_DD_Duration_Days': max_drawdown_duration,
            'Avg_DD_Duration_Days': avg_drawdown_duration,
            'Cumulative_Returns': cumulative,
            'Drawdown': drawdown,
            'Raw_Returns': returns
        }
        
        self.results[strategy_name] = results
        return results
    
    def run_all_strategies(self):
        """Run all strategy variations"""
        if not self.data:
            self.download_data()
        
        results_summary = []
        
        # Original strategy on QQQ
        if 'QQQ' in self.data:
            print("Running Original Strategy on QQQ...")
            result = self.original_strategy(self.data['QQQ'], 'QQQ')
            results_summary.append(result)
        
        # Market regime filter
        if 'QQQ' in self.data:
            print("Running Market Regime Filter Strategy...")
            result = self.market_regime_filter_strategy(self.data['QQQ'], 'QQQ')
            results_summary.append(result)
        
        # Dynamic stop loss
        if 'QQQ' in self.data:
            print("Running Dynamic Stop Loss Strategy...")
            result = self.dynamic_stop_strategy(self.data['QQQ'], 'QQQ')
            results_summary.append(result)
        
        # Long/Short strategy
        if 'QQQ' in self.data and 'PSQ' in self.data:
            print("Running Long/Short Strategy...")
            result = self.long_short_strategy(self.data['QQQ'], self.data['PSQ'])
            results_summary.append(result)
        
        # Buy and hold benchmarks
        if 'QQQ' in self.data:
            print("Calculating QQQ Buy & Hold...")
            returns = self.data['QQQ']['Close'].pct_change().fillna(0)
            result = self.analyze_performance(self.data['QQQ'], returns, 'BuyHold_QQQ')
            results_summary.append(result)
        
        if 'SPY' in self.data:
            print("Calculating SPY Buy & Hold...")
            returns = self.data['SPY']['Close'].pct_change().fillna(0)
            result = self.analyze_performance(self.data['SPY'], returns, 'BuyHold_SPY')
            results_summary.append(result)
        
        return results_summary
    
    def create_summary_table(self):
        """Create summary performance table"""
        summary_data = []
        
        for strategy_name, result in self.results.items():
            summary_data.append({
                'Strategy': result['Strategy'],
                'Total Return (%)': f"{result['Total_Return_%']:.1f}",
                'Annual Return (%)': f"{result['Annual_Return_%']:.1f}",
                'Volatility (%)': f"{result['Volatility_%']:.1f}",
                'Sharpe Ratio': f"{result['Sharpe_Ratio']:.2f}",
                'Sortino Ratio': f"{result['Sortino_Ratio']:.2f}",
                'Max Drawdown (%)': f"{result['Max_Drawdown_%']:.1f}",
                'MAR Ratio': f"{result['MAR_Ratio']:.2f}",
                'Exposure (%)': f"{result['Exposure_Time_%']:.1f}",
                'Win Rate (%)': f"{result['Win_Rate_%']:.1f}",
                'Profit Factor': f"{result['Profit_Factor']:.2f}",
                'Num Trades': int(result['Num_Trades'])
            })
        
        return pd.DataFrame(summary_data)
    
    def plot_equity_curves(self):
        """Plot equity curves for all strategies"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        # Equity curves
        for strategy_name, result in self.results.items():
            if 'Cumulative_Returns' in result:
                cumulative = result['Cumulative_Returns']
                ax1.plot(cumulative.index, cumulative.values, label=strategy_name, linewidth=2)
        
        ax1.set_title('Equity Curves Comparison', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Cumulative Return')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Drawdown comparison
        for strategy_name, result in self.results.items():
            if 'Drawdown' in result:
                drawdown = result['Drawdown']
                ax2.fill_between(drawdown.index, drawdown.values, 0, alpha=0.6, label=strategy_name)
        
        ax2.set_title('Drawdown Comparison', fontsize=16, fontweight='bold')
        ax2.set_ylabel('Drawdown')
        ax2.set_xlabel('Date')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if self.save_charts:
            plt.savefig(os.path.join(self.chart_folder, 'equity_curves.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved equity curves to {self.chart_folder}/equity_curves.png")
        else:
            plt.show()
        
        return fig
    
    def create_monthly_heatmap(self, strategy_name='DynamicStop_QQQ'):
        """Create monthly returns heatmap"""
        if strategy_name not in self.results:
            print(f"Strategy {strategy_name} not found")
            return None
        
        returns = self.results[strategy_name]['Raw_Returns']
        
        # Convert to monthly returns
        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        monthly_returns = monthly_returns * 100  # Convert to percentage
        
        # Create pivot table for heatmap
        monthly_returns.index = pd.to_datetime(monthly_returns.index)
        pivot_table = monthly_returns.groupby([monthly_returns.index.year, monthly_returns.index.month]).sum().unstack()
        pivot_table.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(pivot_table, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
                   cbar_kws={'label': 'Monthly Return (%)'}, ax=ax)
        ax.set_title(f'Monthly Returns Heatmap - {strategy_name}', fontsize=16, fontweight='bold')
        ax.set_ylabel('Year')
        
        if self.save_charts:
            plt.savefig(os.path.join(self.chart_folder, f'monthly_heatmap_{strategy_name}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved monthly heatmap to {self.chart_folder}/monthly_heatmap_{strategy_name}.png")
        else:
            plt.show()
        
        return fig
    
    def create_multiple_monthly_heatmaps(self):
        """Create monthly heatmaps for key strategies"""
        key_strategies = ['Original_QQQ', 'DynamicStop_QQQ', 'MarketRegime_QQQ', 'LongShort']
        
        for strategy_name in key_strategies:
            if strategy_name in self.results:
                print(f"Creating monthly heatmap for {strategy_name}...")
                self.create_monthly_heatmap(strategy_name)
    
    def create_yearly_returns_analysis(self):
        """Create yearly returns comparison for all strategies"""
        yearly_data = {}
        
        # Calculate yearly returns for each strategy
        for strategy_name, result in self.results.items():
            if 'Raw_Returns' in result:
                returns = result['Raw_Returns']
                yearly_returns = returns.resample('Y').apply(lambda x: (1 + x).prod() - 1)
                yearly_returns = yearly_returns * 100  # Convert to percentage
                yearly_returns.index = yearly_returns.index.year
                yearly_data[strategy_name] = yearly_returns
        
        # Create DataFrame
        yearly_df = pd.DataFrame(yearly_data)
        yearly_df = yearly_df.fillna(0)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        # Plot 1: Yearly returns comparison (line plot)
        for strategy_name in yearly_df.columns:
            if 'BuyHold' not in strategy_name:  # Exclude buy&hold from main plot for clarity
                ax1.plot(yearly_df.index, yearly_df[strategy_name], 
                        marker='o', linewidth=2, label=strategy_name)
        
        # Add buy&hold as reference
        if 'BuyHold_QQQ' in yearly_df.columns:
            ax1.plot(yearly_df.index, yearly_df['BuyHold_QQQ'], 
                    linestyle='--', alpha=0.7, color='gray', label='BuyHold_QQQ')
        
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax1.set_title('Yearly Returns Comparison - All Strategies', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Annual Return (%)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Strategy performance heatmap by year
        strategy_subset = [col for col in yearly_df.columns if 'BuyHold' not in col]
        if strategy_subset:
            yearly_subset = yearly_df[strategy_subset]
            
            # Transpose for heatmap (strategies as rows, years as columns)
            heatmap_data = yearly_subset.T
            
            sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
                       cbar_kws={'label': 'Annual Return (%)'}, ax=ax2)
            ax2.set_title('Strategy Performance Heatmap by Year', fontsize=16, fontweight='bold')
            ax2.set_xlabel('Year')
            ax2.set_ylabel('Strategy')
        
        plt.tight_layout()
        
        if self.save_charts:
            plt.savefig(os.path.join(self.chart_folder, 'yearly_returns_analysis.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved yearly returns analysis to {self.chart_folder}/yearly_returns_analysis.png")
        else:
            plt.show()
        
        # Also create summary statistics
        self.create_yearly_returns_summary(yearly_df)
        
        return yearly_df, fig
    
    def create_yearly_returns_summary(self, yearly_df):
        """Create yearly returns summary statistics table"""
        summary_stats = {}
        
        for strategy in yearly_df.columns:
            returns = yearly_df[strategy].dropna()
            if len(returns) > 0:
                summary_stats[strategy] = {
                    'Avg_Annual_Return_%': returns.mean(),
                    'Std_Annual_Return_%': returns.std(),
                    'Best_Year_%': returns.max(),
                    'Worst_Year_%': returns.min(),
                    'Positive_Years': (returns > 0).sum(),
                    'Negative_Years': (returns < 0).sum(),
                    'Win_Rate_%': (returns > 0).mean() * 100,
                    'Years_Data': len(returns)
                }
        
        summary_df = pd.DataFrame(summary_stats).T
        
        # Save summary to text file
        if self.save_charts:
            summary_file = os.path.join(self.chart_folder, 'yearly_returns_summary.txt')
            with open(summary_file, 'w') as f:
                f.write("YEARLY RETURNS SUMMARY STATISTICS\n")
                f.write("="*50 + "\n\n")
                f.write(summary_df.round(2).to_string())
                f.write("\n\nDefinitions:\n")
                f.write("- Avg_Annual_Return_%: Average yearly return\n")
                f.write("- Std_Annual_Return_%: Standard deviation of yearly returns\n")
                f.write("- Best_Year_%: Best single year performance\n")
                f.write("- Worst_Year_%: Worst single year performance\n")
                f.write("- Positive_Years: Number of years with positive returns\n")
                f.write("- Negative_Years: Number of years with negative returns\n")
                f.write("- Win_Rate_%: Percentage of years with positive returns\n")
                f.write("- Years_Data: Total years of data available\n")
            
            print(f"Saved yearly returns summary to {self.chart_folder}/yearly_returns_summary.txt")
        
        print("\nYearly Returns Summary:")
        print(summary_df.round(2).to_string())
        
        return summary_df
    
    def monte_carlo_simulation(self, strategy_name='DynamicStop_QQQ', num_simulations=1000):
        """Perform Monte Carlo simulation"""
        if strategy_name not in self.results:
            print(f"Strategy {strategy_name} not found")
            return None, None
        
        returns = self.results[strategy_name]['Raw_Returns']
        
        # Parameters for simulation
        num_periods = len(returns)
        
        # Bootstrap simulation
        simulated_returns = []
        final_values = []
        max_drawdowns = []
        
        print(f"Running Monte Carlo simulation with {num_simulations} iterations...")
        
        for i in range(num_simulations):
            if i % 100 == 0:
                print(f"Progress: {i}/{num_simulations}")
                
            # Randomly sample returns with replacement
            sim_returns = np.random.choice(returns.dropna(), size=num_periods, replace=True)
            
            # Calculate cumulative returns
            cumulative = np.cumprod(1 + sim_returns)
            final_values.append(cumulative[-1])
            
            # Calculate max drawdown
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            max_drawdowns.append(np.min(drawdown))
            
            simulated_returns.append(cumulative)
        
        # Convert to arrays
        simulated_returns = np.array(simulated_returns)
        final_values = np.array(final_values)
        max_drawdowns = np.array(max_drawdowns)
        
        # Calculate percentiles
        percentiles = [5, 25, 50, 75, 95]
        final_value_percentiles = np.percentile(final_values, percentiles)
        drawdown_percentiles = np.percentile(max_drawdowns, percentiles)
        
        # Plot results
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Simulation paths
        for i in range(min(100, num_simulations)):
            ax1.plot(simulated_returns[i], alpha=0.1, color='blue')
        
        # Add actual strategy performance
        actual_cumulative = self.results[strategy_name]['Cumulative_Returns']
        ax1.plot(actual_cumulative.values, color='red', linewidth=3, label='Actual Strategy')
        ax1.set_title(f'Monte Carlo Simulation Paths ({num_simulations} simulations)')
        ax1.set_ylabel('Cumulative Return')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Final value distribution
        ax2.hist(final_values, bins=50, alpha=0.7, edgecolor='black')
        ax2.axvline(actual_cumulative.iloc[-1], color='red', linestyle='--', linewidth=2, label='Actual Final Value')
        ax2.set_title('Distribution of Final Values')
        ax2.set_xlabel('Final Portfolio Value')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        
        # Max drawdown distribution
        ax3.hist(max_drawdowns * 100, bins=50, alpha=0.7, edgecolor='black')
        actual_max_dd = self.results[strategy_name]['Max_Drawdown_%']
        ax3.axvline(actual_max_dd, color='red', linestyle='--', linewidth=2, label='Actual Max DD')
        ax3.set_title('Distribution of Maximum Drawdowns')
        ax3.set_xlabel('Maximum Drawdown (%)')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        
        # Percentile analysis
        labels = [f'{p}th' for p in percentiles]
        x_pos = np.arange(len(labels))
        
        ax4.bar(x_pos - 0.2, final_value_percentiles, 0.4, label='Final Values', alpha=0.7)
        ax4_twin = ax4.twinx()
        ax4_twin.bar(x_pos + 0.2, drawdown_percentiles * 100, 0.4, 
                    label='Max Drawdowns', alpha=0.7, color='orange')
        
        ax4.set_xlabel('Percentiles')
        ax4.set_ylabel('Final Value')
        ax4_twin.set_ylabel('Max Drawdown (%)')
        ax4.set_title('Monte Carlo Results by Percentile')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(labels)
        ax4.legend(loc='upper left')
        ax4_twin.legend(loc='upper right')
        
        plt.tight_layout()
        
        if self.save_charts:
            plt.savefig(os.path.join(self.chart_folder, f'monte_carlo_{strategy_name}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved Monte Carlo results to {self.chart_folder}/monte_carlo_{strategy_name}.png")
        else:
            plt.show()
        
        # Summary statistics
        mc_summary = {
            'Strategy': strategy_name,
            'Num_Simulations': num_simulations,
            'Final_Value_Mean': np.mean(final_values),
            'Final_Value_Std': np.std(final_values),
            'Final_Value_5th': final_value_percentiles[0],
            'Final_Value_95th': final_value_percentiles[4],
            'MaxDD_Mean_%': np.mean(max_drawdowns) * 100,
            'MaxDD_Std_%': np.std(max_drawdowns) * 100,
            'MaxDD_5th_%': drawdown_percentiles[0] * 100,
            'MaxDD_95th_%': drawdown_percentiles[4] * 100,
            'Prob_Positive': (final_values > 1).mean() * 100,
            'Prob_Beat_Market': 0  # Would need benchmark for this
        }
        
        return fig, mc_summary
    
    def export_to_excel(self, filename='mean_reversion_strategy_results.xlsx'):
        """Export all results to Excel"""
        with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
            
            # Summary table
            summary_df = self.create_summary_table()
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Individual strategy details
            for strategy_name, result in self.results.items():
                sheet_name = strategy_name[:31]  # Excel sheet name limit
                
                # Create strategy details
                details = pd.DataFrame({
                    'Metric': ['Total Return (%)', 'Annual Return (%)', 'Volatility (%)', 
                              'Sharpe Ratio', 'Sortino Ratio', 'Max Drawdown (%)', 
                              'MAR Ratio', 'Exposure Time (%)', 'Win Rate (%)', 
                              'Avg Win (%)', 'Avg Loss (%)', 'Profit Factor', 
                              'Number of Trades', 'Avg Trade Duration'],
                    'Value': [result['Total_Return_%'], result['Annual_Return_%'], 
                             result['Volatility_%'], result['Sharpe_Ratio'], 
                             result['Sortino_Ratio'], result['Max_Drawdown_%'], 
                             result['MAR_Ratio'], result['Exposure_Time_%'], 
                             result['Win_Rate_%'], result['Avg_Win_%'], 
                             result['Avg_Loss_%'], result['Profit_Factor'], 
                             result['Num_Trades'], result['Avg_Trade_Duration']]
                })
                details.to_excel(writer, sheet_name=sheet_name, index=False)
            
            # Trades for each strategy
            for strategy_name, trades_df in self.trades.items():
                if not trades_df.empty:
                    # Remove timezone info from all datetime columns
                    for col in trades_df.select_dtypes(include=['datetimetz']).columns:
                        trades_df[col] = trades_df[col].dt.tz_localize(None)
                    sheet_name = f'Trades_{strategy_name}'[:31]
                    trades_df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            # Daily returns and cumulative performance
            if self.results:
                performance_data = {}
                for strategy_name, result in self.results.items():
                    if 'Cumulative_Returns' in result:
                        performance_data[f'{strategy_name}_Cumulative'] = result['Cumulative_Returns']
                        performance_data[f'{strategy_name}_Drawdown'] = result['Drawdown']
                
                if performance_data:
                    perf_df = pd.DataFrame(performance_data)
                    
                    # Before exporting perf_df to Excel
                    if isinstance(perf_df.index, pd.DatetimeIndex) and perf_df.index.tz is not None:
                        perf_df.index = perf_df.index.tz_localize(None)

                    perf_df.to_excel(writer, sheet_name='Performance_Time_Series')
        
        print(f"Results exported to {filename}")


class EnhancedRiskAnalysis:
    """Enhanced risk analysis and robustness testing"""
    
    def __init__(self, strategy_results, save_charts=True, chart_folder='qqq_mr'):
        self.results = strategy_results
        self.save_charts = save_charts
        self.chart_folder = chart_folder
        
    def calculate_additional_metrics(self, returns, benchmark_returns=None):
        """Calculate additional risk and performance metrics"""
        
        # Remove NaN values
        returns = returns.dropna()
        
        # Basic stats
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)
        
        # Value at Risk (VaR)
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        
        # Conditional VaR (Expected Shortfall)
        cvar_95 = returns[returns <= var_95].mean()
        cvar_99 = returns[returns <= var_99].mean()
        
        # Maximum consecutive losses
        consecutive_losses = 0
        max_consecutive_losses = 0
        for ret in returns:
            if ret < 0:
                consecutive_losses += 1
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
            else:
                consecutive_losses = 0
        
        # Maximum consecutive gains
        consecutive_gains = 0
        max_consecutive_gains = 0
        for ret in returns:
            if ret > 0:
                consecutive_gains += 1
                max_consecutive_gains = max(max_consecutive_gains, consecutive_gains)
            else:
                consecutive_gains = 0
        
        # Gain-to-Pain ratio
        positive_returns = returns[returns > 0].sum()
        negative_returns = abs(returns[returns < 0].sum())
        gain_to_pain = positive_returns / negative_returns if negative_returns > 0 else np.inf
        
        # Ulcer Index
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = ((cumulative - running_max) / running_max) * 100
        ulcer_index = np.sqrt(np.mean(drawdown ** 2))
        
        # Sterling Ratio (annual return / avg max drawdown)
        annual_return = (1 + returns.mean()) ** 252 - 1
        sterling_ratio = annual_return / (ulcer_index / 100) if ulcer_index > 0 else np.inf
        
        # Recovery factor
        total_return = (1 + returns).prod() - 1
        max_dd = drawdown.min() / 100
        recovery_factor = total_return / abs(max_dd) if max_dd < 0 else np.inf
        
        # Omega ratio (gains above threshold / losses below threshold)
        threshold = 0  # Risk-free rate threshold
        gains = returns[returns > threshold] - threshold
        losses = threshold - returns[returns <= threshold]
        omega_ratio = gains.sum() / losses.sum() if losses.sum() > 0 else np.inf
        
        metrics = {
            'Skewness': skewness,
            'Kurtosis': kurtosis,
            'VaR_95_%': var_95 * 100,
            'VaR_99_%': var_99 * 100,
            'CVaR_95_%': cvar_95 * 100,
            'CVaR_99_%': cvar_99 * 100,
            'Max_Consecutive_Losses': max_consecutive_losses,
            'Max_Consecutive_Gains': max_consecutive_gains,
            'Gain_to_Pain_Ratio': gain_to_pain,
            'Ulcer_Index': ulcer_index,
            'Sterling_Ratio': sterling_ratio,
            'Recovery_Factor': recovery_factor,
            'Omega_Ratio': omega_ratio
        }
        
        # Beta and correlation analysis if benchmark provided
        if benchmark_returns is not None:
            aligned_returns, aligned_benchmark = returns.align(benchmark_returns, join='inner')
            if len(aligned_returns) > 1:
                correlation = aligned_returns.corr(aligned_benchmark)
                
                # Calculate beta using linear regression
                X = aligned_benchmark.values.reshape(-1, 1)
                y = aligned_returns.values
                reg = LinearRegression().fit(X, y)
                beta = reg.coef_[0]
                
                # Treynor ratio
                excess_return = annual_return
                treynor_ratio = excess_return / beta if beta != 0 else np.inf
                
                # Information ratio
                excess_returns = aligned_returns - aligned_benchmark
                tracking_error = excess_returns.std() * np.sqrt(252)
                information_ratio = (excess_returns.mean() * 252) / tracking_error if tracking_error > 0 else np.inf
                
                metrics.update({
                    'Beta': beta,
                    'Correlation': correlation,
                    'Treynor_Ratio': treynor_ratio,
                    'Information_Ratio': information_ratio,
                    'Tracking_Error_%': tracking_error * 100
                })
        
        return metrics
    
    def rolling_performance_analysis(self, strategy_name, window_years=2):
        """Analyze rolling performance metrics"""
        if strategy_name not in self.results:
            return None, None
        
        returns = self.results[strategy_name]['Raw_Returns']
        window_days = window_years * 252
        
        if len(returns) < window_days:
            print(f"Insufficient data for {window_years}-year rolling analysis")
            return None, None
        
        rolling_metrics = []
        
        for i in range(window_days, len(returns)):
            window_returns = returns.iloc[i-window_days:i]
            
            annual_return = (1 + window_returns.mean()) ** 252 - 1
            volatility = window_returns.std() * np.sqrt(252)
            sharpe = annual_return / volatility if volatility > 0 else 0
            
            cumulative = (1 + window_returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()
            
            rolling_metrics.append({
                'Date': returns.index[i],
                'Annual_Return': annual_return,
                'Volatility': volatility,
                'Sharpe_Ratio': sharpe,
                'Max_Drawdown': max_drawdown
            })
        
        rolling_df = pd.DataFrame(rolling_metrics).set_index('Date')
        
        # Plot rolling metrics
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        rolling_df['Annual_Return'].plot(ax=axes[0,0], title=f'{window_years}-Year Rolling Annual Return')
        axes[0,0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[0,0].set_ylabel('Annual Return')
        
        rolling_df['Sharpe_Ratio'].plot(ax=axes[0,1], title=f'{window_years}-Year Rolling Sharpe Ratio')
        axes[0,1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[0,1].set_ylabel('Sharpe Ratio')
        
        rolling_df['Volatility'].plot(ax=axes[1,0], title=f'{window_years}-Year Rolling Volatility')
        axes[1,0].set_ylabel('Volatility')
        
        rolling_df['Max_Drawdown'].plot(ax=axes[1,1], title=f'{window_years}-Year Rolling Max Drawdown')
        axes[1,1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[1,1].set_ylabel('Max Drawdown')
        
        for ax in axes.flatten():
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Rolling Performance Analysis - {strategy_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if self.save_charts:
            plt.savefig(os.path.join(self.chart_folder, f'rolling_analysis_{strategy_name}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved rolling analysis to {self.chart_folder}/rolling_analysis_{strategy_name}.png")
        else:
            plt.show()
        
        return rolling_df, fig
    
    def regime_analysis(self, strategy_name, regime_indicator='VIX'):
        """Analyze performance across different market regimes"""
        if strategy_name not in self.results:
            return None, None
        
        returns = self.results[strategy_name]['Raw_Returns']
        
        # Create market regimes based on rolling volatility as proxy for VIX
        rolling_vol = returns.rolling(30).std() * np.sqrt(252)
        vol_terciles = rolling_vol.quantile([0.33, 0.67])
        
        low_vol_mask = rolling_vol <= vol_terciles.iloc[0]
        med_vol_mask = (rolling_vol > vol_terciles.iloc[0]) & (rolling_vol <= vol_terciles.iloc[1])
        high_vol_mask = rolling_vol > vol_terciles.iloc[1]
        
        regimes = {
            'Low_Volatility': returns[low_vol_mask],
            'Medium_Volatility': returns[med_vol_mask], 
            'High_Volatility': returns[high_vol_mask]
        }
        
        regime_stats = {}
        for regime_name, regime_returns in regimes.items():
            if len(regime_returns) > 0:
                annual_return = (1 + regime_returns.mean()) ** 252 - 1
                volatility = regime_returns.std() * np.sqrt(252)
                sharpe = annual_return / volatility if volatility > 0 else 0
                
                regime_stats[regime_name] = {
                    'Annual_Return_%': annual_return * 100,
                    'Volatility_%': volatility * 100,
                    'Sharpe_Ratio': sharpe,
                    'Win_Rate_%': (regime_returns > 0).mean() * 100,
                    'Avg_Return_%': regime_returns.mean() * 100,
                    'Periods': len(regime_returns)
                }
        
        regime_df = pd.DataFrame(regime_stats).T
        
        # Plot regime analysis
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        regime_df['Annual_Return_%'].plot(kind='bar', ax=axes[0,0], 
                                        title='Annual Return by Regime')
        axes[0,0].set_ylabel('Annual Return (%)')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        regime_df['Sharpe_Ratio'].plot(kind='bar', ax=axes[0,1], 
                                     title='Sharpe Ratio by Regime')
        axes[0,1].set_ylabel('Sharpe Ratio')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        regime_df['Win_Rate_%'].plot(kind='bar', ax=axes[1,0], 
                                   title='Win Rate by Regime')
        axes[1,0].set_ylabel('Win Rate (%)')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        regime_df['Volatility_%'].plot(kind='bar', ax=axes[1,1], 
                                     title='Volatility by Regime')
        axes[1,1].set_ylabel('Volatility (%)')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        for ax in axes.flatten():
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Market Regime Analysis - {strategy_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if self.save_charts:
            plt.savefig(os.path.join(self.chart_folder, f'regime_analysis_{strategy_name}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved regime analysis to {self.chart_folder}/regime_analysis_{strategy_name}.png")
        else:
            plt.show()
        
        return regime_df, fig
    
    def parameter_sensitivity_analysis(self, data, parameter_ranges):
        """Test sensitivity to parameter changes"""
        sensitivity_results = []
        
        # Base parameters
        base_params = {
            'hl_window': 25,
            'high_window': 10, 
            'multiplier': 2.5,
            'ibs_threshold': 0.3,
            'sma_window': 300
        }
        
        print("Running parameter sensitivity analysis...")
        
        for param_name, param_values in parameter_ranges.items():
            print(f"Testing {param_name}: {param_values}")
            for param_value in param_values:
                # Create modified parameters
                test_params = base_params.copy()
                test_params[param_name] = param_value
                
                # Run strategy with modified parameters
                result = self.test_strategy_with_params(data, test_params)
                
                sensitivity_results.append({
                    'Parameter': param_name,
                    'Value': param_value,
                    'Annual_Return_%': result['Annual_Return_%'],
                    'Sharpe_Ratio': result['Sharpe_Ratio'],
                    'Max_Drawdown_%': result['Max_Drawdown_%'],
                    'Num_Trades': result['Num_Trades']
                })
        
        sensitivity_df = pd.DataFrame(sensitivity_results)
        
        # Plot sensitivity analysis
        unique_params = sensitivity_df['Parameter'].unique()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        metrics = ['Annual_Return_%', 'Sharpe_Ratio', 'Max_Drawdown_%', 'Num_Trades']
        
        for i, metric in enumerate(metrics):
            for param in unique_params:
                param_data = sensitivity_df[sensitivity_df['Parameter'] == param]
                axes[i].plot(param_data['Value'], param_data[metric], 
                           marker='o', label=param, linewidth=2)
            
            axes[i].set_title(f'{metric} Sensitivity')
            axes[i].set_ylabel(metric)
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.suptitle('Parameter Sensitivity Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if self.save_charts:
            plt.savefig(os.path.join(self.chart_folder, 'parameter_sensitivity.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved parameter sensitivity to {self.chart_folder}/parameter_sensitivity.png")
        else:
            plt.show()
        
        return sensitivity_df, fig
    
    def test_strategy_with_params(self, data, params):
        """Test strategy with custom parameters"""
        df = data.copy()
        
        # Calculate indicators with custom parameters
        df['HL_Mean'] = (df['High'] - df['Low']).rolling(params['hl_window']).mean()
        df['IBS'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
        df['High_Roll'] = df['High'].rolling(params['high_window']).max()
        df['Lower_Band'] = df['High_Roll'] - params['multiplier'] * df['HL_Mean']
        df['SMA'] = df['Close'].rolling(params['sma_window']).mean()
        df['Prev_High'] = df['High'].shift(1)
        
        # Strategy logic
        entry_condition = (df['Close'] < df['Lower_Band']) & (df['IBS'] < params['ibs_threshold'])
        exit_condition = (df['Close'] > df['Prev_High']) | (df['Close'] < df['SMA'])
        
        # Simple backtesting
        position = 0
        returns = []
        trades = 0
        
        for i in range(len(df)):
            if position == 0 and entry_condition.iloc[i]:
                position = 1
                entry_price = df['Close'].iloc[i]
            elif position == 1 and exit_condition.iloc[i]:
                position = 0
                exit_price = df['Close'].iloc[i]
                trade_return = (exit_price / entry_price) - 1
                returns.append(trade_return)
                trades += 1
        
        if len(returns) > 0:
            returns_series = pd.Series(returns)
            annual_return = (1 + returns_series.mean()) ** 252 - 1
            volatility = returns_series.std() * np.sqrt(252)
            sharpe = annual_return / volatility if volatility > 0 else 0
            
            cumulative = (1 + returns_series).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()
        else:
            annual_return = sharpe = max_drawdown = 0
            
        return {
            'Annual_Return_%': annual_return * 100,
            'Sharpe_Ratio': sharpe,
            'Max_Drawdown_%': max_drawdown * 100,
            'Num_Trades': trades
        }
    
    def stress_testing(self, strategy_name, stress_scenarios):
        """Perform stress testing under various scenarios"""
        if strategy_name not in self.results:
            return None, None
        
        returns = self.results[strategy_name]['Raw_Returns']
        original_cumulative = (1 + returns).cumprod()
        
        stress_results = {}
        
        print("Running stress testing scenarios...")
        
        for scenario_name, scenario_params in stress_scenarios.items():
            print(f"Testing scenario: {scenario_name}")
            stressed_returns = returns.copy()
            
            if scenario_name == 'Market_Crash':
                # Simulate market crash: consecutive negative days
                crash_start = len(stressed_returns) // 2
                crash_duration = scenario_params.get('duration', 20)
                crash_magnitude = scenario_params.get('magnitude', -0.05)
                
                for i in range(crash_duration):
                    if crash_start + i < len(stressed_returns):
                        stressed_returns.iloc[crash_start + i] = crash_magnitude
            
            elif scenario_name == 'High_Volatility':
                # Increase volatility by multiplying returns
                multiplier = scenario_params.get('multiplier', 2.0)
                stressed_returns = stressed_returns * multiplier
            
            elif scenario_name == 'Extended_Bear_Market':
                # Gradual decline over extended period
                bear_start = len(stressed_returns) // 3
                bear_duration = scenario_params.get('duration', 252)  # 1 year
                daily_decline = scenario_params.get('daily_decline', -0.002)
                
                for i in range(bear_duration):
                    if bear_start + i < len(stressed_returns):
                        stressed_returns.iloc[bear_start + i] = daily_decline
            
            # Calculate stressed performance
            stressed_cumulative = (1 + stressed_returns).cumprod()
            stressed_annual_return = (stressed_cumulative.iloc[-1]) ** (252 / len(stressed_returns)) - 1
            stressed_volatility = stressed_returns.std() * np.sqrt(252)
            stressed_sharpe = stressed_annual_return / stressed_volatility if stressed_volatility > 0 else 0
            
            running_max = stressed_cumulative.expanding().max()
            drawdown = (stressed_cumulative - running_max) / running_max
            stressed_max_drawdown = drawdown.min()
            
            stress_results[scenario_name] = {
                'Annual_Return_%': stressed_annual_return * 100,
                'Volatility_%': stressed_volatility * 100,
                'Sharpe_Ratio': stressed_sharpe,
                'Max_Drawdown_%': stressed_max_drawdown * 100,
                'Final_Value': stressed_cumulative.iloc[-1]
            }
        
        # Compare with original
        original_stats = self.results[strategy_name]
        stress_results['Original'] = {
            'Annual_Return_%': original_stats['Annual_Return_%'],
            'Volatility_%': original_stats['Volatility_%'],
            'Sharpe_Ratio': original_stats['Sharpe_Ratio'],
            'Max_Drawdown_%': original_stats['Max_Drawdown_%'],
            'Final_Value': original_cumulative.iloc[-1]
        }
        
        stress_df = pd.DataFrame(stress_results).T
        
        # Plot stress test results
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        stress_df['Annual_Return_%'].plot(kind='bar', ax=axes[0,0], 
                                        title='Annual Return - Stress Scenarios')
        axes[0,0].set_ylabel('Annual Return (%)')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        stress_df['Sharpe_Ratio'].plot(kind='bar', ax=axes[0,1], 
                                     title='Sharpe Ratio - Stress Scenarios')
        axes[0,1].set_ylabel('Sharpe Ratio')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        stress_df['Max_Drawdown_%'].plot(kind='bar', ax=axes[1,0], 
                                       title='Max Drawdown - Stress Scenarios')
        axes[1,0].set_ylabel('Max Drawdown (%)')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        stress_df['Final_Value'].plot(kind='bar', ax=axes[1,1], 
                                    title='Final Portfolio Value - Stress Scenarios')
        axes[1,1].set_ylabel('Final Value')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        for ax in axes.flatten():
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Stress Testing Results - {strategy_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if self.save_charts:
            plt.savefig(os.path.join(self.chart_folder, f'stress_testing_{strategy_name}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved stress testing to {self.chart_folder}/stress_testing_{strategy_name}.png")
        else:
            plt.show()
        
        return stress_df, fig


def run_enhanced_analysis(strategy_results, data, save_charts=True, chart_folder='qqq_mr'):
    """Run comprehensive enhanced analysis"""
    
    enhanced_analyzer = EnhancedRiskAnalysis(strategy_results, save_charts, chart_folder)
    
    print("="*60)
    print("ENHANCED RISK ANALYSIS")
    print("="*60)
    
    # Additional risk metrics for each strategy
    enhanced_metrics = {}
    for strategy_name, result in strategy_results.items():
        if 'Raw_Returns' in result:
            returns = result['Raw_Returns']
            
            # Get benchmark returns for comparison
            benchmark_returns = None
            if 'BuyHold_SPY' in strategy_results:
                benchmark_returns = strategy_results['BuyHold_SPY']['Raw_Returns']
            
            metrics = enhanced_analyzer.calculate_additional_metrics(returns, benchmark_returns)
            enhanced_metrics[strategy_name] = metrics
    
    # Create enhanced metrics table
    enhanced_df = pd.DataFrame(enhanced_metrics).T
    print("\nAdditional Risk Metrics:")
    print(enhanced_df.round(3).to_string())
    
    # Rolling performance analysis
    print("\n" + "="*60)
    print("ROLLING PERFORMANCE ANALYSIS")
    print("="*60)
    
    rolling_df, rolling_fig = enhanced_analyzer.rolling_performance_analysis('DynamicStop_QQQ')
    if rolling_df is not None:
        print("\nRolling performance statistics:")
        print(rolling_df.describe())
    
    # Market regime analysis
    print("\n" + "="*60)
    print("MARKET REGIME ANALYSIS")
    print("="*60)
    
    regime_df, regime_fig = enhanced_analyzer.regime_analysis('DynamicStop_QQQ')
    if regime_df is not None:
        print("\nPerformance by market regime:")
        print(regime_df.to_string())
    
    # Stress testing
    print("\n" + "="*60)
    print("STRESS TESTING")
    print("="*60)
    
    stress_scenarios = {
        'Market_Crash': {'duration': 20, 'magnitude': -0.05},
        'High_Volatility': {'multiplier': 2.0},
        'Extended_Bear_Market': {'duration': 252, 'daily_decline': -0.002}
    }
    
    stress_df, stress_fig = enhanced_analyzer.stress_testing('DynamicStop_QQQ', stress_scenarios)
    if stress_df is not None:
        print("\nStress testing results:")
        print(stress_df.to_string())
    
    # Parameter sensitivity analysis
    print("\n" + "="*60)
    print("PARAMETER SENSITIVITY ANALYSIS")
    print("="*60)
    
    if 'QQQ' in data:
        parameter_ranges = {
            'hl_window': [20, 25, 30, 35],
            'multiplier': [2.0, 2.5, 3.0, 3.5],
            'ibs_threshold': [0.2, 0.3, 0.4, 0.5],
            'sma_window': [200, 250, 300, 350]
        }
        
        sensitivity_df, sensitivity_fig = enhanced_analyzer.parameter_sensitivity_analysis(
            data['QQQ'], parameter_ranges)
        if sensitivity_df is not None:
            print("\nParameter sensitivity results:")
            print(sensitivity_df.groupby('Parameter').agg({
                'Annual_Return_%': ['mean', 'std'],
                'Sharpe_Ratio': ['mean', 'std'],
                'Max_Drawdown_%': ['mean', 'std']
            }).round(3))
    
    return {
        'enhanced_metrics': enhanced_df,
        'rolling_analysis': rolling_df if 'rolling_df' in locals() else None,
        'regime_analysis': regime_df if 'regime_df' in locals() else None,
        'stress_testing': stress_df if 'stress_df' in locals() else None,
        'sensitivity_analysis': sensitivity_df if 'sensitivity_df' in locals() else None
    }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    print("="*80)
    print("MEAN REVERSION STRATEGY BACKTESTING SUITE")
    print("Based on Quantitativo's 2.11 Sharpe Strategy")
    print("="*80)
    
    # Initialize strategy with chart saving
    strategy = MeanReversionStrategy(save_charts=True, chart_folder='qqq_mr')
    
    # Download data
    print("\nStep 1: Downloading Data")
    print("-" * 30)
    data = strategy.download_data(['QQQ', 'SPY', 'PSQ'], start_date='1999-01-01')
    
    # Run all strategies
    print("\nStep 2: Running All Strategies")
    print("-" * 30)
    results_summary = strategy.run_all_strategies()
    
    # Display summary table
    print("\nStep 3: Performance Summary")
    print("-" * 30)
    summary_table = strategy.create_summary_table()
    print(summary_table.to_string(index=False))
    
    # Create visualizations (saved to folder)
    print("\nStep 4: Creating Visualizations")
    print("-" * 30)
    
    # Equity curves
    print("Creating equity curves...")
    equity_fig = strategy.plot_equity_curves()
    
    # Monthly heatmaps for multiple strategies
    print("Creating monthly heatmaps...")
    strategy.create_multiple_monthly_heatmaps()
    
    # Yearly returns analysis
    print("Creating yearly returns analysis...")
    yearly_df, yearly_fig = strategy.create_yearly_returns_analysis()
    
    # Monte Carlo simulation
    print("\nStep 5: Monte Carlo Simulation")
    print("-" * 30)
    print("Running Monte Carlo simulation...")
    mc_fig, mc_summary = strategy.monte_carlo_simulation('DynamicStop_QQQ', num_simulations=1000)
    if mc_summary:
        print("\nMonte Carlo Summary:")
        for key, value in mc_summary.items():
            if isinstance(value, float):
                print(f"{key}: {value:.3f}")
            else:
                print(f"{key}: {value}")
    
    # Enhanced analysis (charts saved to folder)
    print("\nStep 6: Enhanced Risk Analysis")
    print("-" * 30)
    enhanced_results = run_enhanced_analysis(strategy.results, strategy.data, 
                                           save_charts=True, chart_folder='qqq_mr')
    
    # Export to Excel
    print("\nStep 7: Exporting Results")
    print("-" * 30)
    strategy.export_to_excel('complete_mean_reversion_results.xlsx')
    
    # Final summary
    print("\n" + "="*80)
    print("BACKTESTING COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\nGenerated Files:")
    print("- complete_mean_reversion_results.xlsx (All data and results)")
    print(f"- Charts saved in folder: {strategy.chart_folder}/")
    print("\nStrategy Performance Highlights:")
    
    # Show best performing strategy
    best_sharpe_strategy = None
    best_sharpe = 0
    
    for strategy_name, result in strategy.results.items():
        if 'BuyHold' not in strategy_name and result['Sharpe_Ratio'] > best_sharpe:
            best_sharpe = result['Sharpe_Ratio']
            best_sharpe_strategy = strategy_name
    
    if best_sharpe_strategy:
        result = strategy.results[best_sharpe_strategy]
        print(f"\nBest Performing Strategy: {best_sharpe_strategy}")
        print(f"- Annual Return: {result['Annual_Return_%']:.1f}%")
        print(f"- Sharpe Ratio: {result['Sharpe_Ratio']:.2f}")
        print(f"- Max Drawdown: {result['Max_Drawdown_%']:.1f}%")
        print(f"- Win Rate: {result['Win_Rate_%']:.1f}%")
        print(f"- Number of Trades: {int(result['Num_Trades'])}")
        print(f"- Exposure Time: {result['Exposure_Time_%']:.1f}%")
    
    print(f"\nBenchmark Comparison (Buy & Hold):")
    if 'BuyHold_QQQ' in strategy.results:
        qqq_result = strategy.results['BuyHold_QQQ']
        print(f"QQQ: {qqq_result['Annual_Return_%']:.1f}% return, {qqq_result['Sharpe_Ratio']:.2f} Sharpe")
    
    if 'BuyHold_SPY' in strategy.results:
        spy_result = strategy.results['BuyHold_SPY']
        print(f"SPY: {spy_result['Annual_Return_%']:.1f}% return, {spy_result['Sharpe_Ratio']:.2f} Sharpe")
    
    # Print long/short strategy details
    if 'LongShort' in strategy.results:
        ls_result = strategy.results['LongShort']
        print(f"\nLong/Short Strategy Results:")
        print(f"- Annual Return: {ls_result['Annual_Return_%']:.1f}%")
        print(f"- Sharpe Ratio: {ls_result['Sharpe_Ratio']:.2f}")
        print(f"- Max Drawdown: {ls_result['Max_Drawdown_%']:.1f}%")
        print(f"- Number of Trades: {int(ls_result['Num_Trades'])}")
        
        if 'LongShort' in strategy.trades and not strategy.trades['LongShort'].empty:
            trades_by_symbol = strategy.trades['LongShort']['Symbol'].value_counts()
            print(f"- QQQ Trades: {trades_by_symbol.get('QQQ', 0)}")
            print(f"- PSQ Trades: {trades_by_symbol.get('PSQ', 0)}")
    
    print(f"\nAnalysis complete! Check the Excel file and {strategy.chart_folder}/ folder for detailed results.")