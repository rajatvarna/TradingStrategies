import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
from scipy import stats
import os
warnings.filterwarnings('ignore')

class MAStrategyBacktest:
    def __init__(self, tickers, ma_period=10, commission=0.005, slippage=0.001, output_folder='backtest_results'):
        """
        Initialize the backtesting system
        
        Parameters:
        -----------
        tickers : list
            List of ticker symbols
        ma_period : int or dict
            Moving average period in months. Can be:
            - int: same period for all tickers
            - dict: different period per ticker, e.g. {'SPY': 10, 'BTC-USD': 6}
        commission : float
            Commission per share in dollars
        slippage : float
            Slippage as percentage (0.001 = 0.1%)
        output_folder : str
            Folder to save all output files
        """
        self.tickers = tickers
        
        # Handle both int and dict for ma_period
        if isinstance(ma_period, dict):
            self.ma_period = ma_period
        else:
            # Convert single value to dict for all tickers
            self.ma_period = {ticker: ma_period for ticker in tickers}
        
        self.commission = commission
        self.slippage = slippage
        self.output_folder = output_folder
        self.data = {}
        self.results = {}
        self.trades = {}
        self.buy_hold_results = {}
        
        # Create output folder if it doesn't exist
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
            print(f"Created output folder: {self.output_folder}")
        
    def download_data(self):
        """Download historical data from Yahoo Finance"""
        print("Downloading data from Yahoo Finance...")
        
        for ticker in self.tickers:
            try:
                # Download from earliest available date
                stock = yf.Ticker(ticker)
                df = stock.history(start='2000-01-01', end=datetime.now().strftime('%Y-%m-%d'))
                
                if df.empty:
                    print(f"No data available for {ticker}")
                    continue
                
                # Use Close for calculations (already adjusted by yfinance)
                df = df[['Close']].copy()
                df = df.dropna()
                
                print(f"{ticker}: {len(df)} days from {df.index[0].date()} to {df.index[-1].date()}")
                self.data[ticker] = df
                
            except Exception as e:
                print(f"Error downloading {ticker}: {e}")
    
    def calculate_signals(self):
        """Calculate MA and generate trading signals using ticker-specific MA periods"""
        print("\nCalculating signals...")
        
        for ticker in self.data.keys():
            df = self.data[ticker].copy()
            
            # Get MA period for this ticker
            ma_period = self.ma_period[ticker]
            
            # Calculate MA (approximately 21 trading days per month)
            df['MA'] = df['Close'].rolling(window=ma_period * 21).mean()
            
            # Generate signals: 1 = Long, 0 = Flat
            df['Signal'] = 0
            df.loc[df['Close'] > df['MA'], 'Signal'] = 1
            
            # Position changes
            df['Position'] = df['Signal'].diff()
            
            print(f"  {ticker}: Using {ma_period}-month MA ({ma_period * 21} days)")
            
            self.data[ticker] = df
    
    def backtest_strategy(self):
        """Backtest the strategy and calculate returns with costs"""
        print("\nBacktesting strategy...")
        
        for ticker in self.data.keys():
            df = self.data[ticker].copy()
            
            # Calculate daily returns
            df['Returns'] = df['Close'].pct_change()
            
            # Calculate transaction costs
            # Assume 100 shares per position
            shares = 100
            df['Transaction_Cost'] = 0.0
            
            # Entry cost (commission + slippage)
            entry_mask = df['Position'] == 1
            df.loc[entry_mask, 'Transaction_Cost'] = (self.commission * shares + 
                                                       df.loc[entry_mask, 'Close'] * shares * self.slippage)
            
            # Exit cost (commission + slippage)
            exit_mask = df['Position'] == -1
            df.loc[exit_mask, 'Transaction_Cost'] = (self.commission * shares + 
                                                      df.loc[exit_mask, 'Close'] * shares * self.slippage)
            
            # Calculate cost as percentage of position value
            df['Cost_Pct'] = 0.0
            df.loc[entry_mask | exit_mask, 'Cost_Pct'] = df.loc[entry_mask | exit_mask, 'Transaction_Cost'] / (
                df.loc[entry_mask | exit_mask, 'Close'] * shares)
            
            # Strategy returns (shifted by 1 to avoid lookahead bias)
            df['Strategy_Returns'] = df['Signal'].shift(1) * df['Returns'] - df['Cost_Pct']
            
            # Cumulative returns
            df['Cumulative_Returns'] = (1 + df['Returns']).cumprod()
            df['Strategy_Cumulative'] = (1 + df['Strategy_Returns']).cumprod()
            
            # Calculate equity curve (starting with $10,000)
            df['Equity'] = 10000 * df['Strategy_Cumulative']
            df['BH_Equity'] = 10000 * df['Cumulative_Returns']
            
            self.data[ticker] = df
            
    def extract_trades(self):
        """Extract individual trades from the backtest"""
        print("\nExtracting trades...")
        
        for ticker in self.data.keys():
            df = self.data[ticker].copy()
            trades = []
            
            entry_date = None
            entry_price = None
            shares = 100
            
            for i in range(len(df)):
                # Entry signal
                if df['Position'].iloc[i] == 1:
                    entry_date = df.index[i]
                    entry_price = df['Close'].iloc[i]
                    # Apply slippage and commission at entry
                    entry_cost = self.commission * shares + entry_price * shares * self.slippage
                    entry_price_adj = entry_price * (1 + self.slippage) + self.commission
                
                # Exit signal
                elif df['Position'].iloc[i] == -1 and entry_date is not None:
                    exit_date = df.index[i]
                    exit_price = df['Close'].iloc[i]
                    # Apply slippage and commission at exit
                    exit_cost = self.commission * shares + exit_price * shares * self.slippage
                    exit_price_adj = exit_price * (1 - self.slippage) - self.commission
                    
                    # Calculate P&L with costs
                    pnl_pct = (exit_price_adj - entry_price_adj) / entry_price_adj * 100
                    pnl_dollar = (exit_price_adj - entry_price_adj) * shares
                    
                    trades.append({
                        'Ticker': ticker,
                        'Entry_Date': entry_date,
                        'Entry_Price': entry_price,
                        'Exit_Date': exit_date,
                        'Exit_Price': exit_price,
                        'Entry_Price_Adj': entry_price_adj,
                        'Exit_Price_Adj': exit_price_adj,
                        'PnL_%': pnl_pct,
                        'PnL_$': pnl_dollar,
                        'Duration_Days': (exit_date - entry_date).days,
                        'Trade_Type': 'Long'
                    })
                    
                    entry_date = None
                    entry_price = None
            
            self.trades[ticker] = pd.DataFrame(trades)
    
    def calculate_statistics(self):
        """Calculate comprehensive performance statistics"""
        print("\nCalculating statistics...")
        
        for ticker in self.data.keys():
            df = self.data[ticker].copy()
            trades_df = self.trades[ticker]
            
            # Filter out NaN values
            strategy_returns = df['Strategy_Returns'].dropna()
            returns = df['Returns'].dropna()
            
            # Basic stats
            total_return = (df['Equity'].iloc[-1] / 10000 - 1) * 100
            bh_return = (df['BH_Equity'].iloc[-1] / 10000 - 1) * 100
            
            # Number of years
            years = (df.index[-1] - df.index[0]).days / 365.25
            
            # CAGR
            cagr = ((df['Equity'].iloc[-1] / 10000) ** (1/years) - 1) * 100
            bh_cagr = ((df['BH_Equity'].iloc[-1] / 10000) ** (1/years) - 1) * 100
            
            # Drawdown calculation
            equity_curve = df['Equity']
            running_max = equity_curve.expanding().max()
            drawdown = (equity_curve - running_max) / running_max * 100
            max_drawdown = drawdown.min()
            avg_drawdown = drawdown[drawdown < 0].mean() if len(drawdown[drawdown < 0]) > 0 else 0
            
            # Drawdown duration
            is_drawdown = drawdown < 0
            drawdown_periods = is_drawdown.astype(int).groupby((is_drawdown != is_drawdown.shift()).cumsum()).sum()
            max_drawdown_duration = drawdown_periods.max() if len(drawdown_periods) > 0 else 0
            
            bh_equity_curve = df['BH_Equity']
            bh_running_max = bh_equity_curve.expanding().max()
            bh_drawdown = (bh_equity_curve - bh_running_max) / bh_running_max * 100
            bh_max_drawdown = bh_drawdown.min()
            
            # Trade statistics
            if len(trades_df) > 0:
                winning_trades = trades_df[trades_df['PnL_%'] > 0]
                losing_trades = trades_df[trades_df['PnL_%'] < 0]
                
                win_rate = len(winning_trades) / len(trades_df) * 100
                avg_win = winning_trades['PnL_%'].mean() if len(winning_trades) > 0 else 0
                avg_loss = losing_trades['PnL_%'].mean() if len(losing_trades) > 0 else 0
                
                best_trade = trades_df['PnL_%'].max()
                worst_trade = trades_df['PnL_%'].min()
                
                total_wins = winning_trades['PnL_%'].sum() if len(winning_trades) > 0 else 0
                total_losses = abs(losing_trades['PnL_%'].sum()) if len(losing_trades) > 0 else 0
                profit_factor = total_wins / total_losses if total_losses != 0 else np.inf
                
                avg_trade_duration = trades_df['Duration_Days'].mean()
                
                # Expectancy
                expectancy = (win_rate/100 * avg_win) + ((1 - win_rate/100) * avg_loss)
            else:
                win_rate = avg_win = avg_loss = profit_factor = avg_trade_duration = 0
                best_trade = worst_trade = expectancy = 0
                winning_trades = losing_trades = pd.DataFrame()
            
            # Sharpe Ratio (annualized, assuming 252 trading days)
            if strategy_returns.std() != 0:
                sharpe = (strategy_returns.mean() / strategy_returns.std()) * np.sqrt(252)
            else:
                sharpe = 0
            
            if returns.std() != 0:
                bh_sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
            else:
                bh_sharpe = 0
            
            # Sortino Ratio (using downside deviation)
            downside_returns = strategy_returns[strategy_returns < 0]
            if len(downside_returns) > 0 and downside_returns.std() != 0:
                sortino = (strategy_returns.mean() / downside_returns.std()) * np.sqrt(252)
            else:
                sortino = 0
            
            bh_downside_returns = returns[returns < 0]
            if len(bh_downside_returns) > 0 and bh_downside_returns.std() != 0:
                bh_sortino = (returns.mean() / bh_downside_returns.std()) * np.sqrt(252)
            else:
                bh_sortino = 0
            
            # Calmar Ratio (CAGR / abs(Max Drawdown))
            calmar = abs(cagr / max_drawdown) if max_drawdown != 0 else 0
            bh_calmar = abs(bh_cagr / bh_max_drawdown) if bh_max_drawdown != 0 else 0
            
            # MAR Ratio (same as Calmar)
            mar = calmar
            bh_mar = bh_calmar
            
            # Omega Ratio (threshold = 0)
            threshold = 0
            returns_above = strategy_returns[strategy_returns > threshold].sum()
            returns_below = abs(strategy_returns[strategy_returns < threshold].sum())
            omega = returns_above / returns_below if returns_below != 0 else np.inf
            
            bh_returns_above = returns[returns > threshold].sum()
            bh_returns_below = abs(returns[returns < threshold].sum())
            bh_omega = bh_returns_above / bh_returns_below if bh_returns_below != 0 else np.inf
            
            # Exposure (% of time in market)
            exposure = (df['Signal'].sum() / len(df)) * 100
            
            # Volatility (annualized)
            volatility = strategy_returns.std() * np.sqrt(252) * 100
            bh_volatility = returns.std() * np.sqrt(252) * 100
            
            # Skewness and Kurtosis
            skew = stats.skew(strategy_returns.dropna())
            kurt = stats.kurtosis(strategy_returns.dropna())
            
            bh_skew = stats.skew(returns.dropna())
            bh_kurt = stats.kurtosis(returns.dropna())
            
            # Value at Risk (VaR) and Conditional VaR (CVaR) at 95%
            var_95 = np.percentile(strategy_returns.dropna(), 5) * np.sqrt(252) * 100
            cvar_95 = strategy_returns[strategy_returns <= np.percentile(strategy_returns, 5)].mean() * np.sqrt(252) * 100
            
            bh_var_95 = np.percentile(returns.dropna(), 5) * np.sqrt(252) * 100
            bh_cvar_95 = returns[returns <= np.percentile(returns, 5)].mean() * np.sqrt(252) * 100
            
            # Store results
            self.results[ticker] = {
                'MA_Period_Months': self.ma_period[ticker],
                'Total_Return_%': total_return,
                'BH_Return_%': bh_return,
                'CAGR_%': cagr,
                'BH_CAGR_%': bh_cagr,
                'Max_Drawdown_%': max_drawdown,
                'Avg_Drawdown_%': avg_drawdown,
                'Max_DD_Duration_Days': max_drawdown_duration,
                'BH_Max_Drawdown_%': bh_max_drawdown,
                'Sharpe_Ratio': sharpe,
                'BH_Sharpe_Ratio': bh_sharpe,
                'Sortino_Ratio': sortino,
                'BH_Sortino_Ratio': bh_sortino,
                'Calmar_Ratio': calmar,
                'BH_Calmar_Ratio': bh_calmar,
                'MAR_Ratio': mar,
                'BH_MAR_Ratio': bh_mar,
                'Omega_Ratio': omega,
                'BH_Omega_Ratio': bh_omega,
                'Win_Rate_%': win_rate,
                'Avg_Win_%': avg_win,
                'Avg_Loss_%': avg_loss,
                'Best_Trade_%': best_trade,
                'Worst_Trade_%': worst_trade,
                'Expectancy_%': expectancy,
                'Profit_Factor': profit_factor,
                'Total_Trades': len(trades_df),
                'Winning_Trades': len(winning_trades),
                'Losing_Trades': len(losing_trades),
                'Avg_Trade_Duration_Days': avg_trade_duration,
                'Exposure_%': exposure,
                'Volatility_%': volatility,
                'BH_Volatility_%': bh_volatility,
                'Skewness': skew,
                'Kurtosis': kurt,
                'BH_Skewness': bh_skew,
                'BH_Kurtosis': bh_kurt,
                'VaR_95_%': var_95,
                'CVaR_95_%': cvar_95,
                'BH_VaR_95_%': bh_var_95,
                'BH_CVaR_95_%': bh_cvar_95,
                'Years': years,
                'Start_Date': df.index[0].date(),
                'End_Date': df.index[-1].date()
            }
    
    def calculate_monthly_returns(self):
        """Calculate monthly and yearly returns"""
        print("\nCalculating monthly returns...")
        
        self.monthly_returns = {}
        self.yearly_returns = {}
        
        for ticker in self.data.keys():
            df = self.data[ticker].copy()
            
            # Resample to monthly
            monthly = df['Strategy_Returns'].resample('M').apply(lambda x: (1 + x).prod() - 1) * 100
            monthly_df = pd.DataFrame(monthly)
            monthly_df['Year'] = monthly_df.index.year
            monthly_df['Month'] = monthly_df.index.month
            
            # Pivot to create heatmap format
            heatmap_data = monthly_df.pivot(index='Year', columns='Month', values='Strategy_Returns')
            heatmap_data.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            
            self.monthly_returns[ticker] = heatmap_data
            
            # Yearly returns
            yearly_strategy = df.groupby(df.index.year)['Strategy_Returns'].apply(lambda x: (1 + x).prod() - 1) * 100
            yearly_bh = df.groupby(df.index.year)['Returns'].apply(lambda x: (1 + x).prod() - 1) * 100
            
            self.yearly_returns[ticker] = pd.DataFrame({
                'Strategy': yearly_strategy,
                'Buy_Hold': yearly_bh
            })
    
    def optimize_parameters(self, ma_range=range(6, 15)):
        """Optimize MA period parameter"""
        print(f"\nOptimizing MA period from {min(ma_range)} to {max(ma_range)} months...")
        
        self.optimization_results = {}
        
        for ticker in self.data.keys():
            print(f"  Optimizing {ticker}...")
            opt_results = []
            
            for ma_period in ma_range:
                # Create temporary backtest with this MA period
                temp_backtest = MAStrategyBacktest([ticker], ma_period={ticker: ma_period}, 
                                                   commission=self.commission, slippage=self.slippage,
                                                   output_folder=self.output_folder)
                temp_backtest.data[ticker] = self.data[ticker][['Close']].copy()
                
                # Run backtest
                temp_backtest.calculate_signals()
                temp_backtest.backtest_strategy()
                temp_backtest.extract_trades()
                
                df = temp_backtest.data[ticker]
                trades_df = temp_backtest.trades[ticker]
                
                # Calculate key metrics
                years = (df.index[-1] - df.index[0]).days / 365.25
                cagr = ((df['Equity'].iloc[-1] / 10000) ** (1/years) - 1) * 100
                
                equity_curve = df['Equity']
                running_max = equity_curve.expanding().max()
                drawdown = (equity_curve - running_max) / running_max * 100
                max_dd = drawdown.min()
                
                sharpe = 0
                strategy_returns = df['Strategy_Returns'].dropna()
                if len(strategy_returns) > 0 and strategy_returns.std() != 0:
                    sharpe = (strategy_returns.mean() / strategy_returns.std()) * np.sqrt(252)
                
                calmar = abs(cagr / max_dd) if max_dd != 0 else 0
                
                win_rate = 0
                if len(trades_df) > 0:
                    win_rate = len(trades_df[trades_df['PnL_%'] > 0]) / len(trades_df) * 100
                
                opt_results.append({
                    'MA_Period': ma_period,
                    'CAGR_%': cagr,
                    'Max_DD_%': max_dd,
                    'Sharpe': sharpe,
                    'Calmar': calmar,
                    'Win_Rate_%': win_rate,
                    'Total_Trades': len(trades_df),
                    'Final_Equity': df['Equity'].iloc[-1]
                })
            
            self.optimization_results[ticker] = pd.DataFrame(opt_results)
    
    def monte_carlo_simulation(self, n_simulations=1000):
        """Run Monte Carlo simulation for robustness testing"""
        print(f"\nRunning Monte Carlo simulation ({n_simulations} iterations)...")
        
        self.mc_results = {}
        
        for ticker in self.data.keys():
            trades_df = self.trades[ticker]
            
            if len(trades_df) == 0:
                print(f"  Skipping {ticker} (no trades)")
                continue
            
            print(f"  Simulating {ticker}...")
            
            # Get trade returns
            trade_returns = trades_df['PnL_%'].values
            
            # Run simulations
            final_returns = []
            max_drawdowns = []
            sharpe_ratios = []
            calmar_ratios = []
            
            for _ in range(n_simulations):
                # Random sampling with replacement
                simulated_trades = np.random.choice(trade_returns, size=len(trade_returns), replace=True)
                
                # Calculate cumulative return
                cum_returns = (1 + simulated_trades/100).cumprod()
                final_return = (cum_returns[-1] - 1) * 100
                final_returns.append(final_return)
                
                # Calculate max drawdown
                running_max = np.maximum.accumulate(cum_returns)
                drawdown = (cum_returns - running_max) / running_max * 100
                max_dd = drawdown.min()
                max_drawdowns.append(max_dd)
                
                # Calculate Sharpe
                if simulated_trades.std() != 0:
                    sharpe = (simulated_trades.mean() / simulated_trades.std()) * np.sqrt(12)
                    sharpe_ratios.append(sharpe)
                
                # Calculate Calmar
                years = len(trades_df) / 12  # Approximate
                sim_cagr = ((cum_returns[-1]) ** (1/years) - 1) * 100
                calmar = abs(sim_cagr / max_dd) if max_dd != 0 else 0
                calmar_ratios.append(calmar)
            
            # Risk of Ruin (probability of losing more than 20%)
            risk_of_ruin = len([r for r in final_returns if r < -20]) / len(final_returns) * 100
            
            self.mc_results[ticker] = {
                'Final_Returns': final_returns,
                'Max_Drawdowns': max_drawdowns,
                'Sharpe_Ratios': sharpe_ratios,
                'Calmar_Ratios': calmar_ratios,
                'Mean_Return': np.mean(final_returns),
                'Median_Return': np.median(final_returns),
                'Std_Return': np.std(final_returns),
                'Min_Return': np.min(final_returns),
                'Max_Return': np.max(final_returns),
                'Percentile_5': np.percentile(final_returns, 5),
                'Percentile_95': np.percentile(final_returns, 95),
                'Mean_MaxDD': np.mean(max_drawdowns),
                'Worst_MaxDD': np.min(max_drawdowns),
                'Mean_Sharpe': np.mean(sharpe_ratios) if sharpe_ratios else 0,
                'Mean_Calmar': np.mean(calmar_ratios) if calmar_ratios else 0,
                'Risk_of_Ruin_%': risk_of_ruin
            }
    
    def plot_equity_curves(self):
        """Plot equity curves for all assets"""
        print("\nGenerating equity curves...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for idx, ticker in enumerate(self.data.keys()):
            df = self.data[ticker]
            ma_period = self.ma_period[ticker]
            
            axes[idx].plot(df.index, df['Equity'], label='Strategy', linewidth=2, color='#2E86AB')
            axes[idx].plot(df.index, df['BH_Equity'], label='Buy & Hold', linewidth=2, alpha=0.7, color='#A23B72')
            axes[idx].set_title(f'{ticker} - Equity Curve ({ma_period}-Month MA)', fontsize=14, fontweight='bold')
            axes[idx].set_xlabel('Date')
            axes[idx].set_ylabel('Portfolio Value ($)')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
            axes[idx].set_yscale('log')
            
            # Add final values
            final_strategy = df['Equity'].iloc[-1]
            final_bh = df['BH_Equity'].iloc[-1]
            axes[idx].text(0.02, 0.98, f'Strategy: ${final_strategy:,.0f}\nBuy&Hold: ${final_bh:,.0f}', 
                          transform=axes[idx].transAxes, verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        filepath = os.path.join(self.output_folder, 'equity_curves.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_monthly_heatmaps(self):
        """Plot monthly return heatmaps"""
        print("\nGenerating monthly heatmaps...")
        
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, ticker in enumerate(self.monthly_returns.keys()):
            data = self.monthly_returns[ticker]
            
            sns.heatmap(data, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
                       ax=axes[idx], cbar_kws={'label': 'Return (%)'}, 
                       linewidths=0.5, linecolor='gray')
            axes[idx].set_title(f'{ticker} - Monthly Returns Heatmap (%)', fontsize=14, fontweight='bold')
            axes[idx].set_xlabel('Month')
            axes[idx].set_ylabel('Year')
        
        plt.tight_layout()
        filepath = os.path.join(self.output_folder, 'monthly_heatmaps.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_optimization_results(self):
        """Plot parameter optimization results"""
        if not hasattr(self, 'optimization_results'):
            return
        
        print("\nGenerating optimization charts...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for idx, ticker in enumerate(self.optimization_results.keys()):
            opt_df = self.optimization_results[ticker]
            
            ax = axes[idx]
            ax2 = ax.twinx()
            
            # Plot CAGR and Sharpe
            line1 = ax.plot(opt_df['MA_Period'], opt_df['CAGR_%'], 'o-', 
                           label='CAGR', color='#2E86AB', linewidth=2, markersize=8)
            line2 = ax2.plot(opt_df['MA_Period'], opt_df['Sharpe'], 's-', 
                            label='Sharpe', color='#A23B72', linewidth=2, markersize=8)
            
            ax.set_xlabel('MA Period (months)', fontsize=12)
            ax.set_ylabel('CAGR (%)', fontsize=12, color='#2E86AB')
            ax2.set_ylabel('Sharpe Ratio', fontsize=12, color='#A23B72')
            ax.set_title(f'{ticker} - Parameter Optimization', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='y', labelcolor='#2E86AB')
            ax2.tick_params(axis='y', labelcolor='#A23B72')
            
            # Highlight best CAGR
            best_idx = opt_df['CAGR_%'].idxmax()
            best_ma = opt_df.loc[best_idx, 'MA_Period']
            ax.axvline(x=best_ma, color='green', linestyle='--', alpha=0.5, linewidth=2)
            
            # Legend
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc='upper left')
        
        plt.tight_layout()
        filepath = os.path.join(self.output_folder, 'optimization_results.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_monte_carlo_distributions(self):
        """Plot Monte Carlo simulation distributions"""
        if not hasattr(self, 'mc_results'):
            return
        
        print("\nGenerating Monte Carlo distributions...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for idx, ticker in enumerate(self.mc_results.keys()):
            mc_data = self.mc_results[ticker]
            
            axes[idx].hist(mc_data['Final_Returns'], bins=50, alpha=0.7, color='#2E86AB', edgecolor='black')
            axes[idx].axvline(mc_data['Mean_Return'], color='red', linestyle='--', 
                             linewidth=2, label=f"Mean: {mc_data['Mean_Return']:.1f}%")
            axes[idx].axvline(mc_data['Percentile_5'], color='orange', linestyle='--', 
                             linewidth=2, label=f"5th %ile: {mc_data['Percentile_5']:.1f}%")
            axes[idx].axvline(mc_data['Percentile_95'], color='green', linestyle='--', 
                             linewidth=2, label=f"95th %ile: {mc_data['Percentile_95']:.1f}%")
            
            axes[idx].set_title(f'{ticker} - Monte Carlo Return Distribution', fontsize=14, fontweight='bold')
            axes[idx].set_xlabel('Total Return (%)')
            axes[idx].set_ylabel('Frequency')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        filepath = os.path.join(self.output_folder, 'monte_carlo_distributions.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
    
    def export_to_excel(self, filename='10MA_Strategy_Complete_Results.xlsx'):
        """Export all results to Excel"""
        filepath = os.path.join(self.output_folder, filename)
        print(f"\nExporting results to {filepath}...")
        
        with pd.ExcelWriter(filepath, engine='xlsxwriter') as writer:
            workbook = writer.book
            
            # Format definitions
            header_format = workbook.add_format({
                'bold': True,
                'bg_color': '#4472C4',
                'font_color': 'white',
                'border': 1
            })
            
            # Summary Statistics
            summary_df = pd.DataFrame(self.results).T
            summary_df.to_excel(writer, sheet_name='Summary_Stats')
            
            worksheet = writer.sheets['Summary_Stats']
            for col_num, value in enumerate(summary_df.columns.values):
                worksheet.write(0, col_num + 1, value, header_format)
            
            # Individual Trades
            for ticker in self.trades.keys():
                if len(self.trades[ticker]) > 0:
                    trades_df = self.trades[ticker].copy()
                    # Remove timezone info from datetime columns
                    if 'Entry_Date' in trades_df.columns:
                        trades_df['Entry_Date'] = trades_df['Entry_Date'].dt.tz_localize(None)
                    if 'Exit_Date' in trades_df.columns:
                        trades_df['Exit_Date'] = trades_df['Exit_Date'].dt.tz_localize(None)
                    trades_df.to_excel(writer, sheet_name=f'{ticker}_Trades', index=False)
            
            # Monthly Returns
            for ticker in self.monthly_returns.keys():
                monthly_df = self.monthly_returns[ticker].copy()
                # Remove timezone info from index if present
                if hasattr(monthly_df.index, 'tz') and monthly_df.index.tz is not None:
                    monthly_df.index = monthly_df.index.tz_localize(None)
                monthly_df.to_excel(writer, sheet_name=f'{ticker}_Monthly')
            
            # Yearly Returns Comparison
            for ticker in self.yearly_returns.keys():
                yearly_df = self.yearly_returns[ticker].copy()
                # Remove timezone info from index if present
                if hasattr(yearly_df.index, 'tz') and yearly_df.index.tz is not None:
                    yearly_df.index = yearly_df.index.tz_localize(None)
                yearly_df.to_excel(writer, sheet_name=f'{ticker}_Yearly')
            
            # Monte Carlo Results
            if hasattr(self, 'mc_results'):
                mc_summary = pd.DataFrame({
                    ticker: {
                        'Mean_Return_%': self.mc_results[ticker]['Mean_Return'],
                        'Median_Return_%': self.mc_results[ticker]['Median_Return'],
                        'Std_Return_%': self.mc_results[ticker]['Std_Return'],
                        'Min_Return_%': self.mc_results[ticker]['Min_Return'],
                        'Max_Return_%': self.mc_results[ticker]['Max_Return'],
                        '5th_Percentile_%': self.mc_results[ticker]['Percentile_5'],
                        '95th_Percentile_%': self.mc_results[ticker]['Percentile_95'],
                        'Mean_MaxDD_%': self.mc_results[ticker]['Mean_MaxDD'],
                        'Worst_MaxDD_%': self.mc_results[ticker]['Worst_MaxDD'],
                        'Mean_Sharpe': self.mc_results[ticker]['Mean_Sharpe'],
                        'Mean_Calmar': self.mc_results[ticker]['Mean_Calmar'],
                        'Risk_of_Ruin_%': self.mc_results[ticker]['Risk_of_Ruin_%']
                    } for ticker in self.mc_results.keys()
                }).T
                mc_summary.to_excel(writer, sheet_name='Monte_Carlo_Stats')
            
            # Optimization Results
            if hasattr(self, 'optimization_results'):
                for ticker in self.optimization_results.keys():
                    self.optimization_results[ticker].to_excel(writer, sheet_name=f'{ticker}_Optimization', index=False)
            
            # Full time series data
            for ticker in self.data.keys():
                df = self.data[ticker][['Close', 'MA', 'Signal', 'Equity', 'BH_Equity']].copy()
                # Remove timezone info from index if present
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                df.to_excel(writer, sheet_name=f'{ticker}_TimeSeries')
        
        print(f"✓ Results saved to {filename}")
    
    def print_summary(self):
        """Print summary statistics"""
        print("\n" + "="*90)
        print("BACKTEST SUMMARY - MOVING AVERAGE STRATEGY")
        print(f"Transaction Costs: ${self.commission}/share + {self.slippage*100}% slippage")
        print("="*90)
        
        for ticker in self.results.keys():
            stats = self.results[ticker]
            ma_period = self.ma_period[ticker]
            print(f"\n{ticker} ({ma_period}-Month MA):")
            print(f"  Period: {stats['Start_Date']} to {stats['End_Date']} ({stats['Years']:.1f} years)")
            print(f"\n  STRATEGY PERFORMANCE:")
            print(f"    Total Return:        {stats['Total_Return_%']:>10.2f}%")
            print(f"    CAGR:                {stats['CAGR_%']:>10.2f}%")
            print(f"    Max Drawdown:        {stats['Max_Drawdown_%']:>10.2f}%")
            print(f"    Avg Drawdown:        {stats['Avg_Drawdown_%']:>10.2f}%")
            print(f"    Max DD Duration:     {stats['Max_DD_Duration_Days']:>10.0f} days")
            print(f"    Sharpe Ratio:        {stats['Sharpe_Ratio']:>10.2f}")
            print(f"    Sortino Ratio:       {stats['Sortino_Ratio']:>10.2f}")
            print(f"    Calmar Ratio:        {stats['Calmar_Ratio']:>10.2f}")
            print(f"    Omega Ratio:         {stats['Omega_Ratio']:>10.2f}")
            print(f"    Volatility:          {stats['Volatility_%']:>10.2f}%")
            print(f"    Skewness:            {stats['Skewness']:>10.2f}")
            print(f"    Kurtosis:            {stats['Kurtosis']:>10.2f}")
            print(f"    VaR (95%):           {stats['VaR_95_%']:>10.2f}%")
            print(f"    CVaR (95%):          {stats['CVaR_95_%']:>10.2f}%")
            
            print(f"\n  BUY & HOLD PERFORMANCE:")
            print(f"    Total Return:        {stats['BH_Return_%']:>10.2f}%")
            print(f"    CAGR:                {stats['BH_CAGR_%']:>10.2f}%")
            print(f"    Max Drawdown:        {stats['BH_Max_Drawdown_%']:>10.2f}%")
            print(f"    Sharpe Ratio:        {stats['BH_Sharpe_Ratio']:>10.2f}")
            print(f"    Sortino Ratio:       {stats['BH_Sortino_Ratio']:>10.2f}")
            print(f"    Calmar Ratio:        {stats['BH_Calmar_Ratio']:>10.2f}")
            print(f"    Omega Ratio:         {stats['BH_Omega_Ratio']:>10.2f}")
            print(f"    Volatility:          {stats['BH_Volatility_%']:>10.2f}%")
            
            print(f"\n  TRADE STATISTICS:")
            print(f"    Total Trades:        {stats['Total_Trades']:>10}")
            print(f"    Winning Trades:      {stats['Winning_Trades']:>10}")
            print(f"    Losing Trades:       {stats['Losing_Trades']:>10}")
            print(f"    Win Rate:            {stats['Win_Rate_%']:>10.2f}%")
            print(f"    Avg Win:             {stats['Avg_Win_%']:>10.2f}%")
            print(f"    Avg Loss:            {stats['Avg_Loss_%']:>10.2f}%")
            print(f"    Best Trade:          {stats['Best_Trade_%']:>10.2f}%")
            print(f"    Worst Trade:         {stats['Worst_Trade_%']:>10.2f}%")
            print(f"    Expectancy:          {stats['Expectancy_%']:>10.2f}%")
            print(f"    Profit Factor:       {stats['Profit_Factor']:>10.2f}")
            print(f"    Avg Duration:        {stats['Avg_Trade_Duration_Days']:>10.1f} days")
            print(f"    Market Exposure:     {stats['Exposure_%']:>10.2f}%")
        
        # Print optimization summary
        if hasattr(self, 'optimization_results'):
            print("\n" + "="*90)
            print("PARAMETER OPTIMIZATION SUMMARY")
            print("="*90)
            for ticker in self.optimization_results.keys():
                opt_df = self.optimization_results[ticker]
                best_idx = opt_df['CAGR_%'].idxmax()
                best_ma = opt_df.loc[best_idx, 'MA_Period']
                best_cagr = opt_df.loc[best_idx, 'CAGR_%']
                best_sharpe = opt_df.loc[best_idx, 'Sharpe']
                print(f"\n{ticker}:")
                print(f"  Best MA Period: {best_ma} months")
                print(f"  Best CAGR: {best_cagr:.2f}%")
                print(f"  Sharpe at Best: {best_sharpe:.2f}")
        
        # Print Monte Carlo summary
        if hasattr(self, 'mc_results'):
            print("\n" + "="*90)
            print("MONTE CARLO SIMULATION SUMMARY (1000 iterations)")
            print("="*90)
            for ticker in self.mc_results.keys():
                mc = self.mc_results[ticker]
                print(f"\n{ticker}:")
                print(f"  Mean Return:         {mc['Mean_Return']:>10.2f}%")
                print(f"  Median Return:       {mc['Median_Return']:>10.2f}%")
                print(f"  5th Percentile:      {mc['Percentile_5']:>10.2f}%")
                print(f"  95th Percentile:     {mc['Percentile_95']:>10.2f}%")
                print(f"  Mean Max DD:         {mc['Mean_MaxDD']:>10.2f}%")
                print(f"  Worst Max DD:        {mc['Worst_MaxDD']:>10.2f}%")
                print(f"  Risk of Ruin (>20%): {mc['Risk_of_Ruin_%']:>10.2f}%")
        
        print("\n" + "="*90)

# Run the backtest
if __name__ == "__main__":
    # Define tickers
    tickers = ['SPY', 'QQQ', 'BTC-USD', 'GLD']  # GLD for Gold ETF
    
    # Define MA periods for each ticker
    ma_periods = {
        'SPY': 10,
        'QQQ': 10,
        'BTC-USD': 6,  # 6 months for Bitcoin
        'GLD': 8       # 8 months for Gold
    }
    
    # Initialize backtester with transaction costs, custom MA periods, and output folder
    output_folder = 'backtest_results'
    backtest = MAStrategyBacktest(tickers, ma_period=ma_periods, commission=0.005, 
                                  slippage=0.001, output_folder=output_folder)
    
    print(f"\n{'='*90}")
    print("STRATEGY CONFIGURATION:")
    print(f"{'='*90}")
    for ticker in tickers:
        print(f"  {ticker}: {ma_periods[ticker]}-month MA")
    print(f"{'='*90}\n")
    
    # Run complete analysis
    backtest.download_data()
    backtest.calculate_signals()
    backtest.backtest_strategy()
    backtest.extract_trades()
    backtest.calculate_statistics()
    backtest.calculate_monthly_returns()
    
    # Parameter optimization (6-14 months) - optimize around the chosen periods
    backtest.optimize_parameters(ma_range=range(6, 15))
    
    # Monte Carlo simulation
    backtest.monte_carlo_simulation(n_simulations=1000)
    
    # Generate visualizations
    backtest.plot_equity_curves()
    backtest.plot_monthly_heatmaps()
    backtest.plot_optimization_results()
    backtest.plot_monte_carlo_distributions()
    
    # Print results
    backtest.print_summary()
    
    # Export to Excel
    backtest.export_to_excel()
    
    print(f"\n✓ Backtest complete! All files saved to '{output_folder}/' folder:")
    print(f"  - {output_folder}/10MA_Strategy_Complete_Results.xlsx")
    print(f"  - {output_folder}/equity_curves.png")
    print(f"  - {output_folder}/monthly_heatmaps.png")
    print(f"  - {output_folder}/optimization_results.png")
    print(f"  - {output_folder}/monte_carlo_distributions.png")