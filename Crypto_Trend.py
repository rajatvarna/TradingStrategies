import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Statistical and analysis libraries
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# For Excel export
import xlsxwriter

class CryptoTrendStrategy:
    def __init__(self, symbols=None, benchmark_symbols=None, start_date='2014-01-01', 
                 transaction_cost=0.001, target_volatility=0.30, max_leverage=2.0,
                 fast_ma=50, slow_ma=55):
        """
        Initialize the Crypto Trend Trading Strategy
        
        Parameters:
        -----------
        symbols : list
            List of cryptocurrency symbols to trade
        benchmark_symbols : list
            List of benchmark symbols for comparison
        start_date : str
            Start date for backtesting
        transaction_cost : float
            Transaction cost as decimal (0.001 = 10 basis points)
        target_volatility : float
            Target annualized volatility for position sizing
        max_leverage : float
            Maximum leverage allowed (2.0 = 200%)
        fast_ma : int
            Fast moving average period
        slow_ma : int
            Slow moving average period
        """
        
        # Default cryptocurrency symbols (top liquid cryptos)
        if symbols is None:
            symbols = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD', 
                      'SOL-USD', 'DOGE-USD', 'DOT-USD', 'MATIC-USD', 'LTC-USD',
                      'AVAX-USD', 'LINK-USD', 'ATOM-USD', 'NEAR-USD', 'ALGO-USD',
                      'VET-USD', 'FIL-USD', 'MANA-USD', 'SAND-USD', 'CRV-USD']
        
        # Default benchmark symbols
        if benchmark_symbols is None:
            benchmark_symbols = ['SPY', 'QQQ', 'BTC-USD']
        
        self.symbols = symbols
        self.benchmark_symbols = benchmark_symbols
        self.start_date = start_date
        self.end_date = datetime.now().strftime('%Y-%m-%d')
        self.transaction_cost = transaction_cost
        self.target_volatility = target_volatility
        self.max_leverage = max_leverage
        self.fast_ma = fast_ma
        self.slow_ma = slow_ma
        
        # Initialize data containers
        self.data = {}
        self.benchmark_data = {}
        self.signals = {}
        self.positions = {}
        self.returns = {}
        self.portfolio_equity = None
        self.trades = []
        self.performance_metrics = {}
        
    def download_data(self):
        """Download price data for all symbols"""
        print("Downloading cryptocurrency data...")
        
        # Download crypto data
        successful_downloads = 0
        for symbol in self.symbols:
            try:
                print(f"Downloading {symbol}...")
                data = yf.download(symbol, start=self.start_date, end=self.end_date, progress=False)
                
                if data.empty:
                    print(f"✗ No data returned for {symbol}")
                    continue
                
                # Handle multi-level columns if present
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
                
                # Ensure we have the required columns
                required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                if not all(col in data.columns for col in required_cols):
                    print(f"✗ Missing required columns for {symbol}")
                    continue
                
                # Clean the data
                data = data.dropna(subset=['Open', 'High', 'Low', 'Close'])
                
                # Check if we have sufficient data for MA calculation
                min_required = max(self.fast_ma, self.slow_ma) + 50  # Extra buffer
                if len(data) < min_required:
                    print(f"✗ Insufficient data for {symbol}: {len(data)} < {min_required}")
                    continue
                
                # Ensure proper data types
                for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    data[col] = pd.to_numeric(data[col], errors='coerce')
                
                self.data[symbol] = data
                successful_downloads += 1
                print(f"✓ Successfully processed {symbol}: {len(data)} rows")
                
            except Exception as e:
                print(f"✗ Error downloading {symbol}: {e}")
        
        # Download benchmark data
        print(f"\nDownloading benchmark data...")
        for symbol in self.benchmark_symbols:
            try:
                data = yf.download(symbol, start=self.start_date, end=self.end_date, progress=False)
                
                if not data.empty:
                    # Handle multi-level columns
                    if isinstance(data.columns, pd.MultiIndex):
                        data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
                    
                    data = data.dropna(subset=['Close'])
                    self.benchmark_data[symbol] = data
                    print(f"✓ Downloaded benchmark {symbol}: {len(data)} rows")
                    
            except Exception as e:
                print(f"✗ Error downloading benchmark {symbol}: {e}")
        
        print(f"\nSuccessfully downloaded {successful_downloads} cryptocurrencies and {len(self.benchmark_data)} benchmarks")
        
        if successful_downloads == 0:
            raise ValueError("No cryptocurrency data was successfully downloaded!")
    
    def calculate_ma_crossover_signals(self, data):
        """Calculate Moving Average Crossover signals"""
        
        # Calculate moving averages
        fast_ma = data['Close'].rolling(window=self.fast_ma).mean()
        slow_ma = data['Close'].rolling(window=self.slow_ma).mean()
        
        # Initialize signals
        signals = pd.Series(0.0, index=data.index)
        
        # Calculate crossover signals
        # Entry: Fast MA crosses above Slow MA
        # Exit: Fast MA crosses below Slow MA
        
        # Identify crossover points
        ma_diff = fast_ma - slow_ma
        ma_diff_prev = ma_diff.shift(1)
        
        # Bullish crossover: previous diff was negative, current diff is positive
        bullish_cross = (ma_diff_prev < 0) & (ma_diff > 0)
        
        # Bearish crossover: previous diff was positive, current diff is negative  
        bearish_cross = (ma_diff_prev > 0) & (ma_diff < 0)
        
        # Track position state
        in_position = False
        
        for i in range(max(self.fast_ma, self.slow_ma), len(data)):
            
            # Check for valid data
            if pd.isna(fast_ma.iloc[i]) or pd.isna(slow_ma.iloc[i]):
                continue
            
            # Check for bullish crossover (entry signal)
            if bullish_cross.iloc[i]:
                signals.iloc[i] = 1.0
                in_position = True
                
            # Check for bearish crossover (exit signal)
            elif bearish_cross.iloc[i]:
                signals.iloc[i] = 0.0
                in_position = False
                
            # Maintain current position if no crossover
            elif in_position:
                signals.iloc[i] = 1.0
            else:
                signals.iloc[i] = 0.0
        
        return signals, fast_ma, slow_ma
    
    def calculate_volatility_position_size(self, returns, target_vol=None):
        """Calculate position size based on volatility targeting"""
        if target_vol is None:
            target_vol = self.target_volatility
        
        # Calculate 30-day rolling volatility (annualized)
        vol_window = min(30, len(returns))
        rolling_vol = returns.rolling(window=vol_window).std() * np.sqrt(252)
        
        # Avoid division by zero
        rolling_vol = rolling_vol.fillna(method='bfill').fillna(0.1)
        rolling_vol = np.maximum(rolling_vol, 0.05)  # Minimum 5% vol assumption
        
        # Position size = target_vol / realized_vol
        position_sizes = target_vol / rolling_vol
        
        # Cap at maximum leverage
        position_sizes = np.minimum(position_sizes, self.max_leverage)
        
        return position_sizes
    
    def generate_asset_strategy(self, symbol):
        """Generate trading strategy for a single asset using MA crossover"""
        try:
            data = self.data[symbol].copy()
            
            # Calculate returns
            data['Returns'] = data['Close'].pct_change()
            
            # Generate MA crossover signals
            raw_signals, fast_ma, slow_ma = self.calculate_ma_crossover_signals(data)
            
            # Calculate position sizes based on volatility
            position_sizes = self.calculate_volatility_position_size(data['Returns'])
            
            # Final positions = signal * position_size
            positions = raw_signals * position_sizes
            
            # Calculate strategy returns
            strategy_returns = positions.shift(1) * data['Returns']
            
            # Store results including MA data for analysis
            self.signals[symbol] = raw_signals
            self.positions[symbol] = positions
            self.returns[symbol] = strategy_returns
            
            # Store MA data for plotting/analysis
            data['Fast_MA'] = fast_ma
            data['Slow_MA'] = slow_ma
            self.data[symbol] = data
            
            # Calculate some basic stats
            total_trades = (raw_signals.diff() != 0).sum()
            long_periods = raw_signals.sum()
            avg_position = positions.mean()
            
            print(f"✓ {symbol}: {total_trades:.0f} trades, {long_periods:.0f} long days, avg position: {avg_position:.2f}")
            
            return strategy_returns
            
        except Exception as e:
            print(f"✗ Error processing {symbol}: {e}")
            return pd.Series(dtype=float)
    
    def calculate_portfolio_performance(self):
        """Calculate equal-weighted portfolio performance"""
        print("\nCalculating portfolio performance...")
        
        strategy_returns_dict = {}
        successful_assets = 0
        
        # Generate strategies for each asset
        for symbol in self.data.keys():
            asset_returns = self.generate_asset_strategy(symbol)
            if len(asset_returns.dropna()) > 0:
                strategy_returns_dict[symbol] = asset_returns
                successful_assets += 1
        
        if successful_assets == 0:
            raise ValueError("No assets generated valid strategy returns")
        
        print(f"Successfully processed {successful_assets} assets")
        
        # Create portfolio returns DataFrame
        returns_df = pd.DataFrame(strategy_returns_dict)
        returns_df = returns_df.dropna(how='all')
        
        # Equal-weighted portfolio (each asset gets 1/n weight)
        portfolio_weight = 1.0 / successful_assets
        portfolio_returns = returns_df.sum(axis=1) * portfolio_weight
        
        # Apply transaction costs (simplified)
        # Calculate turnover as sum of absolute position changes
        total_turnover = 0
        for symbol in self.positions.keys():
            if len(self.positions[symbol]) > 0:
                position_changes = self.positions[symbol].diff().abs()
                total_turnover += position_changes.sum()
        
        # Apply transaction costs to portfolio
        avg_daily_turnover = total_turnover / len(portfolio_returns) / successful_assets
        daily_tc = avg_daily_turnover * self.transaction_cost
        portfolio_returns_net = portfolio_returns - daily_tc
        
        # Calculate equity curve
        equity_curve = (1 + portfolio_returns_net).cumprod()
        
        # Store results
        self.portfolio_returns = portfolio_returns_net
        self.portfolio_equity = equity_curve
        
        return portfolio_returns_net, equity_curve
    
    def calculate_performance_metrics(self, returns, equity_curve):
        """Calculate comprehensive performance metrics"""
        
        # Basic metrics
        total_return = equity_curve.iloc[-1] - 1
        n_years = len(returns) / 252
        cagr = (equity_curve.iloc[-1] ** (1/n_years)) - 1 if n_years > 0 else 0
        
        # Risk metrics
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = (returns.mean() * 252) / volatility if volatility > 0 else 0
        
        # Downside metrics
        downside_returns = returns[returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else volatility
        sortino_ratio = (returns.mean() * 252) / downside_volatility if downside_volatility > 0 else 0
        
        # Drawdown metrics
        rolling_max = equity_curve.expanding().max()
        drawdown = (equity_curve - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # MAR ratio (CAGR / abs(Max Drawdown))
        mar_ratio = cagr / abs(max_drawdown) if max_drawdown != 0 else np.nan
        
        # Win rate and profit metrics
        winning_days = returns[returns > 0]
        losing_days = returns[returns < 0]
        
        win_rate = len(winning_days) / len(returns[returns != 0]) if len(returns[returns != 0]) > 0 else 0
        avg_win = winning_days.mean() if len(winning_days) > 0 else 0
        avg_loss = losing_days.mean() if len(losing_days) > 0 else 0
        profit_factor = abs(avg_win * len(winning_days)) / abs(avg_loss * len(losing_days)) if len(losing_days) > 0 and avg_loss != 0 else np.nan
        
        # Calculate average exposure
        total_exposure = 0
        for symbol in self.positions.keys():
            if len(self.positions[symbol]) > 0:
                total_exposure += self.positions[symbol].abs().mean()
        avg_exposure = total_exposure / len(self.positions) if len(self.positions) > 0 else 0
        
        # Calmar ratio
        calmar_ratio = cagr / abs(max_drawdown) if max_drawdown != 0 else np.nan
        
        metrics = {
            'Total Return': total_return,
            'CAGR': cagr,
            'Volatility': volatility,
            'Sharpe Ratio': sharpe_ratio,
            'Sortino Ratio': sortino_ratio,
            'Calmar Ratio': calmar_ratio,
            'Max Drawdown': max_drawdown,
            'MAR Ratio': mar_ratio,
            'Win Rate': win_rate,
            'Avg Win %': avg_win,
            'Avg Loss %': avg_loss,
            'Profit Factor': profit_factor,
            'Exposure %': avg_exposure,
            'Total Trades': len(returns[returns != 0]),
            'Best Day': returns.max(),
            'Worst Day': returns.min(),
            'Skewness': stats.skew(returns.dropna()),
            'Kurtosis': stats.kurtosis(returns.dropna())
        }
        
        return metrics
    
    def calculate_benchmark_performance(self):
        """Calculate performance metrics for benchmark assets"""
        benchmark_metrics = {}
        
        for symbol in self.benchmark_data.keys():
            try:
                data = self.benchmark_data[symbol]
                returns = data['Close'].pct_change().dropna()
                
                # Align with strategy period if possible
                if hasattr(self, 'portfolio_returns') and len(self.portfolio_returns) > 0:
                    start_date = self.portfolio_returns.index[0]
                    end_date = self.portfolio_returns.index[-1]
                    
                    # Find overlap
                    common_dates = returns.index.intersection(pd.date_range(start_date, end_date))
                    if len(common_dates) > 0:
                        returns = returns.loc[common_dates]
                
                if len(returns) > 0:
                    equity_curve = (1 + returns).cumprod()
                    metrics = self.calculate_performance_metrics(returns, equity_curve)
                    benchmark_metrics[symbol] = metrics
                
            except Exception as e:
                print(f"Error calculating benchmark {symbol}: {e}")
        
        return benchmark_metrics
    
    def run_monte_carlo_simulation(self, n_simulations=1000):
        """Run Monte Carlo simulation for robustness testing"""
        print(f"Running {n_simulations} Monte Carlo simulations...")
        
        if not hasattr(self, 'portfolio_returns') or len(self.portfolio_returns) == 0:
            print("Error: Portfolio returns not calculated. Run backtest first.")
            return None, None
        
        returns = self.portfolio_returns.dropna()
        
        if len(returns) < 50:
            print("Insufficient data for Monte Carlo simulation")
            return None, None
        
        # Parameters for simulation
        n_days = len(returns)
        mu = returns.mean()
        sigma = returns.std()
        
        # Run simulations
        final_returns = []
        max_drawdowns = []
        sharpe_ratios = []
        
        for i in range(n_simulations):
            # Generate random returns
            simulated_returns = np.random.normal(mu, sigma, n_days)
            
            # Calculate equity curve
            equity_curve = (1 + pd.Series(simulated_returns)).cumprod()
            
            # Calculate metrics
            total_return = equity_curve.iloc[-1] - 1
            final_returns.append(total_return)
            
            # Max drawdown
            rolling_max = equity_curve.expanding().max()
            drawdown = (equity_curve - rolling_max) / rolling_max
            max_dd = drawdown.min()
            max_drawdowns.append(max_dd)
            
            # Sharpe ratio
            sharpe = (simulated_returns.mean() * 252) / (simulated_returns.std() * np.sqrt(252)) if simulated_returns.std() > 0 else 0
            sharpe_ratios.append(sharpe)
        
        # Calculate confidence intervals
        mc_results = {
            'Mean Return': np.mean(final_returns),
            'Std Return': np.std(final_returns),
            'Return 5%': np.percentile(final_returns, 5),
            'Return 95%': np.percentile(final_returns, 95),
            'Mean Max DD': np.mean(max_drawdowns),
            'Max DD 5%': np.percentile(max_drawdowns, 5),
            'Max DD 95%': np.percentile(max_drawdowns, 95),
            'Mean Sharpe': np.mean(sharpe_ratios),
            'Sharpe 5%': np.percentile(sharpe_ratios, 5),
            'Sharpe 95%': np.percentile(sharpe_ratios, 95),
            'Probability of Loss': np.mean([r < 0 for r in final_returns]),
        }
        
        return mc_results, {
            'returns': final_returns,
            'max_drawdowns': max_drawdowns,
            'sharpe_ratios': sharpe_ratios
        }
    
    def create_monthly_returns_heatmap(self, returns):
        """Create monthly returns heatmap data"""
        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        monthly_returns.index = pd.to_datetime(monthly_returns.index)
        monthly_returns_pct = monthly_returns * 100
        
        # Create year-month matrix
        monthly_returns_pct = monthly_returns_pct.to_frame('Returns')
        monthly_returns_pct['Year'] = monthly_returns_pct.index.year
        monthly_returns_pct['Month'] = monthly_returns_pct.index.month
        
        heatmap_data = monthly_returns_pct.pivot(index='Year', columns='Month', values='Returns')
        
        # Set month names
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        heatmap_data.columns = [month_names[i-1] for i in heatmap_data.columns if i <= 12]
        
        return heatmap_data
    
    def plot_results(self):
        """Create comprehensive performance plots"""
        
        if not hasattr(self, 'portfolio_equity') or len(self.portfolio_equity) == 0:
            print("Error: No portfolio data to plot. Run backtest first.")
            return None, None
        
        # Create main performance plot
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Equity Curves Comparison', 'Drawdown Analysis', 
                           'Monthly Returns Distribution', 'Rolling Sharpe Ratio'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. Equity curves comparison
        fig.add_trace(
            go.Scatter(x=self.portfolio_equity.index, y=self.portfolio_equity.values,
                      name='Crypto Trend Strategy', line=dict(color='blue', width=2)),
            row=1, col=1
        )
        
        # Add benchmark equity curves
        colors = ['red', 'green', 'orange']
        for i, (symbol, data) in enumerate(self.benchmark_data.items()):
            try:
                returns = data['Close'].pct_change().dropna()
                # Align with strategy period
                common_dates = returns.index.intersection(self.portfolio_equity.index)
                if len(common_dates) > 10:
                    returns = returns.loc[common_dates]
                    equity = (1 + returns).cumprod()
                    
                    fig.add_trace(
                        go.Scatter(x=equity.index, y=equity.values,
                                  name=f'{symbol} Buy & Hold', 
                                  line=dict(color=colors[i % len(colors)], width=1)),
                        row=1, col=1
                    )
            except Exception as e:
                print(f"Error plotting {symbol}: {e}")
        
        # 2. Drawdown analysis
        rolling_max = self.portfolio_equity.expanding().max()
        drawdown = (self.portfolio_equity - rolling_max) / rolling_max * 100
        
        fig.add_trace(
            go.Scatter(x=drawdown.index, y=drawdown.values,
                      name='Drawdown %', fill='tonexty',
                      line=dict(color='red')),
            row=2, col=1
        )
        
        # 3. Return distribution
        fig.add_trace(
            go.Histogram(x=self.portfolio_returns.dropna() * 100,
                        name='Daily Returns %', nbinsx=50),
            row=2, col=2
        )
        
        # 4. Rolling Sharpe ratio
        rolling_sharpe = (self.portfolio_returns.rolling(60).mean() * 252) / \
                        (self.portfolio_returns.rolling(60).std() * np.sqrt(252))
        
        fig.add_trace(
            go.Scatter(x=rolling_sharpe.index, y=rolling_sharpe.values,
                      name='Rolling Sharpe (60D)', line=dict(color='purple')),
            row=1, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="Crypto Trend Strategy - Performance Analysis",
            showlegend=True,
            height=800,
            width=1200
        )
        
        # Monthly returns heatmap (separate plot)
        try:
            heatmap_data = self.create_monthly_returns_heatmap(self.portfolio_returns)
            
            fig2 = go.Figure(data=go.Heatmap(
                z=heatmap_data.values,
                x=heatmap_data.columns,
                y=heatmap_data.index,
                colorscale='RdYlGn',
                colorbar=dict(title="Returns %"),
                text=np.round(heatmap_data.values, 2),
                texttemplate="%{text}%",
                textfont={"size": 10}
            ))
            
            fig2.update_layout(
                title="Monthly Returns Heatmap (%)",
                xaxis_title="Month",
                yaxis_title="Year",
                height=400,
                width=800
            )
        except Exception as e:
            print(f"Error creating heatmap: {e}")
            fig2 = None
        
        return fig, fig2
    
    def export_to_excel(self, filename="crypto_trend_strategy_results.xlsx"):
        """Export all results to Excel workbook"""
        print(f"Exporting results to {filename}...")
        
        try:
            with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
                
                # 1. Performance Summary
                if hasattr(self, 'performance_metrics'):
                    summary_df = pd.DataFrame([self.performance_metrics]).T
                    summary_df.columns = ['Strategy']
                    
                    # Add benchmark metrics
                    benchmark_metrics = self.calculate_benchmark_performance()
                    for symbol, metrics in benchmark_metrics.items():
                        summary_df[f'{symbol}_BH'] = pd.Series(metrics)
                    
                    summary_df.to_excel(writer, sheet_name='Performance_Summary')
                
                # 2. Equity Curve Data
                if hasattr(self, 'portfolio_equity'):
                    equity_df = pd.DataFrame({
                        'Strategy_Equity': self.portfolio_equity,
                        'Strategy_Returns': self.portfolio_returns
                    })
                    equity_df.to_excel(writer, sheet_name='Equity_Data')
                
                # 3. Monthly Returns
                if hasattr(self, 'portfolio_returns'):
                    monthly_data = self.create_monthly_returns_heatmap(self.portfolio_returns)
                    monthly_data.to_excel(writer, sheet_name='Monthly_Returns')
                
                # 4. Individual Asset Signals
                if self.signals:
                    signals_df = pd.DataFrame(self.signals)
                    signals_df.to_excel(writer, sheet_name='Signals')
                
                # 5. Individual Asset Positions
                if self.positions:
                    positions_df = pd.DataFrame(self.positions)
                    positions_df.to_excel(writer, sheet_name='Positions')
                
                # 6. Monte Carlo Results
                if hasattr(self, 'monte_carlo_results') and self.monte_carlo_results:
                    mc_df = pd.DataFrame([self.monte_carlo_results]).T
                    mc_df.columns = ['Monte_Carlo']
                    mc_df.to_excel(writer, sheet_name='Monte_Carlo')
            
            print(f"✓ Results exported to {filename}")
            
        except Exception as e:
            print(f"Error exporting to Excel: {e}")
    
    def run_backtest(self):
        """Run the complete backtest"""
        print("=" * 60)
        print("CRYPTO TREND TRADING STRATEGY - MA CROSSOVER")
        print(f"Moving Averages: {self.fast_ma}-{self.slow_ma} day crossover")
        print(f"Target Volatility: {self.target_volatility:.1%}")
        print(f"Max Leverage: {self.max_leverage:.1f}x")
        print("=" * 60)
        
        # 1. Download data
        self.download_data()
        
        # 2. Calculate portfolio performance
        portfolio_returns, equity_curve = self.calculate_portfolio_performance()
        
        # 3. Calculate performance metrics
        self.performance_metrics = self.calculate_performance_metrics(portfolio_returns, equity_curve)
        
        # 4. Calculate benchmark metrics
        benchmark_metrics = self.calculate_benchmark_performance()
        
        # 5. Run Monte Carlo simulation
        mc_results, mc_distributions = self.run_monte_carlo_simulation(1000)
        self.monte_carlo_results = mc_results
        
        # 6. Display results
        print("\n" + "=" * 60)
        print("PERFORMANCE RESULTS")
        print("=" * 60)
        
        print(f"\nSTRATEGY PERFORMANCE:")
        for metric, value in self.performance_metrics.items():
            if isinstance(value, float):
                if 'Rate' in metric or '%' in metric or 'Return' in metric or 'Exposure' in metric:
                    print(f"{metric:<20}: {value:.2%}")
                else:
                    print(f"{metric:<20}: {value:.4f}")
            else:
                print(f"{metric:<20}: {value}")
        
        print(f"\nBENCHMARK COMPARISON:")
        for symbol, metrics in benchmark_metrics.items():
            print(f"\n{symbol} Buy & Hold:")
            print(f"  CAGR: {metrics['CAGR']:.2%}")
            print(f"  Sharpe: {metrics['Sharpe Ratio']:.3f}")
            print(f"  Max DD: {metrics['Max Drawdown']:.2%}")
        
        if mc_results:
            print(f"\nMONTE CARLO SIMULATION:")
            print(f"  Mean Return: {mc_results['Mean Return']:.2%}")
            print(f"  95% CI: {mc_results['Return 5%']:.2%} to {mc_results['Return 95%']:.2%}")
            print(f"  Mean Sharpe: {mc_results['Mean Sharpe']:.3f}")
            print(f"  Probability of Loss: {mc_results['Probability of Loss']:.1%}")
        
        # 7. Export results
        self.export_to_excel()
        
        # 8. Create plots
        print("\nGenerating performance charts...")
        fig1, fig2 = self.plot_results()
        
        print("\n" + "=" * 60)
        print("BACKTEST COMPLETED!")
        print("=" * 60)
        
        return {
            'strategy_metrics': self.performance_metrics,
            'benchmark_metrics': benchmark_metrics,
            'monte_carlo': mc_results,
            'plots': (fig1, fig2)
        }


# Run the strategy
if __name__ == "__main__":
    
    # Initialize strategy with 50-55 day MA crossover
    strategy = CryptoTrendStrategy(
        start_date='2014-01-01',
        fast_ma=50,
        slow_ma=55,
        target_volatility=0.30,  # <-- Changed from 0.25 to 0.30
        max_leverage=2.0,
        transaction_cost=0.001
    )
    
    # Run backtest
    results = strategy.run_backtest()
    
    print("\nKEY NOTES:")
    print("- Uses 50-55 day Moving Average crossover strategy")
    print("- Long when 50-day MA > 55-day MA")
    print("- Flat when 50-day MA < 55-day MA")
    print("- Volatility-based position sizing with 30% target vol")
    print("- Max 2x leverage per asset")
    print("- Equal-weighted portfolio across all cryptos")