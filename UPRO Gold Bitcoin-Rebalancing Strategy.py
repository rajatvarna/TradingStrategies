import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

"""
Enhanced Portfolio Strategy with BTAL Tactical Hedge

Fixed Core Allocations with Tactical BTAL Adjustments:

BASE ALLOCATION:
- Pre-2024: UPRO 35%, GLD 15%, TLT 15%, VOO 35%, BTC 0%, BTAL 5%
- From 2024: UPRO 35%, GLD 15%, TLT 15%, VOO 30%, BTC 5%, BTAL 5%

TACTICAL BTAL HEDGE:
- When QQQ < 225 DMA: BTAL increases to 20% (daily check)
- When QQQ > 225 DMA: BTAL returns to 5% (daily check)
- Core assets rebalance quarterly only

Features:
- Simple, stable core allocation strategy
- Tactical BTAL hedge for downside protection
- Daily BTAL adjustments, quarterly core rebalancing
- Professional transaction cost modeling
"""

class AdvancedPortfolioBacktester:
    def __init__(self, tickers, start_date='2014-01-01', initial_capital=100000,
                 transaction_cost=0.0015, slippage=0.0015, rebalance_freq='Q'):
        """
        Enhanced Portfolio Backtesting System with Tactical BTAL Hedge
        """
        self.tickers = tickers
        self.start_date = start_date
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.rebalance_freq = rebalance_freq
        
        # Fixed base allocations (never change except for Bitcoin timing)
        self.base_allocation = {
            'pre_2024': {
                'UPRO': 0.35,    # 35%
                'GLD': 0.15,     # 15%
                'TLT': 0.15,     # 15%
                'VOO': 0.35,     # 35%
                'BTC-USD': 0.0,  # 0%
                'BTAL': 0.05     # 5%
            },
            'from_2024': {
                'UPRO': 0.35,    # 35%
                'GLD': 0.15,     # 15%
                'TLT': 0.15,     # 15%
                'VOO': 0.30,     # 30%
                'BTC-USD': 0.05, # 5%
                'BTAL': 0.05     # 5%
            }
        }
        
        # BTAL tactical levels
        self.btal_normal = 0.05   # 5% when QQQ > 225 DMA
        self.btal_defensive = 0.20  # 20% when QQQ < 225 DMA
        
        # Storage for results
        self.data = None
        self.qqq_data = None
        self.qqq_225dma = None
        self.portfolio_data = None
        self.trades = []
        self.stats = {}
        self.benchmark_data = None
        
    def get_base_weights(self, date):
        """Get base allocation weights (unchanged by market regime)"""
        if date.year < 2024:
            weights = self.base_allocation['pre_2024'].copy()
        else:
            weights = self.base_allocation['from_2024'].copy()
        
        return weights
    
    def get_btal_target(self, date):
        """Get BTAL target allocation based on QQQ vs 225 DMA"""
        is_defensive = False
        btal_target = self.btal_normal  # Default 5%
        
        try:
            if (self.qqq_data is not None and self.qqq_225dma is not None and 
                date in self.qqq_data.index and date in self.qqq_225dma.index):
                
                qqq_price = float(self.qqq_data.loc[date])
                qqq_ma225 = float(self.qqq_225dma.loc[date])
                
                if not (np.isnan(qqq_price) or np.isnan(qqq_ma225)):
                    if qqq_price < qqq_ma225:
                        is_defensive = True
                        btal_target = self.btal_defensive  # 20%
                        
        except Exception as e:
            print(f"Warning: BTAL target calculation error for {date}: {e}")
            pass
        
        regime = "DEFENSIVE" if is_defensive else "NORMAL"
        return btal_target, regime, is_defensive
        
    def download_data(self):
        """Download price data for all assets and calculate momentum indicators"""
        print("Downloading data...")
        
        # Download main assets
        raw_data = yf.download(self.tickers, start=self.start_date, end=datetime.now(), progress=False)
        
        # Handle data structure
        if isinstance(raw_data.columns, pd.MultiIndex):
            if 'Adj Close' in raw_data.columns.get_level_values(0):
                data = raw_data['Adj Close'].copy()
            else:
                data = raw_data['Close'].copy()
        else:
            data = raw_data.copy()
        
        # Ensure proper column names
        data.columns = self.tickers
        
        # Download QQQ for BTAL tactical signal
        print("Downloading QQQ for BTAL tactical signal...")
        raw_qqq = yf.download('QQQ', start=self.start_date, end=datetime.now(), progress=False)
        
        if 'Adj Close' in raw_qqq.columns:
            qqq_data = raw_qqq['Adj Close'].copy()
        else:
            qqq_data = raw_qqq['Close'].copy()
        
        if hasattr(qqq_data, 'squeeze'):
            qqq_data = qqq_data.squeeze()
        
        # Calculate QQQ 225-day moving average
        qqq_225dma = qqq_data.rolling(window=225, min_periods=225).mean()
        
        # Download benchmarks
        benchmarks = ['SPY', 'QQQ', 'TQQQ']
        raw_benchmark = yf.download(benchmarks, start=self.start_date, end=datetime.now(), progress=False)
        
        if isinstance(raw_benchmark.columns, pd.MultiIndex):
            if 'Adj Close' in raw_benchmark.columns.get_level_values(0):
                benchmark_data = raw_benchmark['Adj Close'].copy()
            else:
                benchmark_data = raw_benchmark['Close'].copy()
        else:
            benchmark_data = raw_benchmark.copy()
        
        benchmark_data.columns = benchmarks
        
        # Forward fill and clean data
        data = data.ffill().dropna()
        qqq_data = qqq_data.ffill()
        qqq_225dma = qqq_225dma.ffill()
        benchmark_data = benchmark_data.ffill().dropna()
        
        self.data = data
        self.qqq_data = qqq_data
        self.qqq_225dma = qqq_225dma
        self.benchmark_data = benchmark_data
        
        print(f"Data downloaded: {data.shape[0]} days, {data.shape[1]} assets")
        print(f"Date range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
        
        # Calculate tactical BTAL statistics
        valid_data = qqq_225dma.dropna()
        if len(valid_data) > 0:
            qqq_aligned = qqq_data.reindex(valid_data.index)
            defensive_days = int((qqq_aligned < valid_data).sum())
            total_days = len(valid_data)
            defensive_pct = float((defensive_days / total_days) * 100)
            print(f"BTAL 20% hedge periods: {defensive_days}/{total_days} ({defensive_pct:.1f}%)")
    
    def calculate_rebalance_dates(self):
        """Calculate quarterly rebalancing dates"""
        dates = self.data.index
        rebalance_dates = pd.date_range(start=dates[0], end=dates[-1], freq='Q')
        rebalance_dates = [date for date in rebalance_dates if date in dates]
        return rebalance_dates
    
    def backtest(self):
        """Run the backtest with tactical BTAL adjustments"""
        print("Running backtest...")
        
        dates = self.data.index
        n_assets = len(self.tickers)
        
        # Initialize tracking
        portfolio_value = np.zeros(len(dates))
        cash = np.zeros(len(dates))
        positions = np.zeros((len(dates), n_assets))
        btal_regime_tracker = []
        
        # Initial setup
        portfolio_value[0] = self.initial_capital
        cash[0] = self.initial_capital
        
        rebalance_dates = self.calculate_rebalance_dates()
        last_rebalance_idx = 0
        previous_btal_target = None
        
        for i in range(len(dates)):
            current_date = dates[i]
            
            # Get base weights and BTAL target
            base_weights = self.get_base_weights(current_date)
            btal_target, regime, is_defensive = self.get_btal_target(current_date)
            btal_regime_tracker.append(regime)
            
            # Adjust BTAL in base weights
            current_weights = base_weights.copy()
            current_weights['BTAL'] = btal_target
            
            if i == 0:
                # Initial allocation
                target_values = np.array([current_weights[ticker] for ticker in self.tickers]) * self.initial_capital
                
                print(f"Initial allocation on {current_date.strftime('%Y-%m-%d')}: BTAL {btal_target*100:.0f}% ({regime})")
                
                for j, ticker in enumerate(self.tickers):
                    if target_values[j] > 0:
                        price = self.data.iloc[i, j]
                        shares_to_buy = target_values[j] / price
                        transaction_value = shares_to_buy * price
                        total_cost = transaction_value * (1 + self.transaction_cost + self.slippage)
                        
                        if cash[i] >= total_cost:
                            positions[i, j] = shares_to_buy
                            cash[i] -= total_cost
                            
                            self.trades.append({
                                'date': current_date,
                                'ticker': ticker,
                                'action': 'BUY',
                                'shares': shares_to_buy,
                                'price': price,
                                'value': transaction_value,
                                'cost': total_cost - transaction_value,
                                'allocation': current_weights[ticker] * 100,
                                'trade_type': 'INITIAL',
                                'regime': regime
                            })
                
                previous_btal_target = btal_target
                
            else:
                # Carry forward positions
                positions[i] = positions[i-1]
                cash[i] = cash[i-1]
                
                # Check for BTAL tactical adjustment (can happen any day)
                if previous_btal_target is not None and abs(btal_target - previous_btal_target) > 0.01:
                    print(f"BTAL tactical adjustment on {current_date.strftime('%Y-%m-%d')}: {previous_btal_target*100:.0f}% → {btal_target*100:.0f}% ({regime})")
                    
                    # Calculate current portfolio value
                    current_portfolio_value = np.sum(positions[i] * self.data.iloc[i]) + cash[i]
                    
                    # Adjust BTAL position
                    btal_idx = self.tickers.index('BTAL')
                    current_btal_value = positions[i, btal_idx] * self.data.iloc[i, btal_idx]
                    target_btal_value = btal_target * current_portfolio_value
                    btal_diff_value = target_btal_value - current_btal_value
                    
                    if abs(btal_diff_value) > 100:  # Only trade if difference > $100
                        price = self.data.iloc[i, btal_idx]
                        
                        if btal_diff_value > 0:  # Buy more BTAL
                            shares_to_buy = btal_diff_value / price
                            total_cost = btal_diff_value * (1 + self.transaction_cost + self.slippage)
                            
                            if cash[i] >= total_cost:
                                positions[i, btal_idx] += shares_to_buy
                                cash[i] -= total_cost
                                
                                self.trades.append({
                                    'date': current_date,
                                    'ticker': 'BTAL',
                                    'action': 'BUY',
                                    'shares': shares_to_buy,
                                    'price': price,
                                    'value': btal_diff_value,
                                    'cost': total_cost - btal_diff_value,
                                    'allocation': btal_target * 100,
                                    'trade_type': 'TACTICAL',
                                    'regime': regime
                                })
                        
                        else:  # Sell BTAL
                            shares_to_sell = abs(btal_diff_value) / price
                            transaction_value = shares_to_sell * price
                            total_proceeds = transaction_value * (1 - self.transaction_cost - self.slippage)
                            
                            positions[i, btal_idx] -= shares_to_sell
                            cash[i] += total_proceeds
                            
                            self.trades.append({
                                'date': current_date,
                                'ticker': 'BTAL',
                                'action': 'SELL',
                                'shares': shares_to_sell,
                                'price': price,
                                'value': transaction_value,
                                'cost': transaction_value - total_proceeds,
                                'allocation': btal_target * 100,
                                'trade_type': 'TACTICAL',
                                'regime': regime
                            })
                
                # Check for quarterly rebalancing (excluding BTAL)
                if current_date in rebalance_dates and i - last_rebalance_idx > 20:
                    print(f"Quarterly rebalancing on {current_date.strftime('%Y-%m-%d')}")
                    
                    # Calculate current portfolio value
                    current_portfolio_value = np.sum(positions[i] * self.data.iloc[i]) + cash[i]
                    
                    # Rebalance all assets except BTAL
                    for j, ticker in enumerate(self.tickers):
                        if ticker != 'BTAL':  # Skip BTAL - handled by tactical adjustments
                            price = self.data.iloc[i, j]
                            target_value = current_weights[ticker] * current_portfolio_value
                            current_value = positions[i, j] * price
                            value_diff = target_value - current_value
                            
                            if abs(value_diff) > 100:  # Only trade if difference > $100
                                if value_diff > 0:  # Buy
                                    shares_to_buy = value_diff / price
                                    total_cost = value_diff * (1 + self.transaction_cost + self.slippage)
                                    
                                    if cash[i] >= total_cost:
                                        positions[i, j] += shares_to_buy
                                        cash[i] -= total_cost
                                        
                                        self.trades.append({
                                            'date': current_date,
                                            'ticker': ticker,
                                            'action': 'BUY',
                                            'shares': shares_to_buy,
                                            'price': price,
                                            'value': value_diff,
                                            'cost': total_cost - value_diff,
                                            'allocation': current_weights[ticker] * 100,
                                            'trade_type': 'QUARTERLY',
                                            'regime': regime
                                        })
                                
                                else:  # Sell
                                    shares_to_sell = abs(value_diff) / price
                                    transaction_value = shares_to_sell * price
                                    total_proceeds = transaction_value * (1 - self.transaction_cost - self.slippage)
                                    
                                    positions[i, j] -= shares_to_sell
                                    cash[i] += total_proceeds
                                    
                                    self.trades.append({
                                        'date': current_date,
                                        'ticker': ticker,
                                        'action': 'SELL',
                                        'shares': shares_to_sell,
                                        'price': price,
                                        'value': transaction_value,
                                        'cost': transaction_value - total_proceeds,
                                        'allocation': current_weights[ticker] * 100,
                                        'trade_type': 'QUARTERLY',
                                        'regime': regime
                                    })
                    
                    last_rebalance_idx = i
                
                previous_btal_target = btal_target
                
                # Calculate portfolio value
                portfolio_value[i] = np.sum(positions[i] * self.data.iloc[i]) + cash[i]
        
        # Create portfolio DataFrame
        self.portfolio_data = pd.DataFrame({
            'Date': dates,
            'Portfolio_Value': portfolio_value,
            'Cash': cash,
            'Daily_Return': np.concatenate([[0], np.diff(portfolio_value) / portfolio_value[:-1]]),
            'Cumulative_Return': (portfolio_value / self.initial_capital - 1) * 100,
            'BTAL_Regime': btal_regime_tracker
        })
        
        # Add individual asset positions
        for i, ticker in enumerate(self.tickers):
            self.portfolio_data[f'{ticker}_Shares'] = positions[:, i]
            self.portfolio_data[f'{ticker}_Value'] = positions[:, i] * self.data[ticker]
        
        # Add BTAL allocation tracking
        self.portfolio_data['BTAL_Allocation_Pct'] = (
            self.portfolio_data['BTAL_Value'] / self.portfolio_data['Portfolio_Value'] * 100
        )
        
        self.portfolio_data.set_index('Date', inplace=True)
        
        print(f"Backtest completed. {len(self.trades)} trades executed.")
    
    def calculate_drawdowns(self):
        """Calculate detailed drawdown statistics"""
        portfolio_value = self.portfolio_data['Portfolio_Value']
        peaks = portfolio_value.expanding().max()
        drawdown = (portfolio_value - peaks) / peaks * 100
        
        # Find drawdown periods
        in_drawdown = drawdown < 0
        drawdown_periods = []
        
        start_date = None
        for i, (date, is_dd) in enumerate(zip(drawdown.index, in_drawdown)):
            if is_dd and start_date is None:
                start_date = date
            elif not is_dd and start_date is not None:
                end_date = drawdown.index[i-1] if i > 0 else date
                min_dd = drawdown.loc[start_date:end_date].min()
                days = len(drawdown.loc[start_date:end_date])
                
                # Find recovery date
                recovery_date = None
                peak_before_dd = peaks.loc[start_date]
                for future_date in portfolio_value.index:
                    if future_date > end_date and portfolio_value.loc[future_date] >= peak_before_dd:
                        recovery_date = future_date
                        break
                
                recovery_days = (recovery_date - start_date).days if recovery_date else None
                
                drawdown_periods.append({
                    'start': start_date,
                    'end': end_date,
                    'recovery': recovery_date,
                    'max_drawdown': min_dd,
                    'duration_days': days,
                    'recovery_days': recovery_days
                })
                start_date = None
        
        # Handle ongoing drawdown
        if start_date is not None:
            end_date = drawdown.index[-1]
            min_dd = drawdown.loc[start_date:end_date].min()
            days = len(drawdown.loc[start_date:end_date])
            
            drawdown_periods.append({
                'start': start_date,
                'end': end_date,
                'recovery': None,
                'max_drawdown': min_dd,
                'duration_days': days,
                'recovery_days': None
            })
        
        drawdown_periods.sort(key=lambda x: x['max_drawdown'])
        return drawdown, drawdown_periods[:3]
    
    def calculate_statistics(self):
        """Calculate comprehensive portfolio statistics"""
        returns = self.portfolio_data['Daily_Return'].dropna()
        portfolio_value = self.portfolio_data['Portfolio_Value']
        
        # Basic metrics
        total_return = (portfolio_value.iloc[-1] / self.initial_capital - 1) * 100
        annual_return = (1 + total_return/100) ** (365.25 / len(returns)) - 1
        
        # Risk metrics
        volatility = returns.std() * np.sqrt(252)
        downside_returns = returns[returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        
        # Drawdown analysis
        drawdown, top_3_drawdowns = self.calculate_drawdowns()
        max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0
        
        # Performance ratios
        risk_free_rate = 0.02
        sharpe_ratio = (annual_return - risk_free_rate) / volatility if volatility > 0 else 0
        sortino_ratio = (annual_return - risk_free_rate) / downside_volatility if downside_volatility > 0 else 0
        mar_ratio = annual_return / (max_drawdown/100) if max_drawdown > 0 else 0
        
        # Win/Loss analysis
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        win_rate = len(positive_returns) / len(returns) * 100 if len(returns) > 0 else 0
        avg_win = positive_returns.mean() * 100 if len(positive_returns) > 0 else 0
        avg_loss = negative_returns.mean() * 100 if len(negative_returns) > 0 else 0
        
        # Exposure analysis
        total_invested = sum([self.portfolio_data[f'{ticker}_Value'] for ticker in self.tickers])
        avg_exposure = (total_invested / portfolio_value).mean() * 100
        
        self.stats = {
            'Total Return (%)': total_return,
            'Annual Return (%)': annual_return * 100,
            'Volatility (%)': volatility * 100,
            'Sharpe Ratio': sharpe_ratio,
            'Sortino Ratio': sortino_ratio,
            'MAR Ratio': mar_ratio,
            'Max Drawdown (%)': max_drawdown,
            'Win Rate (%)': win_rate,
            'Avg Win (%)': avg_win,
            'Avg Loss (%)': avg_loss,
            'Avg Exposure (%)': avg_exposure,
            'Top 3 Drawdowns': top_3_drawdowns,
            'Total Trades': len(self.trades)
        }
        
        return self.stats
    
    def create_visualizations(self):
        """Create comprehensive visualization suite"""
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Equity Curve
        ax1 = plt.subplot(3, 3, 1)
        self.portfolio_data['Portfolio_Value'].plot(ax=ax1, title='Portfolio Equity Curve', color='blue')
        ax1.axhline(y=self.initial_capital, color='red', linestyle='--', alpha=0.7)
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.grid(True)
        
        # 2. Drawdown Chart
        ax2 = plt.subplot(3, 3, 2)
        drawdown, _ = self.calculate_drawdowns()
        drawdown.plot(ax=ax2, title='Drawdown %', color='red', kind='area', alpha=0.3)
        ax2.set_ylabel('Drawdown (%)')
        ax2.grid(True)
        
        # 3. BTAL Allocation Over Time
        ax3 = plt.subplot(3, 3, 3)
        self.portfolio_data['BTAL_Allocation_Pct'].plot(ax=ax3, title='BTAL Tactical Allocation (%)', color='purple')
        ax3.axhline(y=5, color='green', linestyle='--', alpha=0.7, label='Normal (5%)')
        ax3.axhline(y=20, color='red', linestyle='--', alpha=0.7, label='Defensive (20%)')
        ax3.set_ylabel('BTAL Allocation (%)')
        ax3.legend()
        ax3.grid(True)
        
        # 4. Annual Returns Bar Chart
        ax4 = plt.subplot(3, 3, 4)
        annual_returns = self.portfolio_data['Daily_Return'].resample('Y').apply(lambda x: (1 + x).prod() - 1) * 100
        annual_returns.index = annual_returns.index.year
        annual_returns.plot(kind='bar', ax=ax4, title='Annual Returns (%)', color='green')
        ax4.set_ylabel('Return (%)')
        ax4.grid(True)
        
        # 5. Asset Allocation Over Time
        ax5 = plt.subplot(3, 3, 5)
        allocation_data = pd.DataFrame()
        for ticker in self.tickers:
            allocation_data[ticker] = (self.portfolio_data[f'{ticker}_Value'] / 
                                     self.portfolio_data['Portfolio_Value'] * 100)
        allocation_data.plot(kind='area', ax=ax5, title='Asset Allocation Over Time (%)')
        ax5.set_ylabel('Allocation (%)')
        ax5.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax5.grid(True)
        
        # 6. Rolling Sharpe Ratio
        ax6 = plt.subplot(3, 3, 6)
        rolling_returns = self.portfolio_data['Daily_Return'].rolling(252).mean() * 252
        rolling_vol = self.portfolio_data['Daily_Return'].rolling(252).std() * np.sqrt(252)
        rolling_sharpe = (rolling_returns - 0.02) / rolling_vol
        rolling_sharpe.plot(ax=ax6, title='Rolling 1-Year Sharpe Ratio')
        ax6.axhline(y=1, color='red', linestyle='--', alpha=0.7)
        ax6.grid(True)
        
        # 7. Benchmark Comparison
        ax7 = plt.subplot(3, 3, 7)
        portfolio_cumulative = (1 + self.portfolio_data['Daily_Return']).cumprod()
        portfolio_cumulative.plot(ax=ax7, label='Portfolio', linewidth=2, color='blue')
        
        for benchmark in ['SPY', 'QQQ', 'TQQQ']:
            if benchmark in self.benchmark_data.columns:
                bench_returns = self.benchmark_data[benchmark].pct_change()
                bench_cumulative = (1 + bench_returns).cumprod()
                aligned_returns = bench_cumulative.reindex(portfolio_cumulative.index, method='ffill')
                aligned_returns.plot(ax=ax7, label=benchmark, alpha=0.7)
        
        ax7.set_title('Portfolio vs Benchmarks')
        ax7.legend()
        ax7.grid(True)
        
        # 8. QQQ vs 225 DMA with BTAL Regime
        ax8 = plt.subplot(3, 3, 8)
        if self.qqq_data is not None and self.qqq_225dma is not None:
            regime_data = self.portfolio_data['BTAL_Regime']
            qqq_aligned = self.qqq_data.reindex(regime_data.index, method='ffill')
            ma_aligned = self.qqq_225dma.reindex(regime_data.index, method='ffill')
            
            qqq_aligned.plot(ax=ax8, label='QQQ', color='blue')
            ma_aligned.plot(ax=ax8, label='225 DMA', color='red')
            
            # Highlight BTAL 20% periods
            defensive_mask = regime_data == 'DEFENSIVE'
            if defensive_mask.any():
                ax8.fill_between(regime_data.index, qqq_aligned.min(), qqq_aligned.max(), 
                               where=defensive_mask, alpha=0.2, color='red', label='BTAL 20%')
            
            ax8.set_title('QQQ vs 225 DMA & BTAL Regime')
            ax8.legend()
            ax8.grid(True)
        
        # 9. Trade Type Distribution
        ax9 = plt.subplot(3, 3, 9)
        if self.trades:
            trades_df = pd.DataFrame(self.trades)
            trade_types = trades_df['trade_type'].value_counts()
            trade_types.plot(kind='pie', ax=ax9, title='Trade Types', autopct='%1.1f%%')
        
        plt.tight_layout()
        plt.savefig('portfolio_btal_tactical_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(self):
        """Generate comprehensive performance report"""
        stats = self.calculate_statistics()
        
        print("="*80)
        print("ENHANCED PORTFOLIO STRATEGY - TACTICAL BTAL HEDGE")
        print("="*80)
        print(f"Strategy: Fixed Core Allocation + Tactical BTAL Hedge ({self.rebalance_freq})")
        print(f"Assets: {', '.join(self.tickers)}")
        print()
        print("STRATEGY DETAILS:")
        print("  CORE ALLOCATION (Fixed):")
        print("    Pre-2024: UPRO 35%, GLD 15%, TLT 15%, VOO 35%, BTC 0%")
        print("    From 2024: UPRO 35%, GLD 15%, TLT 15%, VOO 30%, BTC 5%")
        print("  TACTICAL BTAL HEDGE:")
        print("    Normal Markets (QQQ > 225 DMA): BTAL 5%")
        print("    Defensive Markets (QQQ < 225 DMA): BTAL 20%")
        print("    Adjustment Frequency: Daily (any day)")
        print("    Core Rebalancing: Quarterly")
        print()
        print(f"Period: {self.data.index[0].strftime('%Y-%m-%d')} to {self.data.index[-1].strftime('%Y-%m-%d')}")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Final Value: ${self.portfolio_data['Portfolio_Value'].iloc[-1]:,.2f}")
        print()
        
        # BTAL Regime Analysis
        btal_regime_counts = self.portfolio_data['BTAL_Regime'].value_counts()
        total_days = len(self.portfolio_data)
        
        print("BTAL TACTICAL ANALYSIS:")
        print("-" * 40)
        for regime in ['NORMAL', 'DEFENSIVE']:
            if regime in btal_regime_counts.index:
                days = btal_regime_counts[regime]
                pct = (days / total_days) * 100
                btal_level = "5%" if regime == "NORMAL" else "20%"
                print(f"BTAL {btal_level} periods: {days:,} days ({pct:.1f}%)")
        
        print("\nPERFORMANCE METRICS:")
        print("-" * 40)
        for key, value in stats.items():
            if key != 'Top 3 Drawdowns':
                if isinstance(value, float):
                    print(f"{key:<25}: {value:>10.2f}")
                else:
                    print(f"{key:<25}: {value:>10}")
        print()
        
        # Trade type analysis
        if self.trades:
            trades_df = pd.DataFrame(self.trades)
            trade_types = trades_df['trade_type'].value_counts()
            print("TRADE TYPE ANALYSIS:")
            print("-" * 40)
            for trade_type in trade_types.index:
                count = trade_types[trade_type]
                pct = (count / len(self.trades)) * 100
                print(f"{trade_type} trades: {count} ({pct:.1f}%)")
            
            # BTAL tactical trades
            btal_tactical_trades = trades_df[
                (trades_df['ticker'] == 'BTAL') & (trades_df['trade_type'] == 'TACTICAL')
            ]
            print(f"BTAL tactical adjustments: {len(btal_tactical_trades)}")
        
        print("\nTOP 3 WORST DRAWDOWNS:")
        print("-" * 40)
        for i, dd in enumerate(stats['Top 3 Drawdowns'], 1):
            print(f"Drawdown #{i}:")
            print(f"  Period: {dd['start'].strftime('%Y-%m-%d')} to {dd['end'].strftime('%Y-%m-%d')}")
            print(f"  Max Drawdown: {dd['max_drawdown']:.2f}%")
            print(f"  Duration: {dd['duration_days']} days")
            if dd['recovery_days']:
                print(f"  Recovery Days: {dd['recovery_days']} days")
            else:
                print(f"  Recovery: Not yet recovered")
            print()
        
        # Latest trades
        print("LATEST 10 TRADES:")
        print("-" * 40)
        recent_trades = sorted(self.trades, key=lambda x: x['date'], reverse=True)[:10]
        for trade in recent_trades:
            trade_type = trade.get('trade_type', 'N/A')
            regime = trade.get('regime', 'N/A')
            print(f"{trade['date'].strftime('%Y-%m-%d')}: {trade['action']} {trade['shares']:.0f} {trade['ticker']} @ ${trade['price']:.2f} [{trade_type}] ({regime})")
        
        # Current status
        current_regime = self.portfolio_data['BTAL_Regime'].iloc[-1]
        current_btal_allocation = self.portfolio_data['BTAL_Allocation_Pct'].iloc[-1]
        current_date = self.portfolio_data.index[-1]
        
        print(f"\nCURRENT STATUS ({current_date.strftime('%Y-%m-%d')}):")
        print("-" * 40)
        print(f"BTAL Regime: {current_regime}")
        print(f"BTAL Allocation: {current_btal_allocation:.1f}%")
        
        if self.qqq_data is not None and self.qqq_225dma is not None:
            try:
                current_qqq = float(self.qqq_data.iloc[-1])
                current_ma = float(self.qqq_225dma.iloc[-1])
                distance = ((current_qqq / current_ma) - 1) * 100
                print(f"QQQ vs 225 DMA: ${current_qqq:.2f} vs ${current_ma:.2f} ({distance:+.1f}%)")
            except:
                print("QQQ vs 225 DMA: Unable to calculate")
        
        # Benchmark comparison
        print("\nBENCHMARK COMPARISON:")
        print("-" * 40)
        portfolio_total_return = (self.portfolio_data['Portfolio_Value'].iloc[-1] / self.initial_capital - 1) * 100
        
        for benchmark in ['SPY', 'QQQ', 'TQQQ']:
            if benchmark in self.benchmark_data.columns:
                bench_start = self.benchmark_data[benchmark].iloc[0]
                bench_end = self.benchmark_data[benchmark].iloc[-1]
                bench_return = (bench_end / bench_start - 1) * 100
                outperformance = portfolio_total_return - bench_return
                print(f"vs {benchmark}: Portfolio {portfolio_total_return:.2f}% vs {bench_return:.2f}% (Outperformance: {outperformance:+.2f}%)")
    
    def save_results(self):
        """Save all results to comprehensive Excel file"""
        excel_filename = 'BTAL_Tactical_Strategy_Analysis.xlsx'
        
        print(f"Saving results to {excel_filename}...")
        
        with pd.ExcelWriter(excel_filename, engine='xlsxwriter') as writer:
            
            # Strategy Summary
            regime_counts = self.portfolio_data['BTAL_Regime'].value_counts()
            total_days = len(self.portfolio_data)
            
            summary_data = {
                'Metric': ['Strategy Name', 'Start Date', 'End Date', 'Initial Capital', 'Final Value', 
                          'Total Return %', 'Total Trades', 'BTAL Normal Days', 'BTAL Defensive Days'],
                'Value': ['BTAL Tactical Strategy', 
                         self.data.index[0].strftime('%Y-%m-%d'),
                         self.data.index[-1].strftime('%Y-%m-%d'),
                         f'${self.initial_capital:,.2f}',
                         f'${self.portfolio_data["Portfolio_Value"].iloc[-1]:,.2f}',
                         f'{((self.portfolio_data["Portfolio_Value"].iloc[-1] / self.initial_capital - 1) * 100):.2f}%',
                         len(self.trades),
                         regime_counts.get('NORMAL', 0),
                         regime_counts.get('DEFENSIVE', 0)]
            }
            
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
            
            # Portfolio Performance
            self.portfolio_data.to_excel(writer, sheet_name='Portfolio_Performance', index=True)
            
            # All Trades
            if self.trades:
                trades_df = pd.DataFrame(self.trades)
                trades_df.to_excel(writer, sheet_name='All_Trades', index=False)
            
            # Performance Statistics
            stats_items = [[k, v] for k, v in self.stats.items() if k != 'Top 3 Drawdowns']
            stats_df = pd.DataFrame(stats_items, columns=['Metric', 'Value'])
            stats_df.to_excel(writer, sheet_name='Statistics', index=False)
            
            # Drawdown Details
            if 'Top 3 Drawdowns' in self.stats:
                drawdown_data = []
                for i, dd in enumerate(self.stats['Top 3 Drawdowns'], 1):
                    drawdown_data.append({
                        'Rank': i,
                        'Start': dd['start'].strftime('%Y-%m-%d'),
                        'End': dd['end'].strftime('%Y-%m-%d'),
                        'Recovery': dd['recovery'].strftime('%Y-%m-%d') if dd['recovery'] else 'Not Recovered',
                        'Max_Drawdown_%': dd['max_drawdown'],
                        'Duration_Days': dd['duration_days'],
                        'Recovery_Days': dd['recovery_days'] if dd['recovery_days'] else 'N/A'
                    })
                
                pd.DataFrame(drawdown_data).to_excel(writer, sheet_name='Drawdowns', index=False)
        
        print("✅ Results saved to single Excel file with multiple sheets!")
    
    def run_full_analysis(self):
        """Run complete analysis"""
        self.download_data()
        self.backtest()
        self.generate_report()
        self.create_visualizations()
        self.save_results()

# Initialize and run the backtester
if __name__ == "__main__":
    tickers = ['UPRO', 'GLD', 'TLT', 'VOO', 'BTC-USD', 'BTAL']
    
    backtester = AdvancedPortfolioBacktester(
        tickers=tickers,
        start_date='2014-01-01',
        initial_capital=100000,
        transaction_cost=0.0015,
        slippage=0.0015,
        rebalance_freq='Q'
    )
    
    print("🎯 ENHANCED TACTICAL BTAL STRATEGY")
    print("="*50)
    print("CORE ALLOCATION (Fixed):")
    print("  Pre-2024: UPRO 35%, GLD 15%, TLT 15%, VOO 35%, BTC 0%")
    print("  From 2024: UPRO 35%, GLD 15%, TLT 15%, VOO 30%, BTC 5%")
    print()
    print("TACTICAL BTAL HEDGE:")
    print("  🟢 Normal (QQQ > 225 DMA): BTAL 5%")
    print("  🔴 Defensive (QQQ < 225 DMA): BTAL 20%")
    print()
    print("⚙️  TRADING LOGIC:")
    print("  • Core assets: Quarterly rebalancing")
    print("  • BTAL hedge: Daily tactical adjustments")
    print("  • Transaction costs: 0.30% total per trade")
    print("="*50)
    print()
    
    backtester.run_full_analysis()