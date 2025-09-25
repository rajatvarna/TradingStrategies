import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class TQQQMovingAverageStrategy:
    def __init__(self, initial_capital=10000, ma_period=168, transaction_cost_per_share=0.005, slippage=0.0015):
        """
        TQQQ Moving Average Strategy
        
        Parameters:
        - initial_capital: Starting capital ($10,000 default)
        - ma_period: Moving average period in trading days (168 = ~8 months)
        - transaction_cost_per_share: Cost per share ($0.005)
        - slippage: Slippage percentage (0.15% = 0.0015)
        """
        self.initial_capital = initial_capital
        self.ma_period = ma_period
        self.transaction_cost_per_share = transaction_cost_per_share
        self.slippage = slippage
        self.trades = []
        self.daily_returns = []
        
    def download_data(self):
        """Download TQQQ, QQQ, SPY, and Treasury data from Yahoo Finance"""
        print("Downloading data from Yahoo Finance...")
        
        # Download data with error handling
        tickers = ['TQQQ', 'QQQ', 'SPY']
        data_dict = {}
        
        for ticker in tickers:
            try:
                print(f"Downloading {ticker}...")
                df = yf.download(ticker, start='2010-02-01', progress=False)
                
                # Handle MultiIndex columns if present
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel(1)  # Remove ticker level
                
                # Ensure we have the required columns
                if 'Adj Close' in df.columns:
                    data_dict[ticker] = df['Adj Close']
                else:
                    # Fallback to Close if Adj Close not available
                    data_dict[ticker] = df['Close']
                    print(f"Warning: Using Close price for {ticker} (Adj Close not available)")
                    
            except Exception as e:
                print(f"Error downloading {ticker}: {e}")
                continue
        
        # Download Treasury 3-month rate
        try:
            print("Downloading Treasury rate...")
            treasury = yf.download('^IRX', start='2010-02-01', progress=False)
            if isinstance(treasury.columns, pd.MultiIndex):
                treasury.columns = treasury.columns.droplevel(1)
            treasury_rate = treasury['Close'] / 100 / 252  # Convert to daily rate
        except:
            print("Warning: Using constant 2% Treasury rate")
            # Create dummy treasury data
            treasury_rate = pd.Series(0.02/252, index=data_dict['TQQQ'].index if 'TQQQ' in data_dict else pd.date_range('2010-02-01', periods=1000))
        
        # Combine data
        if 'TQQQ' not in data_dict:
            raise ValueError("Could not download TQQQ data")
        
        self.data = pd.DataFrame(index=data_dict['TQQQ'].index)
        self.data['TQQQ_Close'] = data_dict['TQQQ']
        self.data['QQQ_Close'] = data_dict.get('QQQ', data_dict['TQQQ'])  # Fallback
        self.data['SPY_Close'] = data_dict.get('SPY', data_dict['TQQQ'])  # Fallback
        
        # Align treasury rate with stock data
        self.data['Treasury_Rate'] = treasury_rate.reindex(self.data.index).fillna(method='ffill').fillna(0.02/252)
        
        # Drop rows with missing critical data
        self.data = self.data.dropna(subset=['TQQQ_Close', 'QQQ_Close'])
        
        print(f"Data downloaded: {self.data.index[0]} to {self.data.index[-1]}")
        print(f"Total days: {len(self.data)}")
        
    def calculate_signals(self):
        """Calculate moving average signals"""
        print(f"Calculating {self.ma_period}-day moving average signals...")
        
        # Calculate moving average of QQQ (signal asset)
        self.data['QQQ_MA'] = self.data['QQQ_Close'].rolling(window=self.ma_period).mean()
        
        # Generate signals: 1 = Buy TQQQ, 0 = Cash
        self.data['Signal'] = np.where(self.data['QQQ_Close'] >= self.data['QQQ_MA'], 1, 0)
        
        # Identify signal changes (trade points)
        self.data['Signal_Change'] = self.data['Signal'].diff()
        self.data['Trade_Date'] = (self.data['Signal_Change'] != 0) & (~self.data['Signal_Change'].isna())
        
        # Calculate daily returns for all assets
        for asset in ['TQQQ', 'QQQ', 'SPY']:
            self.data[f'{asset}_Return'] = self.data[f'{asset}_Close'].pct_change()
        self.data['Treasury_Return'] = self.data['Treasury_Rate']
        
        print(f"Signals calculated. Moving average starts: {self.data['QQQ_MA'].first_valid_index()}")
        
    def calculate_transaction_costs(self, shares, price):
        """Calculate transaction costs including slippage"""
        # Per-share cost
        cost_per_share = shares * self.transaction_cost_per_share
        
        # Slippage cost (percentage of trade value)
        trade_value = shares * price
        slippage_cost = trade_value * self.slippage
        
        return cost_per_share + slippage_cost
        
    def backtest_strategy(self):
        """Run the backtest"""
        print("Running backtest...")
        
        # Initialize tracking variables
        portfolio_value = self.initial_capital
        cash_position = self.initial_capital
        tqqq_shares = 0
        position = 0  # 0 = cash, 1 = TQQQ
        
        portfolio_values = []
        positions = []
        trade_log = []
        
        # Start from when MA is available
        start_idx = self.data['QQQ_MA'].first_valid_index()
        backtest_data = self.data.loc[start_idx:].copy()
        
        for date, row in backtest_data.iterrows():
            current_signal = row['Signal']
            
            # Check if we need to trade
            if current_signal != position:
                # Record trade
                if position == 0 and current_signal == 1:
                    # Buy TQQQ
                    effective_price = row['TQQQ_Close'] * (1 + self.slippage)
                    transaction_cost = self.calculate_transaction_costs(cash_position / effective_price, effective_price)
                    
                    net_cash_for_shares = cash_position - transaction_cost
                    tqqq_shares = net_cash_for_shares / effective_price
                    
                    trade_log.append({
                        'Trade_Date': date,
                        'Action': 'BUY_TQQQ',
                        'Price': row['TQQQ_Close'],
                        'Effective_Price': effective_price,
                        'Shares': tqqq_shares,
                        'Value': net_cash_for_shares,
                        'Transaction_Cost': transaction_cost,
                        'Slippage_Cost': cash_position * self.slippage,
                        'Trade_Type': 'Signal',
                        'Turnover': 100.00,
                        'Allocation': '100.00% TQQQ'
                    })
                    
                    cash_position = 0
                    
                elif position == 1 and current_signal == 0:
                    # Sell TQQQ, move to cash
                    effective_price = row['TQQQ_Close'] * (1 - self.slippage)
                    gross_proceeds = tqqq_shares * effective_price
                    transaction_cost = self.calculate_transaction_costs(tqqq_shares, effective_price)
                    cash_position = gross_proceeds - transaction_cost
                    
                    trade_log.append({
                        'Trade_Date': date,
                        'Action': 'SELL_TQQQ',
                        'Price': row['TQQQ_Close'],
                        'Effective_Price': effective_price,
                        'Shares': tqqq_shares,
                        'Value': gross_proceeds,
                        'Transaction_Cost': transaction_cost,
                        'Slippage_Cost': tqqq_shares * row['TQQQ_Close'] * self.slippage,
                        'Trade_Type': 'Signal',
                        'Turnover': 100.00,
                        'Allocation': '100.00% Cash'
                    })
                    
                    tqqq_shares = 0
                
                position = current_signal
            
            # Calculate daily portfolio value
            if position == 1:  # In TQQQ
                portfolio_value = tqqq_shares * row['TQQQ_Close']
            else:  # In cash, earn Treasury rate
                if cash_position > 0:
                    cash_position *= (1 + row['Treasury_Return'])
                portfolio_value = cash_position
            
            portfolio_values.append(portfolio_value)
            positions.append(position)
        
        # Store results
        self.backtest_data = backtest_data.copy()
        self.backtest_data['Portfolio_Value'] = portfolio_values
        self.backtest_data['Position'] = positions
        self.trade_log = pd.DataFrame(trade_log)
        
        # Calculate portfolio returns
        self.backtest_data['Portfolio_Return'] = self.backtest_data['Portfolio_Value'].pct_change()
        
        # Add trade periods to trade log
        if len(self.trade_log) > 0:
            for i in range(len(self.trade_log)):
                current_date = self.trade_log.iloc[i]['Trade_Date']
                if i < len(self.trade_log) - 1:
                    next_date = self.trade_log.iloc[i+1]['Trade_Date']
                    self.trade_log.loc[i, 'End_Date'] = next_date
                else:
                    self.trade_log.loc[i, 'End_Date'] = self.backtest_data.index[-1]
                
                self.trade_log.loc[i, 'Start_Date'] = current_date
        
        print(f"Backtest completed. Total trades: {len(self.trade_log)}")
        print(f"Final portfolio value: ${portfolio_values[-1]:,.2f}")
        
    def calculate_benchmarks(self):
        """Calculate multiple buy and hold benchmarks"""
        start_value = self.initial_capital
        
        # TQQQ Buy & Hold
        start_price_tqqq = self.backtest_data['TQQQ_Close'].iloc[0]
        shares_tqqq = start_value / start_price_tqqq
        self.backtest_data['BuyHold_TQQQ_Value'] = shares_tqqq * self.backtest_data['TQQQ_Close']
        self.backtest_data['BuyHold_TQQQ_Return'] = self.backtest_data['BuyHold_TQQQ_Value'].pct_change()
        
        # QQQ Buy & Hold
        start_price_qqq = self.backtest_data['QQQ_Close'].iloc[0]
        shares_qqq = start_value / start_price_qqq
        self.backtest_data['BuyHold_QQQ_Value'] = shares_qqq * self.backtest_data['QQQ_Close']
        self.backtest_data['BuyHold_QQQ_Return'] = self.backtest_data['BuyHold_QQQ_Value'].pct_change()
        
        # SPY Buy & Hold
        start_price_spy = self.backtest_data['SPY_Close'].iloc[0]
        shares_spy = start_value / start_price_spy
        self.backtest_data['BuyHold_SPY_Value'] = shares_spy * self.backtest_data['SPY_Close']
        self.backtest_data['BuyHold_SPY_Return'] = self.backtest_data['BuyHold_SPY_Value'].pct_change()
        
    def calculate_comprehensive_metrics(self, returns, benchmark_returns=None, values=None, name="Strategy"):
        """Calculate all the comprehensive metrics from the reference document"""
        
        returns_clean = returns.dropna()
        total_days = len(returns_clean)
        years = total_days / 252
        
        if len(returns_clean) == 0 or years == 0:
            return {}
        
        # Basic return metrics
        if values is not None:
            start_val = values.iloc[0]
            end_val = values.iloc[-1]
            total_return = (end_val / start_val) - 1
            cagr = (end_val / start_val) ** (1/years) - 1
        else:
            total_return = (1 + returns_clean).prod() - 1
            cagr = (1 + returns_clean).prod() ** (252/total_days) - 1
        
        # Risk metrics
        annual_vol = returns_clean.std() * np.sqrt(252)
        monthly_vol = returns_clean.std() * np.sqrt(21)  # Approximate monthly vol
        
        # Risk-free rate
        rf_rate = self.backtest_data['Treasury_Return'].mean() * 252
        excess_return = cagr - rf_rate
        
        # Sharpe ratio
        sharpe_ratio = excess_return / annual_vol if annual_vol > 0 else 0
        
        # Sortino ratio
        downside_returns = returns_clean[returns_clean < 0]
        downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        downside_vol_monthly = downside_returns.std() * np.sqrt(21) if len(downside_returns) > 0 else 0
        sortino_ratio = excess_return / downside_vol if downside_vol > 0 else 0
        
        # Arithmetic and geometric means
        arithmetic_mean_monthly = returns_clean.mean()
        arithmetic_mean_annual = arithmetic_mean_monthly * 252
        geometric_mean_monthly = (1 + returns_clean).prod() ** (1/len(returns_clean)) - 1
        geometric_mean_annual = (1 + geometric_mean_monthly) ** 252 - 1
        
        # Max drawdown
        if values is not None:
            peak = values.expanding(min_periods=1).max()
            drawdown = (values - peak) / peak
            max_dd = drawdown.min()
        else:
            cumulative = (1 + returns_clean).cumprod()
            peak = cumulative.expanding(min_periods=1).max()
            drawdown = (cumulative - peak) / peak
            max_dd = drawdown.min()
        
        # Benchmark-related metrics
        beta, alpha, correlation, r2 = 0, 0, 0, 0
        treynor_ratio = 0
        active_return = 0
        tracking_error = 0
        information_ratio = 0
        upside_capture = 0
        downside_capture = 0
        
        if benchmark_returns is not None:
            benchmark_clean = benchmark_returns.dropna()
            common_idx = returns_clean.index.intersection(benchmark_clean.index)
            
            if len(common_idx) > 10:
                ret_common = returns_clean.loc[common_idx]
                bench_common = benchmark_clean.loc[common_idx]
                
                # Correlation and regression metrics
                correlation = ret_common.corr(bench_common)
                
                if bench_common.var() > 0:
                    covariance = ret_common.cov(bench_common)
                    beta = covariance / bench_common.var()
                    
                    # Alpha (annualized)
                    benchmark_return = (1 + bench_common).prod() ** (252/len(bench_common)) - 1
                    alpha = cagr - (rf_rate + beta * (benchmark_return - rf_rate))
                    
                    # R-squared
                    r2 = correlation ** 2
                    
                    # Treynor ratio
                    treynor_ratio = excess_return / beta if beta != 0 else 0
                    
                    # Active return and tracking error
                    active_returns = ret_common - bench_common
                    active_return = active_returns.mean() * 252
                    tracking_error = active_returns.std() * np.sqrt(252)
                    information_ratio = active_return / tracking_error if tracking_error > 0 else 0
                    
                    # Capture ratios
                    upside_bench = bench_common[bench_common > 0]
                    upside_ret = ret_common.loc[upside_bench.index]
                    if len(upside_bench) > 0 and upside_bench.mean() > 0:
                        upside_capture = (upside_ret.mean() / upside_bench.mean()) * 100
                    
                    downside_bench = bench_common[bench_common < 0]
                    downside_ret = ret_common.loc[downside_bench.index]
                    if len(downside_bench) > 0 and downside_bench.mean() < 0:
                        downside_capture = (downside_ret.mean() / downside_bench.mean()) * 100
        
        # Calmar ratio
        calmar_ratio = abs(cagr / max_dd) if max_dd < 0 else 0
        
        # Modigliani-Modigliani measure (M2)
        if benchmark_returns is not None and annual_vol > 0:
            benchmark_vol = benchmark_returns.dropna().std() * np.sqrt(252)
            m2 = rf_rate + (excess_return / annual_vol) * benchmark_vol - benchmark_return if 'benchmark_return' in locals() else 0
        else:
            m2 = 0
        
        # Higher moments
        skewness = returns_clean.skew()
        excess_kurtosis = returns_clean.kurtosis()
        
        # Value at Risk (VaR)
        var_95 = np.percentile(returns_clean, 5)  # 5% VaR
        cvar_95 = returns_clean[returns_clean <= var_95].mean() if len(returns_clean[returns_clean <= var_95]) > 0 else 0
        
        # Analytical VaR (assuming normal distribution)
        analytical_var_95 = stats.norm.ppf(0.05, returns_clean.mean(), returns_clean.std())
        
        # Win rate and gain/loss ratio
        positive_periods = len(returns_clean[returns_clean > 0])
        total_periods = len(returns_clean)
        win_rate = positive_periods / total_periods if total_periods > 0 else 0
        
        avg_gain = returns_clean[returns_clean > 0].mean() if len(returns_clean[returns_clean > 0]) > 0 else 0
        avg_loss = abs(returns_clean[returns_clean < 0].mean()) if len(returns_clean[returns_clean < 0]) > 0 else 0
        gain_loss_ratio = avg_gain / avg_loss if avg_loss > 0 else 0
        
        # Withdrawal rates (simplified calculation)
        safe_withdrawal_rate = max(0, cagr - annual_vol) if annual_vol > 0 else 0
        perpetual_withdrawal_rate = cagr * 0.7  # Conservative estimate
        
        # Best and worst years
        annual_rets = self.calculate_annual_returns_for_series(values) if values is not None else pd.Series()
        best_year = annual_rets.max() if len(annual_rets) > 0 else 0
        worst_year = annual_rets.min() if len(annual_rets) > 0 else 0
        
        return {
            'Start_Balance': f"${values.iloc[0]:,.2f}" if values is not None else f"${self.initial_capital:,.2f}",
            'End_Balance': f"${values.iloc[-1]:,.2f}" if values is not None else f"${self.initial_capital * (1 + total_return):,.2f}",
            'Total_Return': f"{total_return:.2%}",
            'CAGR': f"{cagr:.2%}",
            'Arithmetic_Mean_Monthly': f"{arithmetic_mean_monthly:.2%}",
            'Arithmetic_Mean_Annual': f"{arithmetic_mean_annual:.2%}",
            'Geometric_Mean_Monthly': f"{geometric_mean_monthly:.2%}",
            'Geometric_Mean_Annual': f"{geometric_mean_annual:.2%}",
            'Annual_Volatility': f"{annual_vol:.2%}",
            'Monthly_Volatility': f"{monthly_vol:.2%}",
            'Downside_Deviation_Monthly': f"{downside_vol_monthly:.2%}",
            'Max_Drawdown': f"{max_dd:.2%}",
            'Sharpe_Ratio': f"{sharpe_ratio:.3f}",
            'Sortino_Ratio': f"{sortino_ratio:.3f}",
            'Calmar_Ratio': f"{calmar_ratio:.3f}",
            'Beta': f"{beta:.2f}",
            'Alpha_Annual': f"{alpha:.2%}",
            'Correlation': f"{correlation:.2f}",
            'R_Squared': f"{r2:.2%}",
            'Treynor_Ratio': f"{treynor_ratio:.2f}",
            'M2_Measure': f"{m2:.2%}",
            'Active_Return': f"{active_return:.2%}",
            'Tracking_Error': f"{tracking_error:.2%}",
            'Information_Ratio': f"{information_ratio:.3f}",
            'Skewness': f"{skewness:.2f}",
            'Excess_Kurtosis': f"{excess_kurtosis:.2f}",
            'Historical_VaR_5pct': f"{var_95:.2%}",
            'Analytical_VaR_5pct': f"{analytical_var_95:.2%}",
            'Conditional_VaR_5pct': f"{cvar_95:.2%}",
            'Upside_Capture_Ratio': f"{upside_capture:.2f}%",
            'Downside_Capture_Ratio': f"{downside_capture:.2f}%",
            'Safe_Withdrawal_Rate': f"{safe_withdrawal_rate:.2%}",
            'Perpetual_Withdrawal_Rate': f"{perpetual_withdrawal_rate:.2%}",
            'Positive_Periods': f"{positive_periods} out of {total_periods} ({win_rate:.2%})",
            'Gain_Loss_Ratio': f"{gain_loss_ratio:.2f}",
            'Best_Year': f"{best_year:.2%}",
            'Worst_Year': f"{worst_year:.2%}",
            'Total_Trades': len(self.trade_log) if name == "Moving Average Strategy" else 1
        }
        
    def calculate_performance_metrics(self):
        """Calculate comprehensive performance statistics for all strategies"""
        print("Calculating comprehensive performance metrics...")
        
        # Define strategies with their data
        strategies = {
            'Moving Average Strategy': {
                'values': self.backtest_data['Portfolio_Value'],
                'returns': self.backtest_data['Portfolio_Return']
            },
            'Buy & Hold TQQQ': {
                'values': self.backtest_data['BuyHold_TQQQ_Value'],
                'returns': self.backtest_data['BuyHold_TQQQ_Return']
            },
            'Buy & Hold QQQ': {
                'values': self.backtest_data['BuyHold_QQQ_Value'],
                'returns': self.backtest_data['BuyHold_QQQ_Return']
            },
            'Buy & Hold SPY': {
                'values': self.backtest_data['BuyHold_SPY_Value'],
                'returns': self.backtest_data['BuyHold_SPY_Return']
            }
        }
        
        self.performance_stats = {}
        
        # Use SPY as benchmark for most calculations
        benchmark_returns = self.backtest_data['BuyHold_SPY_Return']
        
        for name, data in strategies.items():
            values = data['values']
            returns = data['returns']
            
            # Use different benchmarks for different strategies
            if name == 'Moving Average Strategy':
                bench = self.backtest_data['BuyHold_TQQQ_Return']  # Compare strategy to TQQQ buy-hold
            else:
                bench = benchmark_returns  # Compare buy-hold strategies to SPY
                
            self.performance_stats[name] = self.calculate_comprehensive_metrics(
                returns, bench, values, name
            )
        
        print("Performance metrics calculated.")
        
    def calculate_annual_returns_for_series(self, values):
        """Calculate annual returns for a given value series"""
        if values is None or len(values) == 0:
            return pd.Series()
            
        df = pd.DataFrame({'Value': values})
        df['Year'] = df.index.year
        
        annual_returns = []
        for year in df['Year'].unique():
            year_data = df[df['Year'] == year]
            if len(year_data) > 1:
                start_val = year_data['Value'].iloc[0]
                end_val = year_data['Value'].iloc[-1]
                annual_return = (end_val / start_val) - 1
                annual_returns.append(annual_return)
        
        return pd.Series(annual_returns)
        
    def calculate_annual_returns(self):
        """Calculate annual returns for all strategies"""
        self.backtest_data['Year'] = self.backtest_data.index.year
        
        annual_data = []
        for year in sorted(self.backtest_data['Year'].unique()):
            year_data = self.backtest_data[self.backtest_data['Year'] == year]
            
            if len(year_data) > 1:
                row = {'Year': year}
                
                # Strategy returns
                strategies = [('Portfolio', 'Moving Average Strategy'), 
                             ('BuyHold_TQQQ', 'Buy & Hold TQQQ'), 
                             ('BuyHold_QQQ', 'Buy & Hold QQQ'), 
                             ('BuyHold_SPY', 'Buy & Hold SPY')]
                
                for strategy_key, strategy_name in strategies:
                    col_name = f'{strategy_key}_Value'
                    if col_name in year_data.columns:
                        start_val = year_data[col_name].iloc[0]
                        end_val = year_data[col_name].iloc[-1]
                        row[strategy_name] = (end_val / start_val) - 1
                
                # Add Treasury rate and inflation (approximate)
                treasury_annual = year_data['Treasury_Return'].mean() * 252
                row['Treasury_Rate'] = treasury_annual
                row['Inflation'] = 0.025  # Approximate 2.5% inflation
                
                annual_data.append(row)
        
        self.annual_returns = pd.DataFrame(annual_data)
        
    def calculate_monthly_returns(self):
        """Calculate monthly returns table"""
        monthly_data = []
        
        for year_month, group in self.backtest_data.groupby([self.backtest_data.index.year, self.backtest_data.index.month]):
            if len(group) > 1:
                year, month = year_month
                
                row = {
                    'Year': year,
                    'Month': month,
                }
                
                # Calculate returns for each strategy
                strategies = [('Portfolio', 'Moving Average Strategy'), 
                             ('BuyHold_TQQQ', 'Buy & Hold TQQQ'), 
                             ('BuyHold_QQQ', 'Buy & Hold QQQ'), 
                             ('BuyHold_SPY', 'Buy & Hold SPY')]
                
                for strategy_key, strategy_name in strategies:
                    col_name = f'{strategy_key}_Value'
                    if col_name in group.columns:
                        start_val = group[col_name].iloc[0]
                        end_val = group[col_name].iloc[-1]
                        monthly_return = (end_val / start_val) - 1
                        row[f'{strategy_name}_Return'] = monthly_return
                        row[f'{strategy_name}_Balance'] = end_val
                
                # Treasury rate
                row['Treasury_Rate'] = group['Treasury_Return'].mean() * 21  # Monthly rate
                
                monthly_data.append(row)
        
        self.monthly_returns = pd.DataFrame(monthly_data)
        
    def calculate_trailing_returns(self):
        """Calculate trailing returns analysis"""
        
        end_date = self.backtest_data.index[-1]
        periods = {
            '3_Month': 63,    # ~3 months
            'YTD': None,      # Year to date
            '1_Year': 252,    # 1 year
            '3_Year': 252*3,  # 3 years
            '5_Year': 252*5,  # 5 years
            'Full': None      # Full period
        }
        
        trailing_data = []
        
        strategies = ['Moving Average Strategy', 'Buy & Hold TQQQ', 'Buy & Hold QQQ', 'Buy & Hold SPY']
        value_cols = ['Portfolio_Value', 'BuyHold_TQQQ_Value', 'BuyHold_QQQ_Value', 'BuyHold_SPY_Value']
        
        for strategy, value_col in zip(strategies, value_cols):
            row = {'Strategy': strategy}
            
            for period_name, days in periods.items():
                if period_name == 'YTD':
                    # Year to date
                    ytd_start = pd.Timestamp(f"{end_date.year}-01-01")
                    ytd_data = self.backtest_data[self.backtest_data.index >= ytd_start]
                    if len(ytd_data) > 1:
                        start_val = ytd_data[value_col].iloc[0]
                        end_val = ytd_data[value_col].iloc[-1]
                        ytd_return = (end_val / start_val) - 1
                        row[f'{period_name}_Total_Return'] = ytd_return
                        
                        # Annualized return
                        ytd_days = len(ytd_data)
                        if ytd_days > 0:
                            annualized = (1 + ytd_return) ** (252/ytd_days) - 1
                            row[f'{period_name}_Annualized_Return'] = annualized
                            
                elif period_name == 'Full':
                    # Full period
                    start_val = self.backtest_data[value_col].iloc[0]
                    end_val = self.backtest_data[value_col].iloc[-1]
                    total_return = (end_val / start_val) - 1
                    row[f'{period_name}_Total_Return'] = total_return
                    
                    # Annualized return
                    total_days = len(self.backtest_data)
                    years = total_days / 252
                    if years > 0:
                        annualized = (1 + total_return) ** (1/years) - 1
                        row[f'{period_name}_Annualized_Return'] = annualized
                        
                else:
                    # Fixed periods
                    if len(self.backtest_data) >= days:
                        period_data = self.backtest_data.tail(days)
                        start_val = period_data[value_col].iloc[0]
                        end_val = period_data[value_col].iloc[-1]
                        period_return = (end_val / start_val) - 1
                        row[f'{period_name}_Total_Return'] = period_return
                        
                        # Annualized return
                        period_days = len(period_data)
                        if period_days > 0:
                            annualized = (1 + period_return) ** (252/period_days) - 1
                            row[f'{period_name}_Annualized_Return'] = annualized
            
            trailing_data.append(row)
        
        self.trailing_returns = pd.DataFrame(trailing_data)
        
    def calculate_rolling_returns_analysis(self):
        """Calculate rolling returns summary"""
        periods = [252, 252*3, 252*5, 252*7]  # 1, 3, 5, 7 years
        period_names = ['1_Year', '3_Years', '5_Years', '7_Years']
        
        rolling_data = []
        
        strategies = ['Moving Average Strategy', 'Buy & Hold TQQQ']
        return_cols = ['Portfolio_Return', 'BuyHold_TQQQ_Return']
        
        for strategy, return_col in zip(strategies, return_cols):
            row = {'Strategy': strategy}
            
            returns = self.backtest_data[return_col].dropna()
            
            for period, period_name in zip(periods, period_names):
                if len(returns) >= period:
                    rolling_returns = returns.rolling(period).apply(lambda x: (1 + x).prod() - 1)
                    rolling_returns = rolling_returns.dropna()
                    
                    if len(rolling_returns) > 0:
                        # Annualize the returns
                        annualized_rolling = (1 + rolling_returns) ** (252/period) - 1
                        
                        row[f'{period_name}_Average'] = annualized_rolling.mean()
                        row[f'{period_name}_High'] = annualized_rolling.max()
                        row[f'{period_name}_Low'] = annualized_rolling.min()
            
            rolling_data.append(row)
        
        self.rolling_returns_summary = pd.DataFrame(rolling_data)
        
    def calculate_risk_management_performance(self):
        """Calculate risk management performance analysis"""
        
        # Periods in market vs out of market
        total_periods = len(self.backtest_data)
        in_market_periods = self.backtest_data['Position'].sum()
        out_market_periods = total_periods - in_market_periods
        
        # Performance when in market
        in_market_data = self.backtest_data[self.backtest_data['Position'] == 1]
        if len(in_market_data) > 0:
            in_market_returns = in_market_data['Portfolio_Return'].dropna()
            in_market_return_annual = in_market_returns.mean() * 252 if len(in_market_returns) > 0 else 0
            in_market_vol_annual = in_market_returns.std() * np.sqrt(252) if len(in_market_returns) > 0 else 0
            in_market_sharpe = (in_market_return_annual - self.backtest_data['Treasury_Return'].mean() * 252) / in_market_vol_annual if in_market_vol_annual > 0 else 0
        else:
            in_market_return_annual = 0
            in_market_vol_annual = 0
            in_market_sharpe = 0
        
        # Performance when out of market (in cash)
        out_market_data = self.backtest_data[self.backtest_data['Position'] == 0]
        if len(out_market_data) > 0:
            out_market_returns = out_market_data['Portfolio_Return'].dropna()
            out_market_return_annual = out_market_returns.mean() * 252 if len(out_market_returns) > 0 else 0
            out_market_vol_annual = out_market_returns.std() * np.sqrt(252) if len(out_market_returns) > 0 else 0
        else:
            out_market_return_annual = self.backtest_data['Treasury_Return'].mean() * 252  # Treasury rate
            out_market_vol_annual = 0.001  # Very low volatility for cash
        
        risk_free_return = self.backtest_data['Treasury_Return'].mean() * 252
        
        self.risk_management_performance = pd.DataFrame([
            {
                'Type': 'In Market',
                'Periods_Count': in_market_periods,
                'Periods_Percent': f"{in_market_periods/total_periods:.2%}",
                'Return_Annual': f"{in_market_return_annual:.2%}",
                'Stdev_Annual': f"{in_market_vol_annual:.2%}",
                'Sharpe_Ratio': f"{in_market_sharpe:.3f}",
                'Risk_Free_Return': f"{risk_free_return:.2%}"
            },
            {
                'Type': 'Out of Market',
                'Periods_Count': out_market_periods,
                'Periods_Percent': f"{out_market_periods/total_periods:.2%}",
                'Return_Annual': f"{out_market_return_annual:.2%}",
                'Stdev_Annual': f"{out_market_vol_annual:.2%}",
                'Sharpe_Ratio': 'N/A',
                'Risk_Free_Return': f"{out_market_return_annual:.2%}"
            }
        ])
        
    def calculate_trading_performance(self):
        """Calculate comprehensive trading performance metrics"""
        if len(self.trade_log) == 0:
            self.trading_performance = {}
            return
        
        total_trades = len(self.trade_log)
        
        # Calculate turnover
        total_value = self.backtest_data['Portfolio_Value'].iloc[-1]
        years = len(self.backtest_data) / 252
        average_annual_turnover = (total_trades * 100) / years  # Each trade is 100% turnover
        
        # Transaction costs
        total_transaction_costs = self.trade_log['Transaction_Cost'].sum()
        
        # Performance without costs (approximate)
        portfolio_return_with_costs = (self.backtest_data['Portfolio_Value'].iloc[-1] / self.backtest_data['Portfolio_Value'].iloc[0]) - 1
        portfolio_return_without_costs = portfolio_return_with_costs + (total_transaction_costs / self.initial_capital)
        
        # Annualized returns
        annualized_return_with_costs = (1 + portfolio_return_with_costs) ** (1/years) - 1
        annualized_return_without_costs = (1 + portfolio_return_without_costs) ** (1/years) - 1
        cost_impact = annualized_return_without_costs - annualized_return_with_costs
        
        self.trading_performance = {
            'Total_Trades': total_trades,
            'Analyzed_Trades': total_trades,
            'Rebalancing_Trades': 0,
            'Average_Annual_Turnover': f"{average_annual_turnover:.2f}%",
            'Average_Turnover_Per_Trade': "100.00%",
            'Median_Turnover_Per_Trade': "100.00%",
            'Average_Turnover_Per_Trade_Excluding_Rebalancing': "100.00%",
            'Median_Turnover_Per_Trade_Excluding_Rebalancing': "100.00%",
            'Cumulative_Trading_Return_Without_Costs': f"{portfolio_return_without_costs:.2%}",
            'Cumulative_Trading_Return_After_Costs': f"{portfolio_return_with_costs:.2%}",
            'Cumulative_Trading_Cost_Impact': f"{total_transaction_costs/self.initial_capital:.2%}",
            'Annualized_Trading_Return_Without_Costs': f"{annualized_return_without_costs:.2%}",
            'Annualized_Trading_Return_After_Costs': f"{annualized_return_with_costs:.2%}",
            'Annualized_Trading_Cost_Impact': f"{cost_impact:.2%}"
        }
        
    def calculate_detailed_drawdowns(self):
        """Calculate detailed drawdown periods with recovery analysis"""
        values = self.backtest_data['Portfolio_Value']
        peak = values.expanding(min_periods=1).max()
        drawdown = (values - peak) / peak
        
        # Find drawdown periods
        in_drawdown = drawdown < -0.001  # Use small threshold to avoid noise
        drawdown_periods = []
        
        start = None
        peak_value = None
        for i, (date, is_dd) in enumerate(in_drawdown.items()):
            if is_dd and start is None:
                start = date
                peak_value = peak.loc[date]
            elif not is_dd and start is not None:
                # Find recovery date (when value exceeds previous peak)
                end_dd = date
                recovery_date = None
                
                # Look for recovery after drawdown ends
                future_data = values.loc[date:]
                for future_date, future_value in future_data.items():
                    if future_value >= peak_value:
                        recovery_date = future_date
                        break
                
                period_dd = drawdown.loc[start:end_dd]
                max_dd = period_dd.min()
                length_days = len(period_dd)
                
                # Calculate recovery time
                recovery_time_days = 0
                underwater_period_days = length_days
                if recovery_date:
                    recovery_time_days = (recovery_date - end_dd).days
                    underwater_period_days = (recovery_date - start).days
                
                drawdown_periods.append({
                    'Start': start.strftime('%Y-%m-%d'),
                    'End': end_dd.strftime('%Y-%m-%d'),
                    'Length': f"{length_days//30} months" if length_days > 30 else f"{length_days} days",
                    'Recovery_By': recovery_date.strftime('%Y-%m-%d') if recovery_date else "Not Recovered",
                    'Recovery_Time': f"{recovery_time_days//30} months" if recovery_time_days > 30 else f"{recovery_time_days} days",
                    'Underwater_Period': f"{underwater_period_days//30} months" if underwater_period_days > 30 else f"{underwater_period_days} days",
                    'Max_Drawdown': f"{max_dd:.2%}"
                })
                start = None
        
        # Handle if we're still in drawdown at the end
        if start is not None:
            period_dd = drawdown.loc[start:]
            max_dd = period_dd.min()
            length_days = len(period_dd)
            
            drawdown_periods.append({
                'Start': start.strftime('%Y-%m-%d'),
                'End': drawdown.index[-1].strftime('%Y-%m-%d'),
                'Length': f"{length_days//30} months" if length_days > 30 else f"{length_days} days",
                'Recovery_By': "Not Recovered",
                'Recovery_Time': "Ongoing",
                'Underwater_Period': f"{length_days//30} months" if length_days > 30 else f"{length_days} days",
                'Max_Drawdown': f"{max_dd:.2%}"
            })
        
        # Sort by drawdown magnitude
        drawdown_df = pd.DataFrame(drawdown_periods)
        if not drawdown_df.empty:
            drawdown_df['DD_Value'] = drawdown_df['Max_Drawdown'].str.rstrip('%').astype(float)
            drawdown_df = drawdown_df.sort_values('DD_Value').drop('DD_Value', axis=1)
            drawdown_df.reset_index(drop=True, inplace=True)
            drawdown_df.index = drawdown_df.index + 1
        
        return drawdown_df
        
    def create_visualizations(self):
        """Create comprehensive charts matching the reference document"""
        print("Creating visualizations...")
        
        # Set style
        plt.style.use('default')
        fig = plt.figure(figsize=(24, 20))
        
        colors = {
            'strategy': '#1f77b4',
            'tqqq': '#ff7f0e', 
            'qqq': '#2ca02c',
            'spy': '#d62728'
        }
        
        # 1. Portfolio Value Over Time (Linear)
        ax1 = plt.subplot(4, 3, 1)
        plt.plot(self.backtest_data.index, self.backtest_data['Portfolio_Value'], 
                label='Moving Average Strategy', linewidth=2, color=colors['strategy'])
        plt.plot(self.backtest_data.index, self.backtest_data['BuyHold_TQQQ_Value'], 
                label='Buy & Hold TQQQ', linewidth=2, color=colors['tqqq'], alpha=0.7)
        plt.plot(self.backtest_data.index, self.backtest_data['BuyHold_QQQ_Value'], 
                label='Buy & Hold QQQ', linewidth=2, color=colors['qqq'], alpha=0.7)
        plt.plot(self.backtest_data.index, self.backtest_data['BuyHold_SPY_Value'], 
                label='Buy & Hold SPY', linewidth=2, color=colors['spy'], alpha=0.7)
        plt.title('Portfolio Growth', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Portfolio Value Over Time (Log Scale)
        ax2 = plt.subplot(4, 3, 2)
        plt.semilogy(self.backtest_data.index, self.backtest_data['Portfolio_Value'], 
                    label='Moving Average Strategy', linewidth=2, color=colors['strategy'])
        plt.semilogy(self.backtest_data.index, self.backtest_data['BuyHold_TQQQ_Value'], 
                    label='Buy & Hold TQQQ', linewidth=2, color=colors['tqqq'], alpha=0.7)
        plt.semilogy(self.backtest_data.index, self.backtest_data['BuyHold_QQQ_Value'], 
                    label='Buy & Hold QQQ', linewidth=2, color=colors['qqq'], alpha=0.7)
        plt.semilogy(self.backtest_data.index, self.backtest_data['BuyHold_SPY_Value'], 
                    label='Buy & Hold SPY', linewidth=2, color=colors['spy'], alpha=0.7)
        plt.title('Portfolio Growth (Log Scale)', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. Drawdowns Comparison
        ax3 = plt.subplot(4, 3, 3)
        
        # Calculate drawdowns for all strategies
        for strategy, color, label in [
            ('Portfolio_Value', colors['strategy'], 'Strategy'),
            ('BuyHold_TQQQ_Value', colors['tqqq'], 'TQQQ B&H'),
            ('BuyHold_QQQ_Value', colors['qqq'], 'QQQ B&H'),
            ('BuyHold_SPY_Value', colors['spy'], 'SPY B&H')
        ]:
            peak = self.backtest_data[strategy].expanding(min_periods=1).max()
            dd = (self.backtest_data[strategy] - peak) / peak
            plt.fill_between(self.backtest_data.index, dd*100, 0, alpha=0.6, color=color, label=f'{label}')
        
        plt.title('Drawdowns', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Drawdown (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 4. Position Allocation
        ax4 = plt.subplot(4, 3, 4)
        plt.fill_between(self.backtest_data.index, self.backtest_data['Position']*100, 
                        alpha=0.6, color='green', label='In TQQQ (%)')
        plt.fill_between(self.backtest_data.index, 0, 
                        where=(self.backtest_data['Position'] == 0),
                        alpha=0.6, color='gray', label='In Cash (%)')
        plt.title('Allocation Changes', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Allocation (%)')
        plt.legend()
        plt.ylim(-5, 105)
        plt.grid(True, alpha=0.3)
        
        # 5. QQQ Price vs Moving Average
        ax5 = plt.subplot(4, 3, 5)
        plt.plot(self.backtest_data.index, self.backtest_data['QQQ_Close'], 
                label='QQQ Price', linewidth=1.5, color='black')
        plt.plot(self.backtest_data.index, self.backtest_data['QQQ_MA'], 
                label=f'{self.ma_period}-Day MA', linewidth=2, color='orange')
        
        # Highlight buy/sell signals
        buy_signals = self.backtest_data[self.backtest_data['Signal_Change'] == 1]
        sell_signals = self.backtest_data[self.backtest_data['Signal_Change'] == -1]
        
        plt.scatter(buy_signals.index, buy_signals['QQQ_Close'], 
                   color='green', marker='^', s=50, label='Buy Signal', alpha=0.7)
        plt.scatter(sell_signals.index, sell_signals['QQQ_Close'], 
                   color='red', marker='v', s=50, label='Sell Signal', alpha=0.7)
        
        plt.title('QQQ Price vs Moving Average (Signal)', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 6. Rolling Returns (3-year)
        ax6 = plt.subplot(4, 3, 6)
        
        strategy_rolling_3y = self.backtest_data['Portfolio_Return'].rolling(252*3).apply(
            lambda x: (1 + x).prod() ** (1/3) - 1)
        tqqq_rolling_3y = self.backtest_data['BuyHold_TQQQ_Return'].rolling(252*3).apply(
            lambda x: (1 + x).prod() ** (1/3) - 1)
        
        plt.plot(self.backtest_data.index, strategy_rolling_3y*100, 
                label='Strategy 3Y Rolling Return', linewidth=2, color=colors['strategy'])
        plt.plot(self.backtest_data.index, tqqq_rolling_3y*100, 
                label='TQQQ B&H 3Y Rolling Return', linewidth=2, color=colors['tqqq'], alpha=0.7)
        
        plt.title('Annualized Rolling Return - 3 Years', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('3-Year Rolling Return (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 7. Rolling Returns (5-year)
        ax7 = plt.subplot(4, 3, 7)
        
        strategy_rolling_5y = self.backtest_data['Portfolio_Return'].rolling(252*5).apply(
            lambda x: (1 + x).prod() ** (1/5) - 1)
        tqqq_rolling_5y = self.backtest_data['BuyHold_TQQQ_Return'].rolling(252*5).apply(
            lambda x: (1 + x).prod() ** (1/5) - 1)
        
        plt.plot(self.backtest_data.index, strategy_rolling_5y*100, 
                label='Strategy 5Y Rolling Return', linewidth=2, color=colors['strategy'])
        plt.plot(self.backtest_data.index, tqqq_rolling_5y*100, 
                label='TQQQ B&H 5Y Rolling Return', linewidth=2, color=colors['tqqq'], alpha=0.7)
        
        plt.title('Annualized Rolling Return - 5 Years', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('5-Year Rolling Return (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 8. Annual Returns Bar Chart
        if len(self.annual_returns) > 0:
            ax8 = plt.subplot(4, 3, 8)
            x = np.arange(len(self.annual_returns))
            width = 0.2
            
            strategies_annual = [
                ('Moving Average Strategy', colors['strategy'], 'Strategy'),
                ('Buy & Hold TQQQ', colors['tqqq'], 'TQQQ B&H')
            ]
            
            for i, (col, color, label) in enumerate(strategies_annual):
                if col in self.annual_returns.columns:
                    returns = self.annual_returns[col] * 100
                    plt.bar(x + i*width, returns, width, label=label, alpha=0.8, color=color)
            
            plt.title('Annual Returns', fontsize=14, fontweight='bold')
            plt.xlabel('Year')
            plt.ylabel('Annual Return (%)')
            plt.xticks(x + width/2, self.annual_returns['Year'], rotation=45)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # 9. Monthly Returns Heatmap
        ax9 = plt.subplot(4, 3, 9)
        if len(self.monthly_returns) > 0:
            # Create heatmap data
            monthly_pivot = self.monthly_returns.pivot(index='Year', columns='Month', 
                                                     values='Moving Average Strategy_Return')
            
            if not monthly_pivot.empty:
                im = ax9.imshow(monthly_pivot.values*100, cmap='RdYlGn', aspect='auto', 
                               vmin=-20, vmax=20)
                
                # Set ticks and labels
                ax9.set_xticks(range(12))
                ax9.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
                ax9.set_yticks(range(len(monthly_pivot.index)))
                ax9.set_yticklabels(monthly_pivot.index)
                
                # Add colorbar
                cbar = plt.colorbar(im, ax=ax9)
                cbar.set_label('Monthly Return (%)')
        
        plt.title('Monthly Returns Heatmap', fontsize=14, fontweight='bold')
        
        # 10. Return Distribution
        ax10 = plt.subplot(4, 3, 10)
        
        strategy_returns = self.backtest_data['Portfolio_Return'].dropna() * 100
        tqqq_returns = self.backtest_data['BuyHold_TQQQ_Return'].dropna() * 100
        
        plt.hist(strategy_returns, bins=50, alpha=0.7, label='Strategy', 
                color=colors['strategy'], density=True)
        plt.hist(tqqq_returns, bins=50, alpha=0.5, label='TQQQ B&H', 
                color=colors['tqqq'], density=True)
        plt.title('Daily Returns Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Daily Return (%)')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 11. Risk-Return Scatter
        ax11 = plt.subplot(4, 3, 11)
        
        strategies_risk_return = ['Moving Average Strategy', 'Buy & Hold TQQQ', 'Buy & Hold QQQ', 'Buy & Hold SPY']
        colors_list = [colors['strategy'], colors['tqqq'], colors['qqq'], colors['spy']]
        
        for strategy_name, color in zip(strategies_risk_return, colors_list):
            if strategy_name in self.performance_stats:
                cagr = float(self.performance_stats[strategy_name]['CAGR'].replace('%', ''))
                vol = float(self.performance_stats[strategy_name]['Annual_Volatility'].replace('%', ''))
                
                plt.scatter(vol, cagr, s=100, color=color, alpha=0.7, 
                           label=strategy_name)
                
                # Add labels
                plt.annotate(strategy_name, (vol, cagr), xytext=(5, 5), 
                           textcoords='offset points', fontsize=8)
        
        plt.title('Risk-Return Profile', fontsize=14, fontweight='bold')
        plt.xlabel('Annual Volatility (%)')
        plt.ylabel('CAGR (%)')
        plt.grid(True, alpha=0.3)
        
        # 12. Underwater Curve (Drawdown)
        ax12 = plt.subplot(4, 3, 12)
        
        strategy_peak = self.backtest_data['Portfolio_Value'].expanding(min_periods=1).max()
        strategy_underwater = (self.backtest_data['Portfolio_Value'] - strategy_peak) / strategy_peak * 100
        
        plt.fill_between(self.backtest_data.index, strategy_underwater, 0, 
                        alpha=0.7, color=colors['strategy'], label='Strategy Underwater Curve')
        plt.title('Underwater Curve (Strategy)', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Drawdown (%)')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        plt.tight_layout()
        self.fig = fig
        print("Visualizations created.")
        
    def save_to_excel(self, filename='TQQQ_Strategy_Comprehensive_Results.xlsx'):
        """Save all comprehensive results to Excel file"""
        print(f"Saving comprehensive results to {filename}...")
        
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            
            # 1. Performance Summary (Main metrics like in the document)
            perf_df = pd.DataFrame(self.performance_stats).T
            perf_df.to_excel(writer, sheet_name='Performance Summary')
            
            # 2. Risk and Return Metrics (Comprehensive table from document)
            risk_metrics = []
            main_strategies = ['Moving Average Strategy', 'Buy & Hold TQQQ']
            
            for strategy in main_strategies:
                if strategy in self.performance_stats:
                    stats = self.performance_stats[strategy]
                    risk_metrics.append({
                        'Metric': 'Arithmetic Mean (monthly)',
                        strategy: stats.get('Arithmetic_Mean_Monthly', 'N/A')
                    })
                    
            # Create comprehensive risk metrics table
            risk_return_metrics = pd.DataFrame([
                ['Arithmetic Mean (monthly)', 
                 self.performance_stats.get('Moving Average Strategy', {}).get('Arithmetic_Mean_Monthly', 'N/A'),
                 self.performance_stats.get('Buy & Hold TQQQ', {}).get('Arithmetic_Mean_Monthly', 'N/A')],
                ['Arithmetic Mean (annualized)', 
                 self.performance_stats.get('Moving Average Strategy', {}).get('Arithmetic_Mean_Annual', 'N/A'),
                 self.performance_stats.get('Buy & Hold TQQQ', {}).get('Arithmetic_Mean_Annual', 'N/A')],
                ['Geometric Mean (monthly)', 
                 self.performance_stats.get('Moving Average Strategy', {}).get('Geometric_Mean_Monthly', 'N/A'),
                 self.performance_stats.get('Buy & Hold TQQQ', {}).get('Geometric_Mean_Monthly', 'N/A')],
                ['Geometric Mean (annualized)', 
                 self.performance_stats.get('Moving Average Strategy', {}).get('Geometric_Mean_Annual', 'N/A'),
                 self.performance_stats.get('Buy & Hold TQQQ', {}).get('Geometric_Mean_Annual', 'N/A')],
                ['Standard Deviation (monthly)', 
                 self.performance_stats.get('Moving Average Strategy', {}).get('Monthly_Volatility', 'N/A'),
                 self.performance_stats.get('Buy & Hold TQQQ', {}).get('Monthly_Volatility', 'N/A')],
                ['Standard Deviation (annualized)', 
                 self.performance_stats.get('Moving Average Strategy', {}).get('Annual_Volatility', 'N/A'),
                 self.performance_stats.get('Buy & Hold TQQQ', {}).get('Annual_Volatility', 'N/A')],
                ['Downside Deviation (monthly)', 
                 self.performance_stats.get('Moving Average Strategy', {}).get('Downside_Deviation_Monthly', 'N/A'),
                 self.performance_stats.get('Buy & Hold TQQQ', {}).get('Downside_Deviation_Monthly', 'N/A')],
                ['Maximum Drawdown', 
                 self.performance_stats.get('Moving Average Strategy', {}).get('Max_Drawdown', 'N/A'),
                 self.performance_stats.get('Buy & Hold TQQQ', {}).get('Max_Drawdown', 'N/A')],
                ['Benchmark Correlation', 
                 self.performance_stats.get('Moving Average Strategy', {}).get('Correlation', 'N/A'),
                 self.performance_stats.get('Buy & Hold TQQQ', {}).get('Correlation', 'N/A')],
                ['Beta', 
                 self.performance_stats.get('Moving Average Strategy', {}).get('Beta', 'N/A'),
                 self.performance_stats.get('Buy & Hold TQQQ', {}).get('Beta', 'N/A')],
                ['Alpha (annualized)', 
                 self.performance_stats.get('Moving Average Strategy', {}).get('Alpha_Annual', 'N/A'),
                 self.performance_stats.get('Buy & Hold TQQQ', {}).get('Alpha_Annual', 'N/A')],
                ['R²', 
                 self.performance_stats.get('Moving Average Strategy', {}).get('R_Squared', 'N/A'),
                 self.performance_stats.get('Buy & Hold TQQQ', {}).get('R_Squared', 'N/A')],
                ['Sharpe Ratio', 
                 self.performance_stats.get('Moving Average Strategy', {}).get('Sharpe_Ratio', 'N/A'),
                 self.performance_stats.get('Buy & Hold TQQQ', {}).get('Sharpe_Ratio', 'N/A')],
                ['Sortino Ratio', 
                 self.performance_stats.get('Moving Average Strategy', {}).get('Sortino_Ratio', 'N/A'),
                 self.performance_stats.get('Buy & Hold TQQQ', {}).get('Sortino_Ratio', 'N/A')],
                ['Treynor Ratio (%)', 
                 self.performance_stats.get('Moving Average Strategy', {}).get('Treynor_Ratio', 'N/A'),
                 self.performance_stats.get('Buy & Hold TQQQ', {}).get('Treynor_Ratio', 'N/A')],
                ['Calmar Ratio', 
                 self.performance_stats.get('Moving Average Strategy', {}).get('Calmar_Ratio', 'N/A'),
                 self.performance_stats.get('Buy & Hold TQQQ', {}).get('Calmar_Ratio', 'N/A')],
                ['Information Ratio', 
                 self.performance_stats.get('Moving Average Strategy', {}).get('Information_Ratio', 'N/A'),
                 self.performance_stats.get('Buy & Hold TQQQ', {}).get('Information_Ratio', 'N/A')],
                ['Skewness', 
                 self.performance_stats.get('Moving Average Strategy', {}).get('Skewness', 'N/A'),
                 self.performance_stats.get('Buy & Hold TQQQ', {}).get('Skewness', 'N/A')],
                ['Excess Kurtosis', 
                 self.performance_stats.get('Moving Average Strategy', {}).get('Excess_Kurtosis', 'N/A'),
                 self.performance_stats.get('Buy & Hold TQQQ', {}).get('Excess_Kurtosis', 'N/A')],
                ['Historical Value-at-Risk (5%)', 
                 self.performance_stats.get('Moving Average Strategy', {}).get('Historical_VaR_5pct', 'N/A'),
                 self.performance_stats.get('Buy & Hold TQQQ', {}).get('Historical_VaR_5pct', 'N/A')],
                ['Analytical Value-at-Risk (5%)', 
                 self.performance_stats.get('Moving Average Strategy', {}).get('Analytical_VaR_5pct', 'N/A'),
                 self.performance_stats.get('Buy & Hold TQQQ', {}).get('Analytical_VaR_5pct', 'N/A')],
                ['Conditional Value-at-Risk (5%)', 
                 self.performance_stats.get('Moving Average Strategy', {}).get('Conditional_VaR_5pct', 'N/A'),
                 self.performance_stats.get('Buy & Hold TQQQ', {}).get('Conditional_VaR_5pct', 'N/A')],
                ['Upside Capture Ratio (%)', 
                 self.performance_stats.get('Moving Average Strategy', {}).get('Upside_Capture_Ratio', 'N/A'),
                 self.performance_stats.get('Buy & Hold TQQQ', {}).get('Upside_Capture_Ratio', 'N/A')],
                ['Downside Capture Ratio (%)', 
                 self.performance_stats.get('Moving Average Strategy', {}).get('Downside_Capture_Ratio', 'N/A'),
                 self.performance_stats.get('Buy & Hold TQQQ', {}).get('Downside_Capture_Ratio', 'N/A')],
                ['Positive Periods', 
                 self.performance_stats.get('Moving Average Strategy', {}).get('Positive_Periods', 'N/A'),
                 self.performance_stats.get('Buy & Hold TQQQ', {}).get('Positive_Periods', 'N/A')],
                ['Gain/Loss Ratio', 
                 self.performance_stats.get('Moving Average Strategy', {}).get('Gain_Loss_Ratio', 'N/A'),
                 self.performance_stats.get('Buy & Hold TQQQ', {}).get('Gain_Loss_Ratio', 'N/A')]
            ], columns=['Metric', 'Moving Average Model', 'Buy & Hold Portfolio'])
            
            risk_return_metrics.to_excel(writer, sheet_name='Risk and Return Metrics', index=False)
            
            # 3. Annual Returns
            if hasattr(self, 'annual_returns') and len(self.annual_returns) > 0:
                annual_formatted = self.annual_returns.copy()
                # Format returns as percentages
                for col in annual_formatted.columns:
                    if col != 'Year' and col != 'Inflation' and col != 'Treasury_Rate':
                        if col in annual_formatted.columns:
                            annual_formatted[col] = annual_formatted[col].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")
                
                annual_formatted.to_excel(writer, sheet_name='Annual Returns', index=False)
            
            # 4. Monthly Returns
            if hasattr(self, 'monthly_returns') and len(self.monthly_returns) > 0:
                monthly_formatted = self.monthly_returns.copy()
                # Format returns as percentages and balances as currency
                for col in monthly_formatted.columns:
                    if 'Return' in col:
                        monthly_formatted[col] = monthly_formatted[col].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")
                    elif 'Balance' in col:
                        monthly_formatted[col] = monthly_formatted[col].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "N/A")
                
                monthly_formatted.to_excel(writer, sheet_name='Monthly Returns', index=False)
            
            # 5. Trailing Returns
            if hasattr(self, 'trailing_returns') and len(self.trailing_returns) > 0:
                trailing_formatted = self.trailing_returns.copy()
                # Format returns as percentages
                for col in trailing_formatted.columns:
                    if 'Return' in col and col != 'Strategy':
                        trailing_formatted[col] = trailing_formatted[col].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")
                
                trailing_formatted.to_excel(writer, sheet_name='Trailing Returns', index=False)
            
            # 6. Rolling Returns Summary
            if hasattr(self, 'rolling_returns_summary') and len(self.rolling_returns_summary) > 0:
                rolling_formatted = self.rolling_returns_summary.copy()
                # Format returns as percentages
                for col in rolling_formatted.columns:
                    if col != 'Strategy':
                        rolling_formatted[col] = rolling_formatted[col].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")
                
                rolling_formatted.to_excel(writer, sheet_name='Rolling Returns', index=False)
            
            # 7. Risk Management Performance
            if hasattr(self, 'risk_management_performance'):
                self.risk_management_performance.to_excel(writer, sheet_name='Risk Management Performance', index=False)
            
            # 8. Trading Performance
            if hasattr(self, 'trading_performance') and self.trading_performance:
                trading_df = pd.DataFrame([self.trading_performance]).T
                trading_df.columns = ['Value']
                trading_df.index.name = 'Metric'
                trading_df.to_excel(writer, sheet_name='Trading Performance')
            
            # 9. Detailed Trade Log
            if len(self.trade_log) > 0:
                trade_log_formatted = self.trade_log.copy()
                
                # Format monetary columns
                for col in ['Value', 'Transaction_Cost', 'Slippage_Cost']:
                    if col in trade_log_formatted.columns:
                        trade_log_formatted[col] = trade_log_formatted[col].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "N/A")
                
                # Format price columns
                for col in ['Price', 'Effective_Price']:
                    if col in trade_log_formatted.columns:
                        trade_log_formatted[col] = trade_log_formatted[col].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "N/A")
                
                # Format shares
                if 'Shares' in trade_log_formatted.columns:
                    trade_log_formatted['Shares'] = trade_log_formatted['Shares'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
                
                trade_log_formatted.to_excel(writer, sheet_name='Detailed Trade Log', index=False)
            
            # 10. Drawdown Analysis
            dd_data = self.calculate_detailed_drawdowns()
            if not dd_data.empty:
                dd_data.to_excel(writer, sheet_name='Drawdown Analysis', index=True)
            
            # 11. Daily Data (Full dataset)
            daily_export = self.backtest_data[[
                'TQQQ_Close', 'QQQ_Close', 'SPY_Close', 'QQQ_MA', 'Signal', 'Position',
                'Portfolio_Value', 'BuyHold_TQQQ_Value', 'BuyHold_QQQ_Value', 'BuyHold_SPY_Value',
                'Portfolio_Return', 'BuyHold_TQQQ_Return', 'BuyHold_QQQ_Return', 'BuyHold_SPY_Return',
                'Treasury_Return'
            ]].copy()
            daily_export.to_excel(writer, sheet_name='Daily Data')
            
        # Save chart as image
        chart_filename = filename.replace('.xlsx', '_Charts.png')
        self.fig.savefig(chart_filename, dpi=300, bbox_inches='tight')
        
        print(f"Comprehensive results saved to {filename}")
        print(f"Charts saved to {chart_filename}")
        
    def print_summary(self):
        """Print comprehensive strategy summary matching the reference document"""
        print("\n" + "="*120)
        print("TACTICAL ASSET ALLOCATION - TQQQ MOVING AVERAGE STRATEGY")
        print("="*120)
        
        print("\nMODEL CONFIGURATION:")
        print("-" * 50)
        print(f"Tactical Model: Moving Averages for Asset")
        print(f"Ticker: TQQQ (ProShares UltraPro QQQ)")
        print(f"Signal Asset: QQQ (Invesco QQQ Trust)")
        print(f"Moving Average Type: Simple Moving Average")
        print(f"Lookback Period: {self.ma_period} trading days (8 months)")
        print(f"Buy Signal: Price >= Moving Average")
        print(f"Out of Market Asset: Cash (Treasury Rate)")
        print(f"Trade Execution: End of day price")
        print(f"Transaction Cost: ${self.transaction_cost_per_share} per share + {self.slippage:.2%} slippage")
        print(f"Initial Amount: ${self.initial_capital:,.2f}")
        
        print(f"\nMODEL SIMULATION RESULTS:")
        print("-" * 50)
        print(f"Period: {self.backtest_data.index[0].strftime('%b %Y')} - {self.backtest_data.index[-1].strftime('%b %Y')}")
        print(f"Total Trading Days: {len(self.backtest_data):,}")
        
        if hasattr(self, 'performance_stats') and 'Moving Average Strategy' in self.performance_stats:
            strategy_stats = self.performance_stats['Moving Average Strategy']
            benchmark_stats = self.performance_stats.get('Buy & Hold TQQQ', {})
            
            print(f"\nPERFORMANCE SUMMARY:")
            print("-" * 120)
            print(f"{'Metric':<35} {'Moving Average Model':<25} {'Buy & Hold Portfolio':<25}")
            print("-" * 120)
            
            metrics = [
                ('Start Balance', 'Start_Balance', 'Start_Balance'),
                ('End Balance', 'End_Balance', 'End_Balance'),
                ('Annualized Return (CAGR)', 'CAGR', 'CAGR'),
                ('Standard Deviation', 'Annual_Volatility', 'Annual_Volatility'),
                ('Best Year', 'Best_Year', 'Best_Year'),
                ('Worst Year', 'Worst_Year', 'Worst_Year'),
                ('Maximum Drawdown', 'Max_Drawdown', 'Max_Drawdown'),
                ('Sharpe Ratio', 'Sharpe_Ratio', 'Sharpe_Ratio'),
                ('Sortino Ratio', 'Sortino_Ratio', 'Sortino_Ratio')
            ]
            
            for display_name, strategy_key, benchmark_key in metrics:
                strategy_val = strategy_stats.get(strategy_key, 'N/A')
                benchmark_val = benchmark_stats.get(benchmark_key, 'N/A')
                print(f"{display_name:<35} {strategy_val:<25} {benchmark_val:<25}")
        
        # Risk Management Performance
        if hasattr(self, 'risk_management_performance'):
            print(f"\nRISK MANAGEMENT PERFORMANCE:")
            print("-" * 80)
            print(f"{'Type':<15} {'Periods':<15} {'Periods %':<15} {'Return':<15} {'Stdev':<15}")
            print("-" * 80)
            
            for _, row in self.risk_management_performance.iterrows():
                print(f"{row['Type']:<15} {row['Periods_Count']:<15} {row['Periods_Percent']:<15} {row['Return_Annual']:<15} {row['Stdev_Annual']:<15}")
        
        # Trading Performance
        if hasattr(self, 'trading_performance') and self.trading_performance:
            print(f"\nTRADING PERFORMANCE:")
            print("-" * 80)
            print(f"Total Trades: {self.trading_performance.get('Total_Trades', 'N/A')}")
            print(f"Average Annual Turnover: {self.trading_performance.get('Average_Annual_Turnover', 'N/A')}")
            print(f"Cumulative Return without Costs: {self.trading_performance.get('Cumulative_Trading_Return_Without_Costs', 'N/A')}")
            print(f"Cumulative Return after Costs: {self.trading_performance.get('Cumulative_Trading_Return_After_Costs', 'N/A')}")
            print(f"Trading Cost Impact: {self.trading_performance.get('Cumulative_Trading_Cost_Impact', 'N/A')}")
        
        print("\n" + "="*120)
        
    def run_full_analysis(self):
        """Run complete comprehensive strategy analysis"""
        self.download_data()
        self.calculate_signals()
        self.backtest_strategy()
        self.calculate_benchmarks()
        self.calculate_performance_metrics()
        self.calculate_annual_returns()
        self.calculate_monthly_returns()
        self.calculate_trailing_returns()
        self.calculate_rolling_returns_analysis()
        self.calculate_risk_management_performance()
        self.calculate_trading_performance()
        self.create_visualizations()
        
        self.print_summary()
        self.save_to_excel()
        
        return self

# Run the analysis
if __name__ == "__main__":
    # Initialize and run strategy with parameters matching the reference document
    strategy = TQQQMovingAverageStrategy(
        initial_capital=10000,
        ma_period=168,  # 8 months ≈ 168 trading days
        transaction_cost_per_share=0.005,  # $0.005 per share
        slippage=0.0015  # 0.15% slippage
    )
    
    strategy.run_full_analysis()
    
    print("\nComprehensive analysis complete!")
    print("Generated files:")
    print("- TQQQ_Strategy_Comprehensive_Results.xlsx (11 detailed sheets)")
    print("- TQQQ_Strategy_Comprehensive_Results_Charts.png (12 comprehensive charts)")