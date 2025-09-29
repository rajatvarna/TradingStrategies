"""
TQQQ Multi-Strategy Backtesting System
=====================================

Complete implementation of FTLT1-4 strategies for TQQQ trading.
Downloads data from Yahoo Finance and generates comprehensive Excel reports.

Required packages:
pip install yfinance pandas numpy matplotlib seaborn openpyxl

Usage:
python tqqq_backtest.py
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class TQQQStrategies:
    def __init__(self, start_date='2010-01-01', initial_capital=100000):
        self.start_date = start_date
        self.end_date = datetime.now().strftime('%Y-%m-%d')
        self.initial_capital = initial_capital
        self.symbols = ['TQQQ', 'QQQ', 'SPY', 'UVXY', 'SQQQ']
        self.data = {}
        self.results = {}
        
    def download_data(self):
        """Download data from Yahoo Finance"""
        print("Downloading data from Yahoo Finance...")
        for symbol in self.symbols:
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(start=self.start_date, end=self.end_date)
                if len(data) > 0:
                    self.data[symbol] = data
                    print(f"Downloaded {len(data)} bars for {symbol}")
                else:
                    print(f"No data found for {symbol}")
            except Exception as e:
                print(f"Error downloading {symbol}: {e}")
        
        # Align all data to same dates
        if self.data:
            common_dates = None
            for symbol, data in self.data.items():
                if common_dates is None:
                    common_dates = data.index
                else:
                    common_dates = common_dates.intersection(data.index)
            
            # Trim all data to common dates
            for symbol in self.data:
                self.data[symbol] = self.data[symbol].loc[common_dates]
            
            print(f"Aligned data: {len(common_dates)} common trading days")
    
    def calculate_indicators(self):
        """Calculate technical indicators for all symbols"""
        print("Calculating technical indicators...")
        
        for symbol, data in self.data.items():
            # Moving averages
            data['MA_20'] = data['Close'].rolling(window=20).mean()
            data['MA_200'] = data['Close'].rolling(window=200).mean()
            
            # RSI calculation
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=10).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=10).mean()
            rs = gain / loss
            data['RSI_10'] = 100 - (100 / (1 + rs))
            
            # Trend indicators
            data['uptrend'] = data['Close'] > data['MA_200']
            data['downtrend'] = data['Close'] < data['MA_200']
            
        # Market regime indicators using SPY
        if 'SPY' in self.data:
            spy_data = self.data['SPY']
            bullmkt = spy_data['uptrend']
            bearmkt = spy_data['downtrend']
            
            # Add market regime to all symbols
            for symbol in self.data:
                self.data[symbol]['bullmkt'] = bullmkt
                self.data[symbol]['bearmkt'] = bearmkt
        
        # TQQQ overextended signal for UVXY strategy
        if 'TQQQ' in self.data:
            tqqq_overextended = self.data['TQQQ']['RSI_10'] > 79
            for symbol in self.data:
                self.data[symbol]['tqqq_overextended'] = tqqq_overextended
    
    def run_strategy(self, strategy_name, symbol, entry_condition, exit_condition):
        """Run a single strategy and return results"""
        if symbol not in self.data:
            print(f"No data for {symbol}")
            return None
            
        data = self.data[symbol].copy()
        
        # Initialize strategy columns
        data['position'] = 0
        data['entry_signal'] = entry_condition
        data['exit_signal'] = exit_condition
        data['trades'] = 0
        data['returns'] = 0
        data['cumulative_returns'] = 1.0
        
        # Track position and trades
        position = 0
        entry_price = 0
        trades = []
        entry_index = None
        
        for i in range(len(data)):
            current_price = data.iloc[i]['Close']
            
            # Entry logic
            if position == 0 and data.iloc[i]['entry_signal']:
                position = 1
                entry_price = current_price
                entry_index = i
                data.iloc[i, data.columns.get_loc('position')] = 1
                data.iloc[i, data.columns.get_loc('trades')] = 1
                
            # Exit logic
            elif position == 1 and data.iloc[i]['exit_signal']:
                exit_price = current_price
                trade_return = (exit_price - entry_price) / entry_price
                data.iloc[i, data.columns.get_loc('returns')] = trade_return
                
                trades.append({
                    'entry_date': data.index[entry_index],
                    'exit_date': data.index[i],
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'return': trade_return,
                    'days_held': (data.index[i] - data.index[entry_index]).days
                })
                
                position = 0
                entry_index = None
                data.iloc[i, data.columns.get_loc('position')] = 0
            
            # Maintain position
            elif position == 1:
                data.iloc[i, data.columns.get_loc('position')] = 1
        
        # Calculate cumulative returns
        data['cumulative_returns'] = (1 + data['returns']).cumprod()
        
        return {
            'data': data,
            'trades': pd.DataFrame(trades),
            'strategy_name': strategy_name,
            'symbol': symbol
        }
    
    def run_all_strategies(self):
        """
        Run all four TQQQ-based trading strategies.
        
        STRATEGY EXPLANATIONS:
        ======================
        
        FTLT1 - BULL MARKET MOMENTUM STRATEGY:
        - Asset: TQQQ (3x leveraged NASDAQ ETF)
        - Entry: Bull market (SPY > 200MA) AND RSI(10) < 79 (not overbought)
        - Exit: Bear market (SPY < 200MA) OR RSI(10) > 79 (overbought)
        - Logic: Ride the bull market momentum but avoid overextended periods
        - Risk: High volatility due to 3x leverage, significant drawdowns in corrections
        
        FTLT2 - BEAR MARKET COUNTER-TREND STRATEGY:
        - Asset: TQQQ (3x leveraged NASDAQ ETF) 
        - Entry: Bear market (SPY < 200MA) AND RSI(10) < 31 (oversold)
        - Exit: Bull market (SPY > 200MA) OR RSI(10) > 69 (no longer oversold)
        - Logic: Buy TQQQ when severely oversold during bear markets (dead cat bounces)
        - Risk: Catching falling knives, potential for large losses in sustained downtrends
        
        FTLT3 - BEAR MARKET MEAN REVERSION STRATEGY:
        - Asset: TQQQ (3x leveraged NASDAQ ETF)
        - Entry: Bear market (SPY < 200MA) AND above 20MA AND RSI(10) > 31
        - Exit: Bull market OR RSI(10) < 31 OR below 20MA
        - Logic: Trade bounces during bear markets when price shows relative strength
        - Risk: Bear market rallies can be short-lived, whipsaws possible
        
        FTLT4 - VOLATILITY HEDGE STRATEGY:
        - Asset: UVXY (volatility ETF, typically inverse to equity markets)
        - Entry: Bull market (SPY > 200MA) AND TQQQ RSI(10) > 79 (overextended)
        - Exit: TQQQ RSI(10) <= 79 (no longer overextended)
        - Logic: Hedge against market corrections when TQQQ is overextended
        - Risk: Volatility decay in UVXY, false signals, timing-dependent
        
        WHY COMBINE FTLT1 + FTLT3?
        ==========================
        - FTLT1: Captures bull market trends (primary return driver)
        - FTLT3: Captures bear market bounces (diversification, lower correlation)
        - Together: More consistent returns across market cycles
        - Benefit: Reduced overall portfolio volatility and drawdowns
        """
        print("Running all strategies...")
        
        # =================================================================
        # STRATEGY FTLT1: BULL MARKET MOMENTUM
        # =================================================================
        # Long TQQQ during bull markets when not overbought
        # This is typically the primary return driver in the combined portfolio
        if 'TQQQ' in self.data:
            tqqq_data = self.data['TQQQ']
            # Entry: Bull market AND RSI not overbought
            entry1 = (tqqq_data['bullmkt'] == True) & (tqqq_data['RSI_10'] < 79)
            # Exit: Bear market OR RSI overbought  
            exit1 = (tqqq_data['bullmkt'] == False) | (tqqq_data['RSI_10'] > 79)
            self.results['FTLT1'] = self.run_strategy('FTLT1', 'TQQQ', entry1, exit1)
        
        # =================================================================
        # STRATEGY FTLT2: BEAR MARKET COUNTER-TREND
        # =================================================================
        # Long TQQQ when severely oversold during bear markets
        # Attempts to capture dead cat bounces and bear market rallies
        if 'TQQQ' in self.data:
            tqqq_data = self.data['TQQQ']
            # Entry: Bear market AND severely oversold
            entry2 = (tqqq_data['bearmkt'] == True) & (tqqq_data['RSI_10'] < 31)
            # Exit: Bull market OR no longer oversold
            exit2 = (tqqq_data['bearmkt'] == False) | (tqqq_data['RSI_10'] > 69)
            self.results['FTLT2'] = self.run_strategy('FTLT2', 'TQQQ', entry2, exit2)
        
        # =================================================================
        # STRATEGY FTLT3: BEAR MARKET MEAN REVERSION  
        # =================================================================
        # Long TQQQ during bear markets when showing relative strength
        # Complements FTLT1 by trading during bear market periods
        if 'TQQQ' in self.data:
            tqqq_data = self.data['TQQQ']
            # Entry: Bear market AND above 20MA (relative strength) AND RSI > 31
            entry3 = (tqqq_data['bearmkt'] == True) & (tqqq_data['Close'] > tqqq_data['MA_20']) & (tqqq_data['RSI_10'] > 31)
            # Exit: Bull market OR RSI drops OR below 20MA (weakness)
            exit3 = (tqqq_data['bearmkt'] == False) | (tqqq_data['RSI_10'] < 31) | (tqqq_data['Close'] < tqqq_data['MA_20'])
            self.results['FTLT3'] = self.run_strategy('FTLT3', 'TQQQ', entry3, exit3)
        
        # =================================================================
        # STRATEGY FTLT4: VOLATILITY HEDGE
        # =================================================================
        # Long UVXY when TQQQ is overextended during bull markets
        # Provides portfolio protection during market corrections
        if 'UVXY' in self.data:
            uvxy_data = self.data['UVXY']
            # Entry: Bull market AND TQQQ overextended (hedge position)
            entry4 = (uvxy_data['bullmkt'] == True) & (uvxy_data['tqqq_overextended'] == True)
            # Exit: TQQQ no longer overextended
            exit4 = (uvxy_data['tqqq_overextended'] == False)
            self.results['FTLT4'] = self.run_strategy('FTLT4', 'UVXY', entry4, exit4)
    
    def calculate_statistics(self):
        """
        Calculate comprehensive performance statistics for all strategies.
        
        This method computes detailed metrics for:
        - FTLT1: Bull market TQQQ strategy (long during uptrends when RSI < 79)
        - FTLT2: Bear market TQQQ counter-trend strategy (long when oversold in downtrends) 
        - FTLT3: Bear market TQQQ mean reversion strategy (long above 20MA during downtrends)
        - FTLT4: Volatility strategy (long UVXY when TQQQ overextended during bull markets)
        
        Returns:
            dict: Comprehensive statistics for each strategy including return distributions,
                  risk metrics, trade analysis, and temporal return breakdowns
        """
        stats = {}
        
        for strategy_name, result in self.results.items():
            if result is None:
                continue
                
            trades = result['trades']
            data = result['data']
            
            if len(trades) == 0:
                stats[strategy_name] = {'total_trades': 0}
                continue
            
            # =================================================================
            # BASIC TRADE STATISTICS
            # =================================================================
            total_return = data['cumulative_returns'].iloc[-1] - 1
            winning_trades = trades[trades['return'] > 0]
            losing_trades = trades[trades['return'] <= 0]
            
            win_rate = len(winning_trades) / len(trades) if len(trades) > 0 else 0
            avg_win = winning_trades['return'].mean() if len(winning_trades) > 0 else 0
            avg_loss = losing_trades['return'].mean() if len(losing_trades) > 0 else 0
            avg_trade = trades['return'].mean()
            
            # =================================================================
            # RETURN DISTRIBUTION STATISTICS (as requested multiple times)
            # =================================================================
            # Convert trade returns to percentage for better readability
            trade_returns_pct = trades['return'] * 100
            
            # Return distribution metrics
            return_std = trade_returns_pct.std()
            return_skew = trade_returns_pct.skew() if len(trade_returns_pct) > 2 else 0
            return_kurtosis = trade_returns_pct.kurtosis() if len(trade_returns_pct) > 3 else 0
            
            # Return percentiles for distribution analysis
            return_percentiles = {
                '5th_percentile': trade_returns_pct.quantile(0.05) if len(trade_returns_pct) > 0 else 0,
                '25th_percentile': trade_returns_pct.quantile(0.25) if len(trade_returns_pct) > 0 else 0,
                'median_return': trade_returns_pct.median() if len(trade_returns_pct) > 0 else 0,
                '75th_percentile': trade_returns_pct.quantile(0.75) if len(trade_returns_pct) > 0 else 0,
                '95th_percentile': trade_returns_pct.quantile(0.95) if len(trade_returns_pct) > 0 else 0
            }
            
            # Consecutive win/loss streaks
            consecutive_wins = 0
            consecutive_losses = 0
            max_consecutive_wins = 0
            max_consecutive_losses = 0
            current_win_streak = 0
            current_loss_streak = 0
            
            for trade_return in trades['return']:
                if trade_return > 0:
                    current_win_streak += 1
                    current_loss_streak = 0
                    max_consecutive_wins = max(max_consecutive_wins, current_win_streak)
                else:
                    current_loss_streak += 1
                    current_win_streak = 0
                    max_consecutive_losses = max(max_consecutive_losses, current_loss_streak)
            
            # =================================================================
            # TEMPORAL RETURN ANALYSIS
            # =================================================================
            # Calculate daily portfolio returns for temporal analysis
            daily_returns = data['Close'].pct_change() * data['position'].shift(1)
            daily_returns = daily_returns.fillna(0)
            
            # Monthly returns analysis
            monthly_returns = daily_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
            monthly_returns = monthly_returns[monthly_returns != 0]  # Remove months with no activity
            
            # Annual returns breakdown
            annual_returns = daily_returns.resample('Y').apply(lambda x: (1 + x).prod() - 1)
            annual_returns = annual_returns[annual_returns != 0]  # Remove years with no activity
            
            # Return statistics by time period
            monthly_stats = {
                'avg_monthly_return': monthly_returns.mean() if len(monthly_returns) > 0 else 0,
                'monthly_volatility': monthly_returns.std() if len(monthly_returns) > 1 else 0,
                'positive_months': len(monthly_returns[monthly_returns > 0]) if len(monthly_returns) > 0 else 0,
                'negative_months': len(monthly_returns[monthly_returns < 0]) if len(monthly_returns) > 0 else 0,
                'best_month': monthly_returns.max() if len(monthly_returns) > 0 else 0,
                'worst_month': monthly_returns.min() if len(monthly_returns) > 0 else 0
            }
            
            annual_stats = {
                'avg_annual_return': annual_returns.mean() if len(annual_returns) > 0 else 0,
                'annual_volatility': annual_returns.std() if len(annual_returns) > 1 else 0,
                'positive_years': len(annual_returns[annual_returns > 0]) if len(annual_returns) > 0 else 0,
                'negative_years': len(annual_returns[annual_returns < 0]) if len(annual_returns) > 0 else 0,
                'best_year': annual_returns.max() if len(annual_returns) > 0 else 0,
                'worst_year': annual_returns.min() if len(annual_returns) > 0 else 0
            }
            
            # =================================================================
            # RISK METRICS AND ANNUALIZED CALCULATIONS
            # =================================================================
            total_days = len(data)
            years = total_days / 252
            annualized_return = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
            
            # Risk metrics using daily portfolio returns
            daily_volatility = daily_returns.std()
            annualized_volatility = daily_volatility * np.sqrt(252) if daily_volatility > 0 else 0
            
            # Proper Sharpe ratio (assuming 0% risk-free rate)
            sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else 0
            
            # Maximum drawdown calculation with detailed analysis
            cumulative_returns = data['cumulative_returns']
            rolling_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = abs(drawdown.min())
            
            # Drawdown duration analysis
            in_drawdown = drawdown < -0.01  # Consider 1% as start of meaningful drawdown
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
            
            # =================================================================
            # PROFIT AND LOSS ANALYSIS
            # =================================================================
            gross_profit = winning_trades['return'].sum() if len(winning_trades) > 0 else 0
            gross_loss = abs(losing_trades['return'].sum()) if len(losing_trades) > 0 else 0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # Calmar ratio (return/max drawdown)
            calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0
            
            # =================================================================
            # COMPREHENSIVE STATISTICS DICTIONARY
            # =================================================================
            stats[strategy_name] = {
                # Basic trade metrics
                'total_trades': len(trades),
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': win_rate,
                
                # Return metrics
                'total_return': total_return,
                'annualized_return': annualized_return,
                'avg_trade_return': avg_trade,
                'avg_winning_trade': avg_win,
                'avg_losing_trade': avg_loss,
                'largest_win': trades['return'].max() if len(trades) > 0 else 0,
                'largest_loss': trades['return'].min() if len(trades) > 0 else 0,
                
                # Return distribution statistics (ADDED AS REQUESTED)
                'return_std': return_std / 100,  # Convert back to decimal
                'return_skew': return_skew,
                'return_kurtosis': return_kurtosis,
                'return_5th_percentile': return_percentiles['5th_percentile'] / 100,
                'return_25th_percentile': return_percentiles['25th_percentile'] / 100,
                'return_median': return_percentiles['median_return'] / 100,
                'return_75th_percentile': return_percentiles['75th_percentile'] / 100,
                'return_95th_percentile': return_percentiles['95th_percentile'] / 100,
                
                # Streak analysis
                'max_consecutive_wins': max_consecutive_wins,
                'max_consecutive_losses': max_consecutive_losses,
                
                # Temporal return analysis (ADDED AS REQUESTED)
                'avg_monthly_return': monthly_stats['avg_monthly_return'],
                'monthly_volatility': monthly_stats['monthly_volatility'],
                'positive_months': monthly_stats['positive_months'],
                'negative_months': monthly_stats['negative_months'],
                'best_month': monthly_stats['best_month'],
                'worst_month': monthly_stats['worst_month'],
                'avg_annual_return': annual_stats['avg_annual_return'],
                'annual_volatility': annual_stats['annual_volatility'],
                'positive_years': annual_stats['positive_years'],
                'negative_years': annual_stats['negative_years'],
                'best_year': annual_stats['best_year'],
                'worst_year': annual_stats['worst_year'],
                
                # Risk metrics
                'max_drawdown': max_drawdown,
                'avg_drawdown_duration': avg_drawdown_duration,
                'max_drawdown_duration': max_drawdown_duration,
                'volatility': annualized_volatility,
                'sharpe_ratio': sharpe_ratio,
                'calmar_ratio': calmar_ratio,
                'profit_factor': profit_factor,
                
                # Trade timing
                'avg_days_held': trades['days_held'].mean() if len(trades) > 0 else 0
            }
            
        return stats
    
    def combine_strategies(self, strategy_names, weights=None):
        """
        Combine multiple strategies into a single portfolio allocation.
        
        STRATEGY COMBINATION RATIONALE:
        ===============================
        
        FTLT1 + FTLT3 COMBINATION EXPLAINED:
        
        FTLT1 (Bull Market Momentum):
        - Operates during: Bull markets (SPY > 200MA)
        - Purpose: Capture sustained upward trends in TQQQ
        - Characteristics: High returns, moderate frequency, trend-following
        - Risk: Large drawdowns during market corrections
        
        FTLT3 (Bear Market Mean Reversion):
        - Operates during: Bear markets (SPY < 200MA) 
        - Purpose: Capture bounces and mean reversion during downtrends
        - Characteristics: Lower returns, higher frequency, counter-trend
        - Risk: Catching falling knives, but smaller individual losses
        
        WHY 50/50 ALLOCATION WORKS:
        - Temporal Diversification: FTLT1 trades bull markets, FTLT3 trades bear markets
        - Risk Reduction: When one strategy faces adverse conditions, the other is inactive
        - Smoother Returns: Reduces portfolio volatility while maintaining upside capture
        - Market Cycle Coverage: Portfolio active across different market environments
        
        PORTFOLIO CONSTRUCTION METHODOLOGY:
        - Each strategy gets 50% allocation (weights=[0.5, 0.5])
        - Returns are weighted: Portfolio Return = 0.5 * FTLT1 Return + 0.5 * FTLT3 Return
        - Positions are weighted: Portfolio Position = 0.5 * FTLT1 Position + 0.5 * FTLT3 Position
        - Risk metrics calculated on combined daily returns
        
        EXPECTED DIVERSIFICATION BENEFITS:
        - Lower maximum drawdown than individual strategies
        - Higher risk-adjusted returns (Sharpe ratio)
        - More consistent performance across market cycles
        - Reduced correlation to pure buy-and-hold approaches
        
        Parameters:
            strategy_names (list): Names of strategies to combine (e.g., ['FTLT1', 'FTLT3'])
            weights (list, optional): Allocation weights. Defaults to equal weight.
            
        Returns:
            dict: Combined portfolio data including:
                - 'data': Daily portfolio positions and returns
                - 'trades': All trades from constituent strategies
                - 'stats': Portfolio-level performance statistics
        """
        if weights is None:
            weights = [1.0 / len(strategy_names)] * len(strategy_names)
        
        print(f"Combining strategies: {strategy_names} with weights: {weights}")
        
        # Validate strategy existence and extract data
        datas = []
        for name in strategy_names:
            if name in self.results and self.results[name] is not None:
                datas.append(self.results[name]['data'])
            else:
                print(f"Strategy {name} not found or has no data.")
                return None

        # Align all strategy data to common trading dates
        # This ensures portfolio calculations use same time periods
        common_index = datas[0].index
        for df in datas[1:]:
            common_index = common_index.intersection(df.index)
        datas = [df.loc[common_index] for df in datas]

        # Calculate weighted portfolio positions and returns
        # Position = sum of (weight * individual strategy position)
        # Return = sum of (weight * individual strategy return)
        combined_position = sum(w * df['position'] for w, df in zip(weights, datas))
        combined_returns = sum(w * df['returns'] for w, df in zip(weights, datas))
        combined_cumulative_returns = (1 + combined_returns).cumprod()

        # Create comprehensive portfolio DataFrame
        combined_data = pd.DataFrame({
            'Close': datas[0]['Close'],  # Use first strategy's price data for reference
            'combined_position': combined_position,
            'combined_returns': combined_returns,
            'combined_cumulative_returns': combined_cumulative_returns
        }, index=common_index)

        # Aggregate all trades from constituent strategies
        # This provides complete trade history for analysis
        combined_trades = pd.concat([self.results[name]['trades'] for name in strategy_names if self.results[name] is not None], ignore_index=True)
        combined_trades['weighted_return'] = combined_trades['return']  # Individual trade returns

        # Calculate comprehensive portfolio statistics
        # Uses same methodology as individual strategies but on portfolio level
        total_return = combined_cumulative_returns.iloc[-1] - 1
        daily_returns = combined_data['combined_returns']
        total_days = len(combined_data)
        years = total_days / 252
        annualized_return = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
        annualized_volatility = daily_returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else 0
        
        # Portfolio drawdown analysis
        rolling_max = combined_cumulative_returns.expanding().max()
        drawdown = (combined_cumulative_returns - rolling_max) / rolling_max
        max_drawdown = abs(drawdown.min())
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0

        # Portfolio-level trade statistics
        winning_trades = combined_trades[combined_trades['weighted_return'] > 0]
        losing_trades = combined_trades[combined_trades['weighted_return'] <= 0]
        
        stats = {
            'total_trades': len(combined_trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(combined_trades) if len(combined_trades) > 0 else 0,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'avg_trade_return': combined_trades['weighted_return'].mean() if len(combined_trades) > 0 else 0,
            'avg_winning_trade': winning_trades['weighted_return'].mean() if len(winning_trades) > 0 else 0,
            'avg_losing_trade': losing_trades['weighted_return'].mean() if len(losing_trades) > 0 else 0,
            'max_drawdown': max_drawdown,
            'volatility': annualized_volatility,
            'sharpe_ratio': sharpe_ratio,
            'calmar_ratio': calmar_ratio,
            'profit_factor': abs(winning_trades['weighted_return'].sum() / losing_trades['weighted_return'].sum()) if losing_trades['weighted_return'].sum() != 0 else float('inf'),
            'avg_days_held': combined_trades['days_held'].mean() if 'days_held' in combined_trades.columns and len(combined_trades) > 0 else 0,
            'largest_win': combined_trades['weighted_return'].max() if len(combined_trades) > 0 else 0,
            'largest_loss': combined_trades['weighted_return'].min() if len(combined_trades) > 0 else 0
        }

        return {
            'data': combined_data,
            'trades': combined_trades,
            'stats': stats
        }
        
    def calculate_buy_hold_benchmarks(self):
        """Calculate buy and hold statistics for comparison"""
        benchmarks = {}
        
        try:
            print("Calculating buy & hold benchmarks...")
            
            for symbol in ['SPY', 'QQQ', 'TQQQ']:
                if symbol not in self.data:
                    print(f"No data for {symbol}")
                    continue
                    
                print(f"Processing {symbol}...")
                data = self.data[symbol]
                
                # Calculate daily returns
                daily_returns = data['Close'].pct_change().fillna(0)
                
                # Total return
                total_return = (data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1
                
                # Annualized metrics
                total_days = len(data)
                years = total_days / 252
                annualized_return = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
                
                # Volatility
                annualized_volatility = daily_returns.std() * np.sqrt(252)
                
                # Sharpe ratio (assuming 0% risk-free rate)
                sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else 0
                
                # Maximum drawdown
                cumulative_returns = (1 + daily_returns).cumprod()
                rolling_max = cumulative_returns.expanding().max()
                drawdown = (cumulative_returns - rolling_max) / rolling_max
                max_drawdown = abs(drawdown.min())
                
                # Calmar ratio
                calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0
                
                benchmarks[f'{symbol}_BuyHold'] = {
                    'total_trades': 1,  # Always invested
                    'winning_trades': 1 if total_return > 0 else 0,
                    'losing_trades': 0 if total_return > 0 else 1,
                    'win_rate': 1.0 if total_return > 0 else 0.0,
                    'total_return': total_return,
                    'annualized_return': annualized_return,
                    'avg_trade_return': total_return,  # Single trade
                    'avg_winning_trade': total_return if total_return > 0 else 0,
                    'avg_losing_trade': total_return if total_return <= 0 else 0,
                    'max_drawdown': max_drawdown,
                    'volatility': annualized_volatility,
                    'sharpe_ratio': sharpe_ratio,
                    'calmar_ratio': calmar_ratio,
                    'profit_factor': float('inf') if total_return > 0 else 0,
                    'avg_days_held': total_days,
                    'largest_win': total_return if total_return > 0 else 0,
                    'largest_loss': total_return if total_return <= 0 else 0
                }
            
            print(f"Returning {len(benchmarks)} benchmarks: {list(benchmarks.keys())}")
            return benchmarks
            
        except Exception as e:
            print(f"Error in calculate_buy_hold_benchmarks: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def create_charts(self):
        """Create performance charts"""
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('TQQQ Strategies vs Buy & Hold + Combined Strategy', fontsize=16, fontweight='bold')
        
        # 1. Equity curves with benchmarks and combined strategy
        ax1 = axes[0, 0]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        benchmark_colors = ['#8c564b', '#e377c2', '#7f7f7f']
        
        # Plot individual strategies
        for i, (strategy_name, result) in enumerate(self.results.items()):
            if result is not None and len(result['data']) > 0:
                data = result['data']
                equity_curve = data['cumulative_returns']
                ax1.plot(data.index, equity_curve, 
                        label=f"{strategy_name} ({equity_curve.iloc[-1]:.1f}x)", 
                        linewidth=2, color=colors[i % len(colors)])
        
        # Plot combined strategy
        combined_result = self.combine_strategies(['FTLT1', 'FTLT3'], [0.8, 0.2])
        if combined_result is not None:
            combined_data = combined_result['data']
            combined_curve = combined_data['combined_cumulative_returns']
            ax1.plot(combined_data.index, combined_curve,
                    label=f"FTLT1+FTLT3 ({combined_curve.iloc[-1]:.1f}x)",
                    linewidth=3, linestyle='-', color='purple', alpha=0.8)
        
        # Plot benchmarks
        for i, symbol in enumerate(['SPY', 'QQQ', 'TQQQ']):
            if symbol in self.data:
                data = self.data[symbol]
                benchmark_returns = (1 + data['Close'].pct_change().fillna(0)).cumprod()
                ax1.plot(data.index, benchmark_returns, 
                        label=f"{symbol} B&H ({benchmark_returns.iloc[-1]:.1f}x)", 
                        linewidth=2, linestyle='--', alpha=0.7, 
                        color=benchmark_colors[i % len(benchmark_colors)])
        
        ax1.set_title('Cumulative Returns: Strategies vs Benchmarks', fontweight='bold')
        ax1.set_ylabel('Cumulative Return Multiple')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=1, color='black', linestyle='--', alpha=0.5)
        ax1.set_yscale('log')  # Log scale for better visualization
        
        # 2. Risk-Return Scatter Plot (including combined)
        ax2 = axes[0, 1]
        stats = self.calculate_statistics()
        benchmarks = self.calculate_buy_hold_benchmarks()
        
        # Safety check for benchmarks
        if benchmarks is None:
            benchmarks = {}
            
        all_stats = {**stats, **benchmarks}
        
        # Add combined strategy stats
        combined_result = self.combine_strategies(['FTLT1', 'FTLT3'], [0.8, 0.2])
        if combined_result is not None:
            all_stats['FTLT1+FTLT3_Combined'] = combined_result['stats']
        
        strategy_returns = []
        strategy_volatilities = []
        strategy_names = []
        strategy_colors = []
        
        for strategy_name, stat in all_stats.items():
            if stat.get('total_trades', 0) > 0:
                strategy_returns.append(stat['annualized_return'] * 100)
                strategy_volatilities.append(stat['volatility'] * 100)
                strategy_names.append(strategy_name)
                if strategy_name.endswith('_BuyHold'):
                    strategy_colors.append('red')
                elif 'Combined' in strategy_name:
                    strategy_colors.append('purple')
                else:
                    strategy_colors.append('blue')
        
        scatter = ax2.scatter(strategy_volatilities, strategy_returns, 
                            c=strategy_colors, s=100, alpha=0.7)
        
        for i, name in enumerate(strategy_names):
            ax2.annotate(name, (strategy_volatilities[i], strategy_returns[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax2.set_title('Risk-Return Profile', fontweight='bold')
        ax2.set_xlabel('Volatility (%)')
        ax2.set_ylabel('Annualized Return (%)')
        ax2.grid(True, alpha=0.3)
        
        # Add legend
        blue_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Strategies')
        purple_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', markersize=10, label='Combined')
        red_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Benchmarks')
        ax2.legend(handles=[blue_patch, purple_patch, red_patch])
        
        # 3. Sharpe Ratio Comparison (including combined)
        ax3 = axes[1, 0]
        
        sharpe_names = []
        sharpe_values = []
        bar_colors = []
        
        # Use the same all_stats from above
        for strategy_name, stat in all_stats.items():
            if stat.get('total_trades', 0) > 0:
                sharpe_names.append(strategy_name)
                sharpe_values.append(stat['sharpe_ratio'])
                if strategy_name.endswith('_BuyHold'):
                    bar_colors.append('red')
                elif 'Combined' in strategy_name:
                    bar_colors.append('purple')
                else:
                    bar_colors.append('blue')
        
        bars = ax3.bar(range(len(sharpe_names)), sharpe_values, color=bar_colors, alpha=0.7)
        ax3.set_title('Sharpe Ratio Comparison', fontweight='bold')
        ax3.set_ylabel('Sharpe Ratio')
        ax3.set_xticks(range(len(sharpe_names)))
        ax3.set_xticklabels(sharpe_names, rotation=45, ha='right')
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, sharpe_values)):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.2f}', ha='center', va='bottom', fontsize=9)
        
        # 4. Combined Strategy Position Allocation Over Time
        ax4 = axes[1, 1]
        
        if combined_result is not None:
            combined_data = combined_result['data']
            
            # Get individual strategy positions
            ftlt1_pos = self.results['FTLT1']['data'].loc[combined_data.index]['position'] * 0.5
            ftlt3_pos = self.results['FTLT3']['data'].loc[combined_data.index]['position'] * 0.5
            
            # Create stacked area chart
            ax4.fill_between(combined_data.index, 0, ftlt1_pos, alpha=0.7, label='FTLT1 (50%)', color='blue')
            ax4.fill_between(combined_data.index, ftlt1_pos, ftlt1_pos + ftlt3_pos, alpha=0.7, label='FTLT3 (50%)', color='green')
            
            ax4.set_title('Combined Strategy Position Allocation', fontweight='bold')
            ax4.set_ylabel('Portfolio Weight')
            ax4.set_ylim(0, 1)
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            # Fallback: Show drawdown comparison
            dd_names = []
            dd_values = []
            dd_colors = []
            
            # Use the same all_stats from above
            for strategy_name, stat in all_stats.items():
                if stat.get('total_trades', 0) > 0:
                    dd_names.append(strategy_name)
                    dd_values.append(stat['max_drawdown'] * 100)
                    if strategy_name.endswith('_BuyHold'):
                        dd_colors.append('red')
                    elif 'Combined' in strategy_name:
                        dd_colors.append('purple')
                    else:
                        dd_colors.append('blue')
            
            bars = ax4.bar(range(len(dd_names)), dd_values, color=dd_colors, alpha=0.7)
            ax4.set_title('Maximum Drawdown Comparison', fontweight='bold')
            ax4.set_ylabel('Max Drawdown (%)')
            ax4.set_xticks(range(len(dd_names)))
            ax4.set_xticklabels(dd_names, rotation=45, ha='right')
            ax4.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for i, (bar, value) in enumerate(zip(bars, dd_values)):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                        f'{value:.1f}%', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        return fig
    
    def save_to_excel(self, filename='tqqq_backtest_results.xlsx'):
        """Save all results to Excel file"""
        print(f"Saving results to {filename}...")
        
        def fix_timezone(df):
            """Remove timezone info from datetime columns"""
            df_copy = df.copy()
            for col in df_copy.columns:
                if pd.api.types.is_datetime64_any_dtype(df_copy[col]):
                    if hasattr(df_copy[col].dtype, 'tz') and df_copy[col].dtype.tz is not None:
                        df_copy[col] = df_copy[col].dt.tz_localize(None)
                # Also handle datetime index
            if hasattr(df_copy.index, 'tz') and df_copy.index.tz is not None:
                df_copy.index = df_copy.index.tz_localize(None)
            return df_copy
        
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # 1. Executive Summary
            stats = self.calculate_statistics()
            benchmarks = self.calculate_buy_hold_benchmarks()
            
            # Safety check for benchmarks
            if benchmarks is None:
                benchmarks = {}
                
            all_stats = {**stats, **benchmarks}
            
            # Add combined strategy
            combined_result = self.combine_strategies(['FTLT1', 'FTLT3'], [0.8, 0.2])
            if combined_result is not None:
                all_stats['FTLT1+FTLT3_Combined'] = combined_result['stats']
            
            summary_data = []
            
            for strategy_name, stat in all_stats.items():
                if stat.get('total_trades', 0) > 0:
                    strategy_type = "Strategy" if not strategy_name.endswith("_BuyHold") and "Combined" not in strategy_name else ("Benchmark" if strategy_name.endswith("_BuyHold") else "Combined")
                    summary_data.append({
                        'Name': strategy_name,
                        'Type': strategy_type,
                        'Total Trades': stat['total_trades'],
                        'Winning Trades': stat['winning_trades'],
                        'Losing Trades': stat['losing_trades'],
                        'Win Rate (%)': round(stat['win_rate'] * 100, 2),
                        'Total Return (%)': round(stat['total_return'] * 100, 2),
                        'Annualized Return (%)': round(stat['annualized_return'] * 100, 2),
                        'Avg Trade Return (%)': round(stat['avg_trade_return'] * 100, 2),
                        'Avg Winning Trade (%)': round(stat['avg_winning_trade'] * 100, 2),
                        'Avg Losing Trade (%)': round(stat['avg_losing_trade'] * 100, 2),
                        'Max Drawdown (%)': round(stat['max_drawdown'] * 100, 2),
                        'Volatility (%)': round(stat['volatility'] * 100, 2),
                        'Sharpe Ratio': round(stat['sharpe_ratio'], 2),
                        'Calmar Ratio': round(stat['calmar_ratio'], 2),
                        'Profit Factor': round(stat['profit_factor'], 2),
                        'Avg Days Held': round(stat['avg_days_held'], 1),
                        'Largest Win (%)': round(stat['largest_win'] * 100, 2),
                        'Largest Loss (%)': round(stat['largest_loss'] * 100, 2)
                    })
            
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Executive_Summary', index=False)
            
            # 2. Detailed Statistics (including benchmarks)
            all_stats_df = pd.DataFrame(all_stats).T
            all_stats_df.to_excel(writer, sheet_name='Detailed_Statistics')
            
            # 3. All trades combined (including combined strategy)
            all_trades = []
            for strategy_name, result in self.results.items():
                if result is not None and len(result['trades']) > 0:
                    trades_copy = result['trades'].copy()
                    trades_copy['strategy'] = strategy_name
                    # Fix timezone issues in date columns
                    for col in ['entry_date', 'exit_date']:
                        if col in trades_copy.columns:
                            trades_copy[col] = pd.to_datetime(trades_copy[col]).dt.tz_localize(None)
                    all_trades.append(trades_copy)
            
            # Add combined strategy trades
            if combined_result is not None and len(combined_result['trades']) > 0:
                combined_trades_copy = combined_result['trades'].copy()
                combined_trades_copy['strategy'] = 'FTLT1+FTLT3_Combined'
                # Fix timezone issues in date columns
                for col in ['entry_date', 'exit_date']:
                    if col in combined_trades_copy.columns:
                        combined_trades_copy[col] = pd.to_datetime(combined_trades_copy[col]).dt.tz_localize(None)
                all_trades.append(combined_trades_copy)
            
            if all_trades:
                combined_trades = pd.concat(all_trades, ignore_index=True)
                combined_trades['return_pct'] = combined_trades['return'] * 100
                combined_trades.to_excel(writer, sheet_name='All_Trades', index=False)
            
            # 4. Individual strategy results
            for strategy_name, result in self.results.items():
                if result is not None:
                    # Individual trades
                    if len(result['trades']) > 0:
                        trades_detailed = result['trades'].copy()
                        # Fix timezone issues in date columns
                        for col in ['entry_date', 'exit_date']:
                            if col in trades_detailed.columns:
                                trades_detailed[col] = pd.to_datetime(trades_detailed[col]).dt.tz_localize(None)
                        trades_detailed['return_pct'] = trades_detailed['return'] * 100
                        trades_detailed.to_excel(writer, sheet_name=f'{strategy_name}_Trades', index=False)
                    
                    # Daily signals and positions (last 1000 rows to avoid Excel limits)
                    daily_data = result['data'][['Close', 'MA_20', 'MA_200', 'RSI_10', 'bullmkt', 'bearmkt', 
                                               'entry_signal', 'exit_signal', 'position', 'returns', 
                                               'cumulative_returns']].tail(1000)
                    daily_data['returns_pct'] = daily_data['returns'] * 100
                    daily_data = fix_timezone(daily_data)
                    daily_data.to_excel(writer, sheet_name=f'{strategy_name}_Daily')
            
            # Add combined strategy data
            if combined_result is not None:
                # Combined strategy trades
                if len(combined_result['trades']) > 0:
                    combined_trades_detailed = combined_result['trades'].copy()
                    # Fix timezone issues in date columns
                    for col in ['entry_date', 'exit_date']:
                        if col in combined_trades_detailed.columns:
                            combined_trades_detailed[col] = pd.to_datetime(combined_trades_detailed[col]).dt.tz_localize(None)
                    combined_trades_detailed['return_pct'] = combined_trades_detailed['weighted_return'] * 100
                    combined_trades_detailed.to_excel(writer, sheet_name='Combined_Strategy_Trades', index=False)
                
                # Combined strategy daily data
                combined_daily = combined_result['data'][['Close', 'combined_position', 'combined_returns', 
                                                        'combined_cumulative_returns']].tail(1000)
                combined_daily['combined_returns_pct'] = combined_daily['combined_returns'] * 100
                combined_daily = fix_timezone(combined_daily)
                combined_daily.to_excel(writer, sheet_name='Combined_Strategy_Daily')
            
            # 5. Market data
            market_data_summary = {}
            for symbol in ['TQQQ', 'QQQ', 'SPY', 'UVXY']:
                if symbol in self.data:
                    data = self.data[symbol]
                    market_data_summary[symbol] = {
                        'Start Date': data.index[0].strftime('%Y-%m-%d'),
                        'End Date': data.index[-1].strftime('%Y-%m-%d'),
                        'Total Days': len(data),
                        'Start Price': data['Close'].iloc[0],
                        'End Price': data['Close'].iloc[-1],
                        'Total Return (%)': ((data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1) * 100,
                        'Max Price': data['Close'].max(),
                        'Min Price': data['Close'].min(),
                        'Avg Volume': data['Volume'].mean()
                    }
            
            if market_data_summary:
                market_df = pd.DataFrame(market_data_summary).T
                market_df.to_excel(writer, sheet_name='Market_Data_Summary')
            
            # 6. Raw price data (sample)
            if 'TQQQ' in self.data:
                price_data = self.data['TQQQ'][['Open', 'High', 'Low', 'Close', 'Volume', 
                                               'MA_20', 'MA_200', 'RSI_10']].tail(1000)
                price_data = fix_timezone(price_data)
                price_data.to_excel(writer, sheet_name='TQQQ_Prices_Sample')
        
        # Save charts
        fig = self.create_charts()
        chart_filename = filename.replace('.xlsx', '_charts.png')
        fig.savefig(chart_filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        print(f"✓ Results saved to {filename}")
        print(f"✓ Charts saved to {chart_filename}")
    
    def print_summary_report(self):
        """Print a comprehensive summary report to console"""
        print("\n" + "="*80)
        print("TQQQ MULTI-STRATEGY BACKTEST SUMMARY REPORT")
        print("="*80)
        
        stats = self.calculate_statistics()
        benchmarks = self.calculate_buy_hold_benchmarks()
        
        # Safety check for benchmarks
        if benchmarks is None:
            benchmarks = {}
        
        # Combine strategies and benchmarks
        all_stats = {**stats, **benchmarks}
        
        # Add combined strategy (FTLT1 + FTLT3)
        combined_result = self.combine_strategies(['FTLT1', 'FTLT3'], [0.8, 0.2])
        if combined_result is not None:
            combined_name = 'FTLT1+FTLT3_Combined'
            all_stats[combined_name] = combined_result['stats']
            print(f"\n🔄 Combined Strategy Analysis: 80% FTLT1 + 20% FTLT3")
        
        # Overall summary
        total_strategies = len([s for s in stats.values() if s.get('total_trades', 0) > 0])
        print(f"\nBacktest Period: {self.start_date} to {self.end_date}")
        print(f"Initial Capital: ${self.initial_capital:,}")
        print(f"Active Strategies: {total_strategies}")
        
        # Strategy performance ranking
        strategy_performance = []
        for strategy_name, stat in all_stats.items():
            if stat.get('total_trades', 0) > 0:
                strategy_performance.append((strategy_name, stat['annualized_return']))
        
        strategy_performance.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\nPERFORMANCE RANKING (Annualized Return):")
        print("-" * 60)
        for i, (strategy, return_pct) in enumerate(strategy_performance, 1):
            if strategy.endswith("_BuyHold"):
                strategy_type = "📊 BENCHMARK"
            elif "Combined" in strategy:
                strategy_type = "🔄 COMBINED "
            else:
                strategy_type = "📈 STRATEGY "
            print(f"{i:2d}. {strategy_type} {strategy}: {return_pct:.1%}")
        
        # Detailed comparison table
        print(f"\nDETAILED PERFORMANCE COMPARISON:")
        print("-" * 120)
        print(f"{'Strategy':<20} {'Total Ret':<10} {'Ann. Ret':<9} {'Volatility':<10} {'Sharpe':<7} {'Max DD':<8} {'Calmar':<7} {'Trades':<7}")
        print("-" * 120)
        
        for strategy_name, stat in all_stats.items():
            if stat.get('total_trades', 0) > 0:
                print(f"{strategy_name:<20} "
                      f"{stat['total_return']:>8.1%} "
                      f"{stat['annualized_return']:>8.1%} "
                      f"{stat['volatility']:>9.1%} "
                      f"{stat['sharpe_ratio']:>6.2f} "
                      f"{stat['max_drawdown']:>7.1%} "
                      f"{stat['calmar_ratio']:>6.2f} "
                      f"{stat['total_trades']:>6}")
        
        # Focus on combined strategy analysis
        if combined_result is not None:
            print(f"\n🔄 COMBINED STRATEGY ANALYSIS (FTLT1 + FTLT3):")
            print("-" * 60)
            cstats = combined_result['stats']
            print(f"  💰 Portfolio Allocation: 80% FTLT1, 20% FTLT3")
            print(f"  📊 Total Trades: {cstats['total_trades']}")
            print(f"  🎯 Win Rate: {cstats['win_rate']:.1%}")
            print(f"  📈 Total Return: {cstats['total_return']:.1%}")
            print(f"  📈 Annualized Return: {cstats['annualized_return']:.1%}")
            print(f"  ⚠️  Max Drawdown: {cstats['max_drawdown']:.1%}")
            print(f"  📊 Volatility: {cstats['volatility']:.1%}")
            print(f"  📊 Sharpe Ratio: {cstats['sharpe_ratio']:.2f}")
            print(f"  📊 Calmar Ratio: {cstats['calmar_ratio']:.2f}")
            
            # Compare to individual strategies
            ftlt1_stats = stats.get('FTLT1', {})
            ftlt3_stats = stats.get('FTLT3', {})
            
            if ftlt1_stats and ftlt3_stats:
                print(f"\n  📊 DIVERSIFICATION BENEFITS:")
                print(f"     FTLT1 Sharpe: {ftlt1_stats.get('sharpe_ratio', 0):.2f}")
                print(f"     FTLT3 Sharpe: {ftlt3_stats.get('sharpe_ratio', 0):.2f}")
                print(f"     Combined Sharpe: {cstats['sharpe_ratio']:.2f}")
                print(f"     FTLT1 Max DD: {ftlt1_stats.get('max_drawdown', 0):.1%}")
                print(f"     FTLT3 Max DD: {ftlt3_stats.get('max_drawdown', 0):.1%}")
                print(f"     Combined Max DD: {cstats['max_drawdown']:.1%}")
        
        # Detailed strategy metrics (excluding benchmarks and combined)
        print(f"\nDETAILED INDIVIDUAL STRATEGY METRICS:")
        print("-" * 80)
        
        for strategy_name, stat in stats.items():
            if stat.get('total_trades', 0) > 0:
                print(f"\n{strategy_name.upper()}:")
                print(f"  📊 Total Trades: {stat['total_trades']}")
                print(f"  🎯 Win Rate: {stat['win_rate']:.1%}")
                print(f"  📈 Total Return: {stat['total_return']:.1%}")
                print(f"  📈 Annualized Return: {stat['annualized_return']:.1%}")
                print(f"  💰 Avg Trade Return: {stat['avg_trade_return']:.2%}")
                print(f"  🏆 Avg Winning Trade: {stat['avg_winning_trade']:.2%}")
                print(f"  📉 Avg Losing Trade: {stat['avg_losing_trade']:.2%}")
                print(f"  ⚠️  Max Drawdown: {stat['max_drawdown']:.1%}")
                print(f"  📊 Volatility: {stat['volatility']:.1%}")
                print(f"  📊 Sharpe Ratio: {stat['sharpe_ratio']:.2f}")
                print(f"  📊 Calmar Ratio: {stat['calmar_ratio']:.2f}")
                print(f"  💎 Profit Factor: {stat['profit_factor']:.2f}")
                print(f"  ⏱️  Avg Days Held: {stat['avg_days_held']:.1f}")
                print(f"  🚀 Largest Win: {stat['largest_win']:.1%}")
                print(f"  💥 Largest Loss: {stat['largest_loss']:.1%}")
        
        print("\n" + "="*80)
    
    def run_full_backtest(self):
        """Run the complete backtesting process"""
        print("🚀 Starting TQQQ Multi-Strategy Backtest")
        print("=" * 50)
        
        try:
            # Download data
            self.download_data()
            
            if not self.data:
                print("❌ No data downloaded. Please check your internet connection and try again.")
                return None, None
            
            # Calculate indicators
            self.calculate_indicators()
            
            # Run strategies
            self.run_all_strategies()
            
            # Print comprehensive summary
            self.print_summary_report()
            
            # Save results
            self.save_to_excel()
            
            return self.results, self.calculate_statistics()
            
        except Exception as e:
            print(f"❌ Error during backtest: {e}")
            import traceback
            traceback.print_exc()
            return None, None

def main():
    """Main execution function"""
    print("🎯 TQQQ Multi-Strategy Backtesting System")
    print("=" * 40)
    
    # Configuration
    START_DATE = '2010-01-01'
    INITIAL_CAPITAL = 100000
    
    print(f"📅 Start Date: {START_DATE}")
    print(f"💰 Initial Capital: ${INITIAL_CAPITAL:,}")
    print("📊 Symbols: TQQQ, QQQ, SPY, UVXY, SQQQ")
    print("🔄 Strategies: FTLT1, FTLT2, FTLT3, FTLT4")
    print()
    
    # Initialize and run backtest
    backtest = TQQQStrategies(
        start_date=START_DATE, 
        initial_capital=INITIAL_CAPITAL
    )
    
    results, stats = backtest.run_full_backtest()
    
    if results and stats:
        print("\n🎉 BACKTEST COMPLETED SUCCESSFULLY!")
        print("\n📁 Files Generated:")
        print("   📊 tqqq_backtest_results.xlsx - Complete results and statistics")
        print("   📈 tqqq_backtest_results_charts.png - Performance visualization")
        print("\n✨ Happy Trading! ✨")
    else:
        print("\n❌ Backtest failed. Please check the error messages above.")

if __name__ == "__main__":
    main()