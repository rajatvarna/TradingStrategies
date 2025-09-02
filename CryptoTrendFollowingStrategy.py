import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class CryptoTrendStrategy:
    def __init__(self):
        self.lookback_periods = [5, 10, 20, 30, 60, 90, 150, 250, 360]
        self.target_vol = 0.25  # 25% target volatility
        self.max_leverage = 2.0  # 200% max allocation
        self.transaction_cost = 0.001  # 10 basis points
        self.rebalance_threshold = 0.20  # 20% threshold for rebalancing
        self.vol_window = 90  # 90-day volatility calculation
        
    def download_data(self, symbol, start_date=None):
        """Download data from Yahoo Finance"""
        if start_date is None:
            # Get maximum available history
            ticker = yf.Ticker(symbol)
            start_date = "2010-01-01"  # Start from early date
        
        try:
            data = yf.download(symbol, start=start_date, end=datetime.now().strftime('%Y-%m-%d'))
            if data.empty:
                raise ValueError(f"No data found for {symbol}")
            return data
        except Exception as e:
            print(f"Error downloading {symbol}: {e}")
            return None
    
    def calculate_donchian_channels(self, data, lookback):
        """Calculate Donchian Channel indicators"""
        high_col = 'High' if 'High' in data.columns else data.columns[0]
        low_col = 'Low' if 'Low' in data.columns else data.columns[0]
        close_col = 'Close' if 'Close' in data.columns else data.columns[0]
        
        # Handle multi-level columns from yfinance
        if isinstance(data.columns, pd.MultiIndex):
            close_col = data.columns[data.columns.get_level_values(0) == 'Close'][0]
            high_col = data.columns[data.columns.get_level_values(0) == 'High'][0]
            low_col = data.columns[data.columns.get_level_values(0) == 'Low'][0]
        
        donchian_high = data[close_col].rolling(window=lookback).max()
        donchian_low = data[close_col].rolling(window=lookback).min()
        donchian_mid = 0.5 * (donchian_high + donchian_low)
        
        return donchian_high, donchian_low, donchian_mid
    
    def calculate_volatility(self, returns, window=90):
        """Calculate annualized volatility"""
        return returns.rolling(window=window).std() * np.sqrt(252)
    
    def generate_signals(self, data, lookback):
        """Generate trading signals for a specific lookback period"""
        close_col = 'Close' if 'Close' in data.columns else data.columns[0]
        if isinstance(data.columns, pd.MultiIndex):
            close_col = data.columns[data.columns.get_level_values(0) == 'Close'][0]
        
        prices = data[close_col].copy()
        returns = prices.pct_change()
        
        # Calculate Donchian Channels
        donchian_high, donchian_low, donchian_mid = self.calculate_donchian_channels(data, lookback)
        
        # Initialize signals and trailing stops
        signals = pd.Series(0, index=prices.index)
        trailing_stops = pd.Series(np.nan, index=prices.index)
        positions = pd.Series(0, index=prices.index)
        
        # Generate signals
        for i in range(lookback, len(prices)):
            current_price = prices.iloc[i]
            prev_signal = signals.iloc[i-1] if i > 0 else 0
            
            # Entry signal: price hits upper Donchian
            if current_price >= donchian_high.iloc[i]:
                signals.iloc[i] = 1
                trailing_stops.iloc[i] = donchian_mid.iloc[i]
            
            # Exit signal: price falls below trailing stop
            elif prev_signal == 1:
                if pd.isna(trailing_stops.iloc[i-1]):
                    trailing_stops.iloc[i] = donchian_mid.iloc[i]
                else:
                    # Update trailing stop (never allow it to go down)
                    trailing_stops.iloc[i] = max(trailing_stops.iloc[i-1], donchian_mid.iloc[i])
                
                if current_price <= trailing_stops.iloc[i]:
                    signals.iloc[i] = 0  # Exit
                    trailing_stops.iloc[i] = np.nan
                else:
                    signals.iloc[i] = 1  # Stay in position
            else:
                signals.iloc[i] = 0
        
        # Calculate position sizes with volatility targeting
        volatility = self.calculate_volatility(returns, self.vol_window)
        raw_weights = signals * (self.target_vol / volatility).fillna(0)
        weights = np.minimum(raw_weights, self.max_leverage)
        
        return {
            'signals': signals,
            'weights': weights,
            'trailing_stops': trailing_stops,
            'donchian_high': donchian_high,
            'donchian_low': donchian_low,
            'donchian_mid': donchian_mid,
            'volatility': volatility
        }
    
    def create_ensemble_strategy(self, data):
        """Create ensemble strategy combining all lookback periods"""
        results = {}
        
        for lookback in self.lookback_periods:
            results[f'model_{lookback}'] = self.generate_signals(data, lookback)
        
        # Combine signals (equal weight ensemble)
        ensemble_weights = pd.Series(0.0, index=data.index)
        
        for model_name, model_result in results.items():
            ensemble_weights += model_result['weights'] / len(self.lookback_periods)
        
        results['ensemble'] = {'weights': ensemble_weights}
        
        return results
    
    def backtest_strategy(self, data, weights):
        """Backtest the strategy with transaction costs"""
        close_col = 'Close' if 'Close' in data.columns else data.columns[0]
        if isinstance(data.columns, pd.MultiIndex):
            close_col = data.columns[data.columns.get_level_values(0) == 'Close'][0]
        
        prices = data[close_col].copy()
        returns = prices.pct_change().fillna(0)
        
        # Apply rebalancing threshold
        weights_rebalanced = weights.copy()
        prev_weight = 0
        
        for i in range(1, len(weights)):
            weight_diff = abs(weights.iloc[i] - prev_weight)
            if weight_diff < self.rebalance_threshold * abs(prev_weight) and prev_weight != 0:
                weights_rebalanced.iloc[i] = prev_weight
            else:
                prev_weight = weights_rebalanced.iloc[i]
        
        # Calculate strategy returns
        position_changes = weights_rebalanced.diff().abs()
        transaction_costs = position_changes * self.transaction_cost
        
        gross_returns = weights_rebalanced.shift(1) * returns
        net_returns = gross_returns - transaction_costs
        
        # Calculate cumulative returns
        cum_returns = (1 + net_returns).cumprod()
        
        return {
            'returns': net_returns,
            'gross_returns': gross_returns,
            'cum_returns': cum_returns,
            'weights': weights_rebalanced,
            'transaction_costs': transaction_costs,
            'prices': prices
        }
    
    def calculate_performance_metrics(self, returns):
        """Calculate comprehensive performance metrics"""
        if len(returns) == 0 or returns.isna().all():
            return {}
        
        # Remove any NaN values
        returns = returns.dropna()
        
        if len(returns) == 0:
            return {}
        
        # Basic metrics
        total_return = (1 + returns).prod() - 1
        ann_return = (1 + returns).prod() ** (252 / len(returns)) - 1
        ann_vol = returns.std() * np.sqrt(252)
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0
        
        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino = ann_return / downside_vol if downside_vol > 0 else 0
        
        # Drawdown calculations
        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdown = (cum_returns - rolling_max) / rolling_max
        max_dd = drawdown.min()
        
        # MAR ratio
        mar_ratio = ann_return / abs(max_dd) if max_dd != 0 else 0
        
        # Drawdown duration
        dd_duration = 0
        max_dd_duration = 0
        for i, dd in enumerate(drawdown):
            if dd < 0:
                dd_duration += 1
                max_dd_duration = max(max_dd_duration, dd_duration)
            else:
                dd_duration = 0
        
        # Win rate and average win/loss
        win_rate = (returns > 0).mean()
        avg_win = returns[returns > 0].mean() if (returns > 0).any() else 0
        avg_loss = returns[returns < 0].mean() if (returns < 0).any() else 0
        
        # Average exposure
        # This is a placeholder - in a real implementation, you'd track actual position exposure
        avg_exposure = 1.0  # Assuming full exposure on average
        
        return {
            'Total Return': total_return,
            'Annualized Return': ann_return,
            'Annualized Volatility': ann_vol,
            'Sharpe Ratio': sharpe,
            'Sortino Ratio': sortino,
            'Max Drawdown': max_dd,
            'MAR Ratio': mar_ratio,
            'Max DD Duration (Days)': max_dd_duration,
            'Win Rate': win_rate,
            'Avg Win %': avg_win,
            'Avg Loss %': avg_loss,
            'Average Exposure': avg_exposure
        }
    
    def create_trades_dataframe(self, backtest_result, model_results):
        """Create a DataFrame of individual trades"""
        weights = backtest_result['weights']
        prices = backtest_result['prices']
        
        trades = []
        in_position = False
        entry_date = None
        entry_price = None
        entry_weight = None
        
        for date, weight in weights.items():
            if weight > 0 and not in_position:
                # Entry
                in_position = True
                entry_date = date
                entry_price = prices.loc[date]
                entry_weight = weight
                
            elif weight == 0 and in_position:
                # Exit
                exit_date = date
                exit_price = prices.loc[date]
                
                # Calculate trade return
                trade_return = (exit_price / entry_price - 1) * entry_weight
                
                trades.append({
                    'Entry Date': entry_date,
                    'Exit Date': exit_date,
                    'Entry Price': entry_price,
                    'Exit Price': exit_price,
                    'Weight': entry_weight,
                    'Return': trade_return,
                    'Duration': (exit_date - entry_date).days
                })
                
                in_position = False
        
        return pd.DataFrame(trades)
    
    def plot_results(self, backtest_result, model_results, symbol):
        """Create comprehensive plots"""
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Equity Curve
        ax1 = plt.subplot(3, 3, 1)
        cum_returns = backtest_result['cum_returns']
        cum_returns.plot(ax=ax1, label='Strategy', linewidth=2)
        
        # Buy and hold comparison
        prices = backtest_result['prices']
        buy_hold = prices / prices.iloc[0]
        buy_hold.plot(ax=ax1, label='Buy & Hold', alpha=0.7)
        
        ax1.set_title(f'{symbol} - Equity Curve')
        ax1.legend()
        ax1.set_ylabel('Cumulative Return')
        
        # 2. Drawdown
        ax2 = plt.subplot(3, 3, 2)
        returns = backtest_result['returns']
        cum_ret = (1 + returns).cumprod()
        rolling_max = cum_ret.expanding().max()
        drawdown = (cum_ret - rolling_max) / rolling_max
        drawdown.plot(ax=ax2, color='red', alpha=0.7)
        ax2.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
        ax2.set_title('Drawdown')
        ax2.set_ylabel('Drawdown %')
        
        # 3. Position Weight over time
        ax3 = plt.subplot(3, 3, 3)
        backtest_result['weights'].plot(ax=ax3)
        ax3.set_title('Position Weight Over Time')
        ax3.set_ylabel('Weight')
        
        # 4. Returns Distribution
        ax4 = plt.subplot(3, 3, 4)
        returns.hist(bins=50, ax=ax4, alpha=0.7)
        ax4.set_title('Returns Distribution')
        ax4.set_xlabel('Daily Returns')
        
        # 5. Rolling Sharpe Ratio
        ax5 = plt.subplot(3, 3, 5)
        rolling_sharpe = returns.rolling(252).mean() / returns.rolling(252).std() * np.sqrt(252)
        rolling_sharpe.plot(ax=ax5)
        ax5.set_title('Rolling 1-Year Sharpe Ratio')
        ax5.axhline(y=1, color='r', linestyle='--', alpha=0.5)
        
        # 6. Monthly Returns Heatmap
        ax6 = plt.subplot(3, 3, 6)
        monthly_returns = returns.groupby([returns.index.year, returns.index.month]).sum()
        monthly_returns.index = pd.MultiIndex.from_tuples(monthly_returns.index)
        monthly_pivot = monthly_returns.unstack(level=1)
        monthly_pivot.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        sns.heatmap(monthly_pivot, annot=True, fmt='.1%', cmap='RdYlGn', 
                   center=0, ax=ax6, cbar_kws={'format': '%.0%%'})
        ax6.set_title('Monthly Returns Heatmap')
        
        # 7. Transaction Costs
        ax7 = plt.subplot(3, 3, 7)
        backtest_result['transaction_costs'].cumsum().plot(ax=ax7, color='orange')
        ax7.set_title('Cumulative Transaction Costs')
        ax7.set_ylabel('Cumulative Cost')
        
        # 8. Yearly Returns Bar Chart
        ax8 = plt.subplot(3, 3, 8)
        yearly_returns = returns.groupby(returns.index.year).sum()
        yearly_returns.plot(kind='bar', ax=ax8, color=['green' if x > 0 else 'red' for x in yearly_returns])
        ax8.set_title('Annual Returns')
        ax8.set_ylabel('Annual Return')
        ax8.tick_params(axis='x', rotation=45)
        
        # 9. Signal visualization (latest period)
        ax9 = plt.subplot(3, 3, 9)
        # Show last 200 days
        recent_data = prices.tail(200)
        recent_data.plot(ax=ax9, color='black', alpha=0.7)
        
        # Plot Donchian channels for 20-day model as example
        if 'model_20' in model_results:
            model_20 = model_results['model_20']
            recent_high = model_20['donchian_high'].tail(200)
            recent_low = model_20['donchian_low'].tail(200)
            recent_mid = model_20['donchian_mid'].tail(200)
            
            ax9.plot(recent_high.index, recent_high, 'g--', alpha=0.7, label='Upper Channel')
            ax9.plot(recent_low.index, recent_low, 'r--', alpha=0.7, label='Lower Channel')
            ax9.plot(recent_mid.index, recent_mid, 'b--', alpha=0.7, label='Mid Channel')
        
        ax9.set_title('Recent Price Action with Signals')
        ax9.legend()
        
        plt.tight_layout()
        return fig
    
    def run_full_backtest(self, symbol):
        """Run the complete backtest"""
        print(f"Running backtest for {symbol}...")
        
        # Download data
        data = self.download_data(symbol)
        if data is None:
            return None
        
        print(f"Data downloaded: {len(data)} rows from {data.index[0]} to {data.index[-1]}")
        
        # Generate ensemble strategy
        model_results = self.create_ensemble_strategy(data)
        
        # Backtest ensemble
        ensemble_weights = model_results['ensemble']['weights']
        backtest_result = self.backtest_strategy(data, ensemble_weights)
        
        # Calculate performance metrics
        performance_metrics = self.calculate_performance_metrics(backtest_result['returns'])
        
        # Create trades dataframe
        trades_df = self.create_trades_dataframe(backtest_result, model_results)
        
        # Get latest signals
        latest_signals = {}
        for lookback in self.lookback_periods:
            model_key = f'model_{lookback}'
            if model_key in model_results:
                latest_signal = model_results[model_key]['signals'].iloc[-1]
                latest_weight = model_results[model_key]['weights'].iloc[-1]
                latest_signals[f'{lookback}d'] = {
                    'signal': latest_signal, 
                    'weight': f"{latest_weight:.2%}"
                }
        
        latest_ensemble_weight = ensemble_weights.iloc[-1]
        latest_signals['ensemble'] = {'weight': f"{latest_ensemble_weight:.2%}"}
        
        return {
            'data': data,
            'model_results': model_results,
            'backtest_result': backtest_result,
            'performance_metrics': performance_metrics,
            'trades': trades_df,
            'latest_signals': latest_signals
        }

def main():
    """Main execution function"""
    strategy = CryptoTrendStrategy()
    
    # Symbols to test
    symbols = ['BTC-USD', 'SPY', 'QQQ', 'TQQQ']
    
    results = {}
    
    for symbol in symbols:
        print(f"\n{'='*50}")
        print(f"Processing {symbol}")
        print('='*50)
        
        result = strategy.run_full_backtest(symbol)
        if result is not None:
            results[symbol] = result
            
            # Display performance metrics
            print(f"\nPerformance Metrics for {symbol}:")
            print("-" * 40)
            for metric, value in result['performance_metrics'].items():
                if isinstance(value, float):
                    if 'Return' in metric or 'Win' in metric or 'Loss' in metric:
                        print(f"{metric}: {value:.2%}")
                    else:
                        print(f"{metric}: {value:.3f}")
                else:
                    print(f"{metric}: {value}")
            
            # Display latest signals
            print(f"\nLatest Signals for {symbol}:")
            print("-" * 30)
            for period, signal_info in result['latest_signals'].items():
                if 'signal' in signal_info:
                    status = "LONG" if signal_info['signal'] > 0 else "FLAT"
                    print(f"{period}: {status} (Weight: {signal_info['weight']})")
                else:
                    print(f"{period}: Weight: {signal_info['weight']}")
            
            # Display recent trades
            if not result['trades'].empty:
                print(f"\nRecent Trades for {symbol} (Last 5):")
                print("-" * 50)
                recent_trades = result['trades'].tail(5)
                for _, trade in recent_trades.iterrows():
                    print(f"Entry: {trade['Entry Date'].date()} @ ${trade['Entry Price']:.2f}")
                    print(f"Exit: {trade['Exit Date'].date()} @ ${trade['Exit Price']:.2f}")
                    print(f"Return: {trade['Return']:.2%}, Duration: {trade['Duration']} days")
                    print("-" * 30)
            
            # Create and save plots
            fig = strategy.plot_results(result['backtest_result'], result['model_results'], symbol)
            plt.savefig(f'{symbol.replace("-", "_")}_backtest_results.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            # Save trades to CSV
            if not result['trades'].empty:
                result['trades'].to_csv(f'{symbol.replace("-", "_")}_trades.csv', index=False)
                print(f"Trades saved to {symbol.replace('-', '_')}_trades.csv")
    
    # Create comparison table
    if results:
        print(f"\n{'='*80}")
        print("STRATEGY COMPARISON")
        print('='*80)
        
        comparison_metrics = ['Annualized Return', 'Annualized Volatility', 'Sharpe Ratio', 
                            'Max Drawdown', 'MAR Ratio', 'Win Rate']
        
        comparison_df = pd.DataFrame()
        for symbol, result in results.items():
            metrics = result['performance_metrics']
            comparison_df[symbol] = [metrics.get(metric, np.nan) for metric in comparison_metrics]
        
        comparison_df.index = comparison_metrics
        
        print(comparison_df.round(3))
        
        # Save comparison
        comparison_df.to_csv('strategy_comparison.csv')
        print("\nComparison saved to strategy_comparison.csv")

if __name__ == "__main__":
    main()