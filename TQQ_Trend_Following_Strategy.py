import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class StrategyBacktest:
    def __init__(self, start_date='2010-01-01', initial_capital=100000):
        self.start_date = start_date
        self.initial_capital = initial_capital
        self.commission_per_share = 0.005
        self.slippage = 0.003  # 0.3%
        
    def download_data(self):
        """Download data from Yahoo Finance"""
        print("Downloading data from Yahoo Finance...")
        
        # Download all required tickers
        tickers = ['QQQ', 'TQQQ', 'TLT', 'SPY']
        data = {}
        
        for ticker in tickers:
            print(f"Downloading {ticker}...")
            df = yf.download(ticker, start=self.start_date, progress=False)
            df = df.dropna()
            data[ticker] = df
            
        return data
    
    def calculate_signals(self, qqq_data):
        """Calculate buy/sell signals based on QQQ and 200-day SMA"""
        print("Calculating trading signals...")
        
        df = qqq_data.copy()
        df['SMA_200'] = df['Close'].rolling(window=200, min_periods=200).mean()
        
        # Remove initial period where SMA is not available
        df = df.dropna()
        
        # Calculate thresholds
        df['Buy_Threshold'] = df['SMA_200'] * 1.05
        df['Sell_Threshold'] = df['SMA_200'] * 0.97
        
        # Generate signals (avoid look-ahead bias)
        df['Signal'] = 0
        position = 0  # 0 = cash (TLT), 1 = long (TQQQ)
        
        # Convert to numpy arrays for faster processing and to avoid pandas comparison issues
        closes = df['Close'].values
        buy_thresholds = df['Buy_Threshold'].values
        sell_thresholds = df['Sell_Threshold'].values
        signals = np.zeros(len(df))
        
        for i in range(len(df)):
            if i == 0:
                signals[i] = 0
                continue
                
            current_close = closes[i]
            buy_threshold = buy_thresholds[i]
            sell_threshold = sell_thresholds[i]
            
            if position == 0 and current_close > buy_threshold:
                # Buy signal
                position = 1
                signals[i] = 1
            elif position == 1 and current_close < sell_threshold:
                # Sell signal
                position = 0
                signals[i] = -1
            else:
                # Hold current position
                signals[i] = 0
        
        df['Signal'] = signals
        return df
    
    def execute_trades(self, signals_df, tqqq_data, tlt_data):
        """Execute trades with transaction costs"""
        print("Executing trades...")
        
        # Align all data to common dates
        common_dates = signals_df.index.intersection(tqqq_data.index).intersection(tlt_data.index)
        signals_df = signals_df.loc[common_dates]
        tqqq_data = tqqq_data.loc[common_dates]
        tlt_data = tlt_data.loc[common_dates]
        
        # Count total signals for debugging
        buy_signals = len(signals_df[signals_df['Signal'] == 1])
        sell_signals = len(signals_df[signals_df['Signal'] == -1])
        print(f"Total Buy Signals: {buy_signals}, Total Sell Signals: {sell_signals}")
        
        trades = []
        portfolio_values = []
        positions = []
        exposures = []
        
        cash = self.initial_capital
        shares_tqqq = 0
        shares_tlt = 0
        position = 'TLT'  # Start in TLT
        
        # Initial TLT purchase
        tlt_price = float(tlt_data['Close'].iloc[0])
        shares_tlt = (cash - 100) / (tlt_price * (1 + self.slippage))  # Leave $100 for commissions
        commission = shares_tlt * self.commission_per_share
        cash = cash - (shares_tlt * tlt_price * (1 + self.slippage)) - commission
        
        trades.append({
            'Date': common_dates[0],
            'Action': 'BUY',
            'Symbol': 'TLT',
            'Shares': shares_tlt,
            'Price': tlt_price * (1 + self.slippage),
            'Commission': commission,
            'Cash': cash,
            'Portfolio_Value': shares_tlt * tlt_price + cash,
            'QQQ_Close': float(signals_df.iloc[0]['Close']),
            'QQQ_SMA200': float(signals_df.iloc[0]['SMA_200']),
            'Signal': 0
        })
        
        for i, date in enumerate(common_dates):
            signal = int(signals_df.loc[date, 'Signal'])
            tqqq_price = float(tqqq_data.loc[date, 'Close'])
            tlt_price = float(tlt_data.loc[date, 'Close'])
            qqq_close = float(signals_df.loc[date, 'Close'])
            qqq_sma200 = float(signals_df.loc[date, 'SMA_200'])
            
            # Calculate current portfolio value
            if position == 'TQQQ':
                portfolio_value = shares_tqqq * tqqq_price + cash
                exposure = (shares_tqqq * tqqq_price) / portfolio_value if portfolio_value > 0 else 0
            else:
                portfolio_value = shares_tlt * tlt_price + cash
                exposure = (shares_tlt * tlt_price) / portfolio_value if portfolio_value > 0 else 0
            
            # Execute trades based on signals
            if signal == 1 and position == 'TLT':  # Buy TQQQ, sell TLT
                print(f"BUY signal on {date}: QQQ ${qqq_close:.2f} > ${qqq_sma200*1.05:.2f}")
                
                # Sell all TLT
                sell_proceeds = shares_tlt * tlt_price * (1 - self.slippage)
                commission_sell = shares_tlt * self.commission_per_share
                cash += sell_proceeds - commission_sell
                
                trades.append({
                    'Date': date,
                    'Action': 'SELL',
                    'Symbol': 'TLT',
                    'Shares': shares_tlt,
                    'Price': tlt_price * (1 - self.slippage),
                    'Commission': commission_sell,
                    'Cash': cash,
                    'Portfolio_Value': cash,
                    'QQQ_Close': qqq_close,
                    'QQQ_SMA200': qqq_sma200,
                    'Signal': signal
                })
                
                # Buy TQQQ
                shares_tqqq = (cash - 100) / (tqqq_price * (1 + self.slippage))
                commission_buy = shares_tqqq * self.commission_per_share
                cash = cash - (shares_tqqq * tqqq_price * (1 + self.slippage)) - commission_buy
                
                trades.append({
                    'Date': date,
                    'Action': 'BUY',
                    'Symbol': 'TQQQ',
                    'Shares': shares_tqqq,
                    'Price': tqqq_price * (1 + self.slippage),
                    'Commission': commission_buy,
                    'Cash': cash,
                    'Portfolio_Value': shares_tqqq * tqqq_price + cash,
                    'QQQ_Close': qqq_close,
                    'QQQ_SMA200': qqq_sma200,
                    'Signal': signal
                })
                
                shares_tlt = 0
                position = 'TQQQ'
                
            elif signal == -1 and position == 'TQQQ':  # Sell TQQQ, buy TLT
                print(f"SELL signal on {date}: QQQ ${qqq_close:.2f} < ${qqq_sma200*0.97:.2f}")
                
                # Sell all TQQQ
                sell_proceeds = shares_tqqq * tqqq_price * (1 - self.slippage)
                commission_sell = shares_tqqq * self.commission_per_share
                cash += sell_proceeds - commission_sell
                
                trades.append({
                    'Date': date,
                    'Action': 'SELL',
                    'Symbol': 'TQQQ',
                    'Shares': shares_tqqq,
                    'Price': tqqq_price * (1 - self.slippage),
                    'Commission': commission_sell,
                    'Cash': cash,
                    'Portfolio_Value': cash,
                    'QQQ_Close': qqq_close,
                    'QQQ_SMA200': qqq_sma200,
                    'Signal': signal
                })
                
                # Buy TLT
                shares_tlt = (cash - 100) / (tlt_price * (1 + self.slippage))
                commission_buy = shares_tlt * self.commission_per_share
                cash = cash - (shares_tlt * tlt_price * (1 + self.slippage)) - commission_buy
                
                trades.append({
                    'Date': date,
                    'Action': 'BUY',
                    'Symbol': 'TLT',
                    'Shares': shares_tlt,
                    'Price': tlt_price * (1 + self.slippage),
                    'Commission': commission_buy,
                    'Cash': cash,
                    'Portfolio_Value': shares_tlt * tlt_price + cash,
                    'QQQ_Close': qqq_close,
                    'QQQ_SMA200': qqq_sma200,
                    'Signal': signal
                })
                
                shares_tqqq = 0
                position = 'TLT'
            
            # Update portfolio tracking
            if position == 'TQQQ':
                portfolio_value = shares_tqqq * tqqq_price + cash
                exposure = (shares_tqqq * tqqq_price) / portfolio_value if portfolio_value > 0 else 0
            else:
                portfolio_value = shares_tlt * tlt_price + cash
                exposure = (shares_tlt * tlt_price) / portfolio_value if portfolio_value > 0 else 0
            
            portfolio_values.append(portfolio_value)
            positions.append(position)
            exposures.append(exposure)
        
        print(f"Total trades executed: {len(trades)}")
        
        # Create portfolio dataframe
        portfolio_df = pd.DataFrame({
            'Date': common_dates,
            'Portfolio_Value': portfolio_values,
            'Position': positions,
            'Exposure': exposures
        }).set_index('Date')
        
        return pd.DataFrame(trades), portfolio_df
    
    def calculate_benchmarks(self, data, start_date, end_date):
        """Calculate buy and hold returns for benchmarks"""
        print("Calculating benchmark returns...")
        
        benchmarks = {}
        for ticker in ['SPY', 'QQQ', 'TQQQ', 'TLT']:
            df = data[ticker].loc[start_date:end_date].copy()
            df['Returns'] = df['Close'].pct_change()
            df['Cumulative'] = (1 + df['Returns']).cumprod() * self.initial_capital
            benchmarks[ticker] = df
            
        return benchmarks
    
    def calculate_performance_metrics(self, portfolio_df, trades_df):
        """Calculate comprehensive performance metrics"""
        print("Calculating performance metrics...")
        
        # Calculate returns
        portfolio_df['Returns'] = portfolio_df['Portfolio_Value'].pct_change()
        portfolio_df = portfolio_df.dropna()
        
        # Calculate TQQQ round-trip trade returns
        tqqq_round_trips = []
        tqqq_buy_trade = None
        
        for i, trade in trades_df.iterrows():
            if trade['Action'] == 'BUY' and trade['Symbol'] == 'TQQQ':
                tqqq_buy_trade = trade
            elif trade['Action'] == 'SELL' and trade['Symbol'] == 'TQQQ' and tqqq_buy_trade is not None:
                # Calculate return for this TQQQ round trip
                tqqq_return = (trade['Price'] / tqqq_buy_trade['Price']) - 1
                tqqq_round_trips.append({
                    'return': tqqq_return,
                    'entry_date': tqqq_buy_trade['Date'],
                    'exit_date': trade['Date'],
                    'days_held': (trade['Date'] - tqqq_buy_trade['Date']).days,
                    'entry_price': tqqq_buy_trade['Price'],
                    'exit_price': trade['Price']
                })
                tqqq_buy_trade = None
        
        # Calculate TLT round-trip trade returns
        tlt_round_trips = []
        tlt_buy_trade = None
        
        for i, trade in trades_df.iterrows():
            if trade['Action'] == 'BUY' and trade['Symbol'] == 'TLT':
                tlt_buy_trade = trade
            elif trade['Action'] == 'SELL' and trade['Symbol'] == 'TLT' and tlt_buy_trade is not None:
                # Calculate return for this TLT round trip
                tlt_return = (trade['Price'] / tlt_buy_trade['Price']) - 1
                tlt_round_trips.append({
                    'return': tlt_return,
                    'entry_date': tlt_buy_trade['Date'],
                    'exit_date': trade['Date'],
                    'days_held': (trade['Date'] - tlt_buy_trade['Date']).days,
                    'entry_price': tlt_buy_trade['Price'],
                    'exit_price': trade['Price']
                })
                tlt_buy_trade = None
        
        # Also calculate position changes for additional analysis
        position_changes = []
        last_portfolio_value = self.initial_capital
        current_position = None
        
        for i, trade in trades_df.iterrows():
            # Identify position changes (when we switch from TLT to TQQQ or vice versa)
            if current_position != trade['Symbol']:
                if current_position is not None:  # Skip the first trade
                    # Calculate return from last position change to this one
                    position_return = (trade['Portfolio_Value'] / last_portfolio_value) - 1
                    position_changes.append(position_return)
                
                current_position = trade['Symbol']
                last_portfolio_value = trade['Portfolio_Value']
        
        print(f"Found {len(tqqq_round_trips)} TQQQ round-trip trades")
        print(f"Found {len(tlt_round_trips)} TLT round-trip trades")
        print(f"Found {len(position_changes)} position changes")
        print(f"Total individual trades: {len(trades_df)}")
        
        # Calculate metrics
        total_return = (portfolio_df['Portfolio_Value'].iloc[-1] / self.initial_capital) - 1
        trading_days = len(portfolio_df)
        years = trading_days / 252
        cagr = (portfolio_df['Portfolio_Value'].iloc[-1] / self.initial_capital) ** (1/years) - 1 if years > 0 else 0
        
        # Volatility metrics
        volatility = portfolio_df['Returns'].std() * np.sqrt(252)
        
        # Drawdown calculation
        rolling_max = portfolio_df['Portfolio_Value'].expanding().max()
        drawdown = (portfolio_df['Portfolio_Value'] - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Sharpe ratio (assuming 2% risk-free rate)
        risk_free_rate = 0.02
        excess_returns = portfolio_df['Returns'].mean() * 252 - risk_free_rate
        sharpe_ratio = excess_returns / volatility if volatility > 0 else 0
        
        # Sortino ratio
        negative_returns = portfolio_df['Returns'][portfolio_df['Returns'] < 0]
        downside_volatility = negative_returns.std() * np.sqrt(252)
        sortino_ratio = excess_returns / downside_volatility if len(negative_returns) > 0 and downside_volatility > 0 else 0
        
        # MAR ratio
        mar_ratio = cagr / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # TQQQ trade statistics
        if tqqq_round_trips:
            tqqq_returns = [t['return'] for t in tqqq_round_trips]
            winning_tqqq = [r for r in tqqq_returns if r > 0]
            losing_tqqq = [r for r in tqqq_returns if r <= 0]
            
            tqqq_win_rate = len(winning_tqqq) / len(tqqq_returns)
            tqqq_avg_win = np.mean(winning_tqqq) if winning_tqqq else 0
            tqqq_avg_loss = np.mean(losing_tqqq) if losing_tqqq else 0
            
            # Calculate TQQQ profit factor
            tqqq_total_wins = sum(winning_tqqq) if winning_tqqq else 0
            tqqq_total_losses = abs(sum(losing_tqqq)) if losing_tqqq else 0
            tqqq_profit_factor = tqqq_total_wins / tqqq_total_losses if tqqq_total_losses > 0 else float('inf')
        else:
            tqqq_returns = []
            winning_tqqq = losing_tqqq = []
            tqqq_win_rate = tqqq_avg_win = tqqq_avg_loss = tqqq_profit_factor = 0
        
        # TLT trade statistics
        if tlt_round_trips:
            tlt_returns = [t['return'] for t in tlt_round_trips]
            winning_tlt = [r for r in tlt_returns if r > 0]
            losing_tlt = [r for r in tlt_returns if r <= 0]
            
            tlt_win_rate = len(winning_tlt) / len(tlt_returns)
            tlt_avg_win = np.mean(winning_tlt) if winning_tlt else 0
            tlt_avg_loss = np.mean(losing_tlt) if losing_tlt else 0
            
            # Calculate TLT profit factor
            tlt_total_wins = sum(winning_tlt) if winning_tlt else 0
            tlt_total_losses = abs(sum(losing_tlt)) if losing_tlt else 0
            tlt_profit_factor = tlt_total_wins / tlt_total_losses if tlt_total_losses > 0 else float('inf')
        else:
            tlt_returns = []
            winning_tlt = losing_tlt = []
            tlt_win_rate = tlt_avg_win = tlt_avg_loss = tlt_profit_factor = 0
        
        # Combined trade statistics
        all_round_trips = tqqq_round_trips + tlt_round_trips
        if all_round_trips:
            all_returns = [t['return'] for t in all_round_trips]
            all_winning = [r for r in all_returns if r > 0]
            all_losing = [r for r in all_returns if r <= 0]
            
            overall_win_rate = len(all_winning) / len(all_returns)
            overall_avg_win = np.mean(all_winning) if all_winning else 0
            overall_avg_loss = np.mean(all_losing) if all_losing else 0
            
            overall_total_wins = sum(all_winning) if all_winning else 0
            overall_total_losses = abs(sum(all_losing)) if all_losing else 0
            overall_profit_factor = overall_total_wins / overall_total_losses if overall_total_losses > 0 else float('inf')
        else:
            overall_win_rate = overall_avg_win = overall_avg_loss = overall_profit_factor = 0
            all_winning = all_losing = []
        
        # Position change statistics (for comparison)
        if position_changes:
            winning_positions = [r for r in position_changes if r > 0]
            losing_positions = [r for r in position_changes if r <= 0]
            position_win_rate = len(winning_positions) / len(position_changes)
        else:
            winning_positions = losing_positions = []
            position_win_rate = 0
        
        # Exposure calculation
        tqqq_exposure = len(portfolio_df[portfolio_df['Position'] == 'TQQQ']) / len(portfolio_df)
        tlt_exposure = len(portfolio_df[portfolio_df['Position'] == 'TLT']) / len(portfolio_df)
        
        metrics = {
            'Total Return (%)': total_return * 100,
            'CAGR (%)': cagr * 100,
            'Volatility (%)': volatility * 100,
            'Sharpe Ratio': sharpe_ratio,
            'Sortino Ratio': sortino_ratio,
            'Max Drawdown (%)': max_drawdown * 100,
            'MAR Ratio': mar_ratio,
            
            # Overall strategy metrics
            'Overall Win Rate (%)': overall_win_rate * 100,
            'Overall Avg Win (%)': overall_avg_win * 100,
            'Overall Avg Loss (%)': overall_avg_loss * 100,
            'Overall Profit Factor': overall_profit_factor,
            
            # TQQQ specific metrics
            'TQQQ Win Rate (%)': tqqq_win_rate * 100,
            'TQQQ Avg Win (%)': tqqq_avg_win * 100,
            'TQQQ Avg Loss (%)': tqqq_avg_loss * 100,
            'TQQQ Profit Factor': tqqq_profit_factor,
            'TQQQ Round Trips': len(tqqq_round_trips),
            'TQQQ Winning Trades': len(winning_tqqq),
            'TQQQ Losing Trades': len(losing_tqqq),
            
            # TLT specific metrics
            'TLT Win Rate (%)': tlt_win_rate * 100,
            'TLT Avg Win (%)': tlt_avg_win * 100,
            'TLT Avg Loss (%)': tlt_avg_loss * 100,
            'TLT Profit Factor': tlt_profit_factor,
            'TLT Round Trips': len(tlt_round_trips),
            'TLT Winning Trades': len(winning_tlt),
            'TLT Losing Trades': len(losing_tlt),
            
            # Other metrics
            'Total Round Trips': len(all_round_trips),
            'Total Individual Trades': len(trades_df),
            'Total Position Changes': len(position_changes),
            'Position Change Win Rate (%)': position_win_rate * 100,
            'TQQQ Exposure (%)': tqqq_exposure * 100,
            'TLT Exposure (%)': tlt_exposure * 100,
            'Final Portfolio Value': portfolio_df['Portfolio_Value'].iloc[-1],
            'Years Backtested': years
        }
        
        # Print comprehensive trade summary
        print(f"\n=== TRADE SUMMARY ===")
        print(f"Total Individual Trades    : {len(trades_df)} (every buy/sell action)")
        print(f"Total Position Changes     : {len(position_changes)} (TLT<->TQQQ switches)")
        print(f"Total Round Trips          : {len(all_round_trips)} (TQQQ + TLT cycles)")
        print(f"TQQQ Round Trips          : {len(tqqq_round_trips)} (TQQQ buy->sell cycles)")
        print(f"TLT Round Trips           : {len(tlt_round_trips)} (TLT buy->sell cycles)")
        
        print(f"\n=== OVERALL STRATEGY PERFORMANCE ===")
        print(f"Overall Win Rate          : {overall_win_rate*100:.1f}% ({len(all_winning)} wins, {len(all_losing)} losses)")
        print(f"Overall Avg Win           : {overall_avg_win*100:.2f}%")
        print(f"Overall Avg Loss          : {overall_avg_loss*100:.2f}%")
        print(f"Overall Profit Factor     : {overall_profit_factor:.2f}")
        
        print(f"\n=== TQQQ PERFORMANCE ===")
        if tqqq_round_trips:
            print(f"TQQQ Win Rate             : {tqqq_win_rate*100:.1f}% ({len(winning_tqqq)} wins, {len(losing_tqqq)} losses)")
            print(f"TQQQ Avg Win              : {tqqq_avg_win*100:.2f}%")
            print(f"TQQQ Avg Loss             : {tqqq_avg_loss*100:.2f}%")
            print(f"TQQQ Profit Factor        : {tqqq_profit_factor:.2f}")
            if tqqq_returns:
                print(f"Best TQQQ Trade           : {max(tqqq_returns)*100:.2f}%")
                print(f"Worst TQQQ Trade          : {min(tqqq_returns)*100:.2f}%")
        else:
            print("No TQQQ round trips completed")
        
        print(f"\n=== TLT PERFORMANCE ===")
        if tlt_round_trips:
            print(f"TLT Win Rate              : {tlt_win_rate*100:.1f}% ({len(winning_tlt)} wins, {len(losing_tlt)} losses)")
            print(f"TLT Avg Win               : {tlt_avg_win*100:.2f}%")
            print(f"TLT Avg Loss              : {tlt_avg_loss*100:.2f}%")
            print(f"TLT Profit Factor         : {tlt_profit_factor:.2f}")
            if tlt_returns:
                print(f"Best TLT Trade            : {max(tlt_returns)*100:.2f}%")
                print(f"Worst TLT Trade           : {min(tlt_returns)*100:.2f}%")
        else:
            print("No TLT round trips completed")
        
        print(f"\n=== EXPOSURE ALLOCATION ===")
        print(f"TQQQ Exposure             : {tqqq_exposure*100:.1f}%")
        print(f"TLT Exposure              : {tlt_exposure*100:.1f}%")
        
        return metrics, {'tqqq': tqqq_returns, 'tlt': tlt_returns, 'all': [t['return'] for t in all_round_trips]}
    
    def create_visualizations(self, portfolio_df, benchmarks, save_path=None):
        """Create equity curve and monthly heatmap"""
        print("Creating visualizations...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 15))
        
        # Equity curve
        ax1.plot(portfolio_df.index, portfolio_df['Portfolio_Value'], label='Strategy', linewidth=2, color='red')
        
        for ticker, data in benchmarks.items():
            ax1.plot(data.index, data['Cumulative'], label=f'{ticker} B&H', linewidth=1.5, alpha=0.7)
        
        ax1.set_title('Equity Curves Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Drawdown
        rolling_max = portfolio_df['Portfolio_Value'].expanding().max()
        drawdown = (portfolio_df['Portfolio_Value'] - rolling_max) / rolling_max * 100
        
        ax2.fill_between(portfolio_df.index, drawdown, 0, alpha=0.3, color='red')
        ax2.plot(portfolio_df.index, drawdown, color='red', linewidth=1)
        ax2.set_title('Strategy Drawdown', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Drawdown (%)')
        ax2.grid(True, alpha=0.3)
        
        # Monthly returns heatmap
        portfolio_df['Returns'] = portfolio_df['Portfolio_Value'].pct_change()
        monthly_returns = portfolio_df['Returns'].resample('M').apply(lambda x: (1 + x).prod() - 1) * 100
        
        # Create monthly heatmap data
        monthly_data = []
        for date, ret in monthly_returns.items():
            monthly_data.append({
                'Year': date.year,
                'Month': date.strftime('%b'),
                'Return': ret
            })
        
        monthly_df = pd.DataFrame(monthly_data)
        if not monthly_df.empty:
            pivot_df = monthly_df.pivot(index='Year', columns='Month', values='Return')
            month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            pivot_df = pivot_df.reindex(columns=month_order, fill_value=0)
            
            sns.heatmap(pivot_df, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
                       ax=ax3, cbar_kws={'label': 'Monthly Return (%)'})
            ax3.set_title('Monthly Returns Heatmap (%)', fontsize=14, fontweight='bold')
        
        # Position allocation
        position_counts = portfolio_df['Position'].value_counts()
        colors = ['lightcoral' if pos == 'TQQQ' else 'lightblue' for pos in position_counts.index]
        ax4.pie(position_counts.values, labels=position_counts.index, autopct='%1.1f%%', colors=colors)
        ax4.set_title('Position Allocation', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def save_to_excel(self, metrics, trades_df, portfolio_df, benchmarks, trade_returns, filename='strategy_results.xlsx'):
        """Save all results to Excel"""
        print(f"Saving results to {filename}...")
        
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Performance metrics
            metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
            metrics_df.to_excel(writer, sheet_name='Performance_Metrics', index=False)
            
            # ALL trades (every single buy/sell action)
            trades_df_formatted = trades_df.copy()
            trades_df_formatted['Date'] = trades_df_formatted['Date'].dt.strftime('%Y-%m-%d')
            trades_df_formatted['Price'] = trades_df_formatted['Price'].round(4)
            trades_df_formatted['Commission'] = trades_df_formatted['Commission'].round(2)
            trades_df_formatted['Cash'] = trades_df_formatted['Cash'].round(2)
            trades_df_formatted['Portfolio_Value'] = trades_df_formatted['Portfolio_Value'].round(2)
            if 'QQQ_Close' in trades_df_formatted.columns:
                trades_df_formatted['QQQ_Close'] = trades_df_formatted['QQQ_Close'].round(4)
                trades_df_formatted['QQQ_SMA200'] = trades_df_formatted['QQQ_SMA200'].round(4)
            trades_df_formatted.to_excel(writer, sheet_name='All_Trades', index=False)
            
            # Round-trip trade analysis (TQQQ buy to TQQQ sell cycles)
            round_trip_trades = []
            tqqq_buy_trade = None
            
            for i, trade in trades_df.iterrows():
                if trade['Action'] == 'BUY' and trade['Symbol'] == 'TQQQ':
                    tqqq_buy_trade = trade
                elif trade['Action'] == 'SELL' and trade['Symbol'] == 'TQQQ' and tqqq_buy_trade is not None:
                    # Calculate round-trip metrics
                    days_held = (trade['Date'] - tqqq_buy_trade['Date']).days
                    total_return = (trade['Price'] / tqqq_buy_trade['Price'] - 1) * 100
                    total_cost = tqqq_buy_trade['Commission'] + trade['Commission']
                    
                    round_trip_trades.append({
                        'Entry_Date': tqqq_buy_trade['Date'].strftime('%Y-%m-%d'),
                        'Exit_Date': trade['Date'].strftime('%Y-%m-%d'),
                        'Days_Held': days_held,
                        'Entry_Price': round(tqqq_buy_trade['Price'], 4),
                        'Exit_Price': round(trade['Price'], 4),
                        'Shares': round(tqqq_buy_trade['Shares'], 2),
                        'Total_Return_%': round(total_return, 2),
                        'Total_Commissions': round(total_cost, 2),
                        'Entry_Portfolio_Value': round(tqqq_buy_trade['Portfolio_Value'], 2),
                        'Exit_Portfolio_Value': round(trade['Portfolio_Value'], 2),
                        'QQQ_Entry': round(tqqq_buy_trade.get('QQQ_Close', 0), 4),
                        'QQQ_Exit': round(trade.get('QQQ_Close', 0), 4),
                        'QQQ_SMA200_Entry': round(tqqq_buy_trade.get('QQQ_SMA200', 0), 4),
                        'QQQ_SMA200_Exit': round(trade.get('QQQ_SMA200', 0), 4)
                    })
                    tqqq_buy_trade = None
            
            if round_trip_trades:
                round_trip_df = pd.DataFrame(round_trip_trades)
                round_trip_df.to_excel(writer, sheet_name='TQQQ_Round_Trip_Trades', index=False)
                print(f"Found {len(round_trip_trades)} complete TQQQ round-trip trades")
            
            # TLT Round-trip trade analysis
            tlt_round_trip_trades = []
            tlt_buy_trade = None
            
            for i, trade in trades_df.iterrows():
                if trade['Action'] == 'BUY' and trade['Symbol'] == 'TLT':
                    tlt_buy_trade = trade
                elif trade['Action'] == 'SELL' and trade['Symbol'] == 'TLT' and tlt_buy_trade is not None:
                    # Calculate round-trip metrics
                    days_held = (trade['Date'] - tlt_buy_trade['Date']).days
                    total_return = (trade['Price'] / tlt_buy_trade['Price'] - 1) * 100
                    total_cost = tlt_buy_trade['Commission'] + trade['Commission']
                    
                    tlt_round_trip_trades.append({
                        'Entry_Date': tlt_buy_trade['Date'].strftime('%Y-%m-%d'),
                        'Exit_Date': trade['Date'].strftime('%Y-%m-%d'),
                        'Days_Held': days_held,
                        'Entry_Price': round(tlt_buy_trade['Price'], 4),
                        'Exit_Price': round(trade['Price'], 4),
                        'Shares': round(tlt_buy_trade['Shares'], 2),
                        'Total_Return_%': round(total_return, 2),
                        'Total_Commissions': round(total_cost, 2),
                        'Entry_Portfolio_Value': round(tlt_buy_trade['Portfolio_Value'], 2),
                        'Exit_Portfolio_Value': round(trade['Portfolio_Value'], 2),
                        'QQQ_Entry': round(tlt_buy_trade.get('QQQ_Close', 0), 4),
                        'QQQ_Exit': round(trade.get('QQQ_Close', 0), 4),
                        'QQQ_SMA200_Entry': round(tlt_buy_trade.get('QQQ_SMA200', 0), 4),
                        'QQQ_SMA200_Exit': round(trade.get('QQQ_SMA200', 0), 4)
                    })
                    tlt_buy_trade = None
            
            if tlt_round_trip_trades:
                tlt_round_trip_df = pd.DataFrame(tlt_round_trip_trades)
                tlt_round_trip_df.to_excel(writer, sheet_name='TLT_Round_Trip_Trades', index=False)
                print(f"Found {len(tlt_round_trip_trades)} complete TLT round-trip trades")
            
            # Position change analysis (all position switches)
            position_changes = []
            last_position = None
            last_portfolio_value = self.initial_capital
            
            for i, trade in trades_df.iterrows():
                if last_position != trade['Symbol'] and last_position is not None:
                    position_return = (trade['Portfolio_Value'] / last_portfolio_value - 1) * 100
                    position_changes.append({
                        'Date': trade['Date'].strftime('%Y-%m-%d'),
                        'From_Position': last_position,
                        'To_Position': trade['Symbol'],
                        'Start_Value': round(last_portfolio_value, 2),
                        'End_Value': round(trade['Portfolio_Value'], 2),
                        'Return_%': round(position_return, 2),
                        'QQQ_Price': round(trade.get('QQQ_Close', 0), 4),
                        'QQQ_SMA200': round(trade.get('QQQ_SMA200', 0), 4),
                        'Signal': trade.get('Signal', 0)
                    })
                
                if last_position != trade['Symbol']:
                    last_position = trade['Symbol']
                    last_portfolio_value = trade['Portfolio_Value']
            
            if position_changes:
                position_changes_df = pd.DataFrame(position_changes)
                position_changes_df.to_excel(writer, sheet_name='All_Position_Changes', index=False)
                print(f"Found {len(position_changes)} position changes")
            
            # Portfolio values (daily)
            portfolio_export = portfolio_df.copy()
            portfolio_export.index = portfolio_export.index.strftime('%Y-%m-%d')
            portfolio_export['Portfolio_Value'] = portfolio_export['Portfolio_Value'].round(2)
            portfolio_export['Exposure'] = portfolio_export['Exposure'].round(4)
            portfolio_export.to_excel(writer, sheet_name='Daily_Portfolio_Values')
            
            # Benchmark comparisons
            benchmark_summary = {}
            for ticker, data in benchmarks.items():
                total_return = (data['Cumulative'].iloc[-1] / self.initial_capital - 1) * 100
                volatility = data['Returns'].std() * np.sqrt(252) * 100
                max_dd = ((data['Close'] / data['Close'].cummax()) - 1).min() * 100
                years = len(data) / 252
                cagr = (data['Cumulative'].iloc[-1] / self.initial_capital) ** (1/years) - 1 if years > 0 else 0
                sharpe = (data['Returns'].mean() * 252 - 0.02) / (data['Returns'].std() * np.sqrt(252))
                
                benchmark_summary[ticker] = {
                    'Total Return (%)': round(total_return, 2),
                    'CAGR (%)': round(cagr * 100, 2),
                    'Volatility (%)': round(volatility, 2),
                    'Sharpe Ratio': round(sharpe, 3),
                    'Max Drawdown (%)': round(max_dd, 2),
                    'Final Value': round(data['Cumulative'].iloc[-1], 2)
                }
            
            benchmark_df = pd.DataFrame(benchmark_summary).T
            benchmark_df.to_excel(writer, sheet_name='Benchmark_Comparison')
            
            # Individual trade returns (separate sheets for TQQQ and TLT)
            if trade_returns and isinstance(trade_returns, dict):
                # TQQQ trade returns
                if trade_returns.get('tqqq'):
                    tqqq_returns_df = pd.DataFrame({
                        'TQQQ_Trade_Number': range(1, len(trade_returns['tqqq']) + 1),
                        'Return_Decimal': trade_returns['tqqq'],
                        'Return_Percent': [r * 100 for r in trade_returns['tqqq']]
                    })
                    tqqq_returns_df['Return_Decimal'] = tqqq_returns_df['Return_Decimal'].round(6)
                    tqqq_returns_df['Return_Percent'] = tqqq_returns_df['Return_Percent'].round(4)
                    tqqq_returns_df.to_excel(writer, sheet_name='TQQQ_Trade_Returns', index=False)
                
                # TLT trade returns
                if trade_returns.get('tlt'):
                    tlt_returns_df = pd.DataFrame({
                        'TLT_Trade_Number': range(1, len(trade_returns['tlt']) + 1),
                        'Return_Decimal': trade_returns['tlt'],
                        'Return_Percent': [r * 100 for r in trade_returns['tlt']]
                    })
                    tlt_returns_df['Return_Decimal'] = tlt_returns_df['Return_Decimal'].round(6)
                    tlt_returns_df['Return_Percent'] = tlt_returns_df['Return_Percent'].round(4)
                    tlt_returns_df.to_excel(writer, sheet_name='TLT_Trade_Returns', index=False)
                
                # All trade returns combined
                if trade_returns.get('all'):
                    all_returns_df = pd.DataFrame({
                        'Trade_Number': range(1, len(trade_returns['all']) + 1),
                        'Return_Decimal': trade_returns['all'],
                        'Return_Percent': [r * 100 for r in trade_returns['all']]
                    })
                    all_returns_df['Return_Decimal'] = all_returns_df['Return_Decimal'].round(6)
                    all_returns_df['Return_Percent'] = all_returns_df['Return_Percent'].round(4)
                    all_returns_df.to_excel(writer, sheet_name='All_Trade_Returns', index=False)
            elif trade_returns and isinstance(trade_returns, list):
                # Fallback for old format
                trade_returns_df = pd.DataFrame({
                    'Trade_Number': range(1, len(trade_returns) + 1),
                    'Return_Decimal': trade_returns,
                    'Return_Percent': [r * 100 for r in trade_returns]
                })
                trade_returns_df['Return_Decimal'] = trade_returns_df['Return_Decimal'].round(6)
                trade_returns_df['Return_Percent'] = trade_returns_df['Return_Percent'].round(4)
                trade_returns_df.to_excel(writer, sheet_name='Trade_Returns', index=False)
            
            # Signal history (for debugging) - sample every 10 days to reduce size
            signal_data = []
            for date in portfolio_df.index[::10]:  # Sample every 10 days
                try:
                    qqq_data = benchmarks['QQQ'].loc[date]
                    signal_data.append({
                        'Date': date.strftime('%Y-%m-%d'),
                        'QQQ_Close': round(qqq_data['Close'], 4),
                        'Portfolio_Value': round(portfolio_df.loc[date, 'Portfolio_Value'], 2),
                        'Position': portfolio_df.loc[date, 'Position']
                    })
                except:
                    continue
            
            if signal_data:
                signals_df_export = pd.DataFrame(signal_data)
                signals_df_export.to_excel(writer, sheet_name='Signal_History_Sample', index=False)
        
        print(f"Results saved to {filename}")
        print(f"Total individual trades recorded: {len(trades_df)}")
        if round_trip_trades:
            print(f"Complete TQQQ round-trip trades: {len(round_trip_trades)}")
        if position_changes:
            print(f"Total position changes: {len(position_changes)}")
    
    def run_backtest(self):
        """Run the complete backtest"""
        print("Starting QQQ/TQQQ Strategy Backtest with 200-Day SMA...")
        print("="*50)
        
        # Download data
        data = self.download_data()
        
        # Calculate signals
        signals_df = self.calculate_signals(data['QQQ'])
        
        # Execute trades
        trades_df, portfolio_df = self.execute_trades(signals_df, data['TQQQ'], data['TLT'])
        
        # Calculate benchmarks
        start_date = portfolio_df.index[0]
        end_date = portfolio_df.index[-1]
        benchmarks = self.calculate_benchmarks(data, start_date, end_date)
        
        # Calculate performance metrics
        metrics, trade_returns = self.calculate_performance_metrics(portfolio_df, trades_df)
        
        # Print results
        print("\nStrategy Performance Metrics:")
        print("="*50)
        for metric, value in metrics.items():
            if isinstance(value, float):
                if 'Ratio' in metric or 'Factor' in metric:
                    print(f"{metric:<30}: {value:.3f}")
                else:
                    print(f"{metric:<30}: {value:.2f}")
            else:
                print(f"{metric:<30}: {value:,.0f}")
        
        # Create visualizations
        fig = self.create_visualizations(portfolio_df, benchmarks, 'strategy_charts.png')
        
        # Save to Excel
        self.save_to_excel(metrics, trades_df, portfolio_df, benchmarks, trade_returns)
        
        print("\nBacktest completed successfully!")
        print("Files generated:")
        print("- strategy_results.xlsx (All data and metrics)")
        print("- strategy_charts.png (Visualizations)")
        
        return metrics, trades_df, portfolio_df, benchmarks, fig

# Run the backtest
if __name__ == "__main__":
    backtest = StrategyBacktest()
    results = backtest.run_backtest()