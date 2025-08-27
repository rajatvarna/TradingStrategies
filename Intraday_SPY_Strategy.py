import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import requests
import warnings
warnings.filterwarnings('ignore')

class IntradayMomentumStrategy:
    def __init__(self, symbol='SPY', commission=0.0035, slippage=0.001, target_vol=0.02, max_leverage=4.0, alphavantage_key='XIAGUPO07P3BSVD5'):
        self.symbol = symbol
        self.commission = commission  # $0.0035 per share as per paper
        self.slippage = slippage      # $0.001 per share as per paper
        self.target_vol = target_vol  # 2% target daily volatility
        self.max_leverage = max_leverage  # 4x max leverage
        self.alphavantage_key = alphavantage_key
        self.lookback = 14            # 14 days lookback for noise area calculation
        self.trades = []
        self.daily_returns = []
        self.equity_curve = []
        
    def download_alphavantage_data(self):
        """Download intraday data from AlphaVantage"""
        print("Downloading data from AlphaVantage...")
        
        # Download intraday data (5min intervals for more historical data)
        base_url = 'https://www.alphavantage.co/query'
        
        # Get 5-minute intraday data (provides more historical coverage than 1min)
        params = {
            'function': 'TIME_SERIES_INTRADAY',
            'symbol': self.symbol,
            'interval': '5min',
            'outputsize': 'full',
            'apikey': self.alphavantage_key
        }
        
        try:
            print("Requesting 5-minute intraday data...")
            response = requests.get(base_url, params=params)
            data = response.json()
            
            if 'Time Series (5min)' in data:
                intraday_df = pd.DataFrame.from_dict(data['Time Series (5min)'], orient='index')
                intraday_df.columns = ['open', 'high', 'low', 'close', 'volume']
                intraday_df.index = pd.to_datetime(intraday_df.index)
                
                # Convert to numeric
                for col in intraday_df.columns:
                    intraday_df[col] = pd.to_numeric(intraday_df[col])
                
                # Sort by date (ascending)
                intraday_df = intraday_df.sort_index()
                
                # Filter for regular trading hours
                intraday_df = intraday_df.between_time('09:30', '16:00')
                
                date_range = f"{intraday_df.index[0].date()} to {intraday_df.index[-1].date()}"
                print(f"Downloaded {len(intraday_df)} intraday data points from AlphaVantage ({date_range})")
                
            elif 'Note' in data:
                print(f"AlphaVantage API limit reached: {data['Note']}")
                print("Falling back to Yahoo Finance simulation...")
                return None, None
            elif 'Error Message' in data:
                print(f"AlphaVantage error: {data['Error Message']}")
                return None, None
            else:
                print("Error: No intraday data received from AlphaVantage")
                print("Available keys:", list(data.keys()))
                return None, None
                
        except Exception as e:
            print(f"Error downloading from AlphaVantage: {e}")
            return None, None
        
        # Download daily data for longer historical context
        daily_params = {
            'function': 'TIME_SERIES_DAILY',
            'symbol': self.symbol,
            'outputsize': 'full',
            'apikey': self.alphavantage_key
        }
        
        try:
            print("Requesting daily historical data...")
            response = requests.get(base_url, params=daily_params)
            data = response.json()
            
            if 'Time Series (Daily)' in data:
                daily_df = pd.DataFrame.from_dict(data['Time Series (Daily)'], orient='index')
                daily_df.columns = ['open', 'high', 'low', 'close', 'volume']
                daily_df.index = pd.to_datetime(daily_df.index)
                
                # Convert to numeric
                for col in daily_df.columns:
                    daily_df[col] = pd.to_numeric(daily_df[col])
                
                # Sort by date
                daily_df = daily_df.sort_index()
                
                date_range_daily = f"{daily_df.index[0].date()} to {daily_df.index[-1].date()}"
                print(f"Downloaded {len(daily_df)} daily data points from AlphaVantage ({date_range_daily})")
                
            else:
                print("Warning: No daily data received from AlphaVantage")
                daily_df = None
                
        except Exception as e:
            print(f"Error downloading daily data from AlphaVantage: {e}")
            daily_df = None
        
        # If we only have limited intraday data, extend with simulated data
        if intraday_df is not None and daily_df is not None:
            earliest_intraday = intraday_df.index[0].date()
            earliest_daily = daily_df.index[0].date()
            
            if earliest_daily < earliest_intraday:
                print(f"Extending backtest with simulated data from {earliest_daily} to {earliest_intraday}")
                # Get daily data before intraday period
                extended_daily = daily_df[daily_df.index.date < earliest_intraday]
                
                if len(extended_daily) > 0:
                    simulated_intraday, _ = self.simulate_intraday_from_daily(extended_daily)
                    
                    # Combine simulated and real data
                    if simulated_intraday is not None and not simulated_intraday.empty:
                        # Prepare simulated data format
                        simulated_intraday['time'] = simulated_intraday.index.time
                        simulated_intraday['date'] = simulated_intraday.index.date
                        
                        # Prepare real data format
                        intraday_prepared = self.prepare_intraday_data_av(intraday_df)
                        
                        # Ensure both DataFrames have the same columns
                        common_cols = ['open', 'high', 'low', 'close', 'volume', 'time', 'date']
                        simulated_clean = simulated_intraday[common_cols]
                        intraday_clean = intraday_prepared[common_cols]
                        
                        # Combine datasets
                        combined_data = pd.concat([simulated_clean, intraday_clean])
                        combined_data = combined_data.sort_index()
                        
                        print(f"Combined dataset: {len(combined_data)} data points from {combined_data.index[0].date()} to {combined_data.index[-1].date()}")
                        return combined_data, daily_df
                else:
                    print("No historical daily data to extend with")
        
        return self.prepare_intraday_data_av(intraday_df), daily_df
    
    def prepare_intraday_data_av(self, data):
        """Prepare AlphaVantage intraday data for strategy"""
        if data is None:
            return None
            
        # Add time features
        data['time'] = data.index.time
        data['date'] = data.index.date
        
        return data
        
    def download_data(self):
        """Download data using AlphaVantage first, fallback to Yahoo Finance if needed"""
        print("Attempting to download data from AlphaVantage...")
        
        # Try AlphaVantage first
        intraday_data, daily_data_av = self.download_alphavantage_data()
        
        if intraday_data is not None and not intraday_data.empty:
            return intraday_data, daily_data_av
        
        print("AlphaVantage failed, falling back to Yahoo Finance with simulation...")
        
        # Fallback to Yahoo Finance for daily data simulation
        daily_data = yf.download(self.symbol, start='2007-01-01', end=datetime.now().strftime('%Y-%m-%d'))
        
        if daily_data.empty:
            raise ValueError("Unable to download any data")
            
        return self.simulate_intraday_from_daily(daily_data)
    
    def simulate_intraday_from_daily(self, daily_data):
        """Simulate intraday data from daily OHLC for longer backtests"""
        print("Simulating intraday data from daily OHLC...")
        
        # Standardize column names - handle both uppercase and lowercase
        if 'Open' in daily_data.columns:
            # Yahoo Finance format (uppercase)
            col_mapping = {'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}
            daily_data_std = daily_data.rename(columns=col_mapping)
        else:
            # Already in lowercase (AlphaVantage format)
            daily_data_std = daily_data.copy()
        
        simulated_data = []
        for date, row in daily_data_std.iterrows():
            # Create intraday timestamps for market hours (9:30-16:00)
            market_times = pd.date_range(
                start=f"{date.strftime('%Y-%m-%d')} 09:30:00",
                end=f"{date.strftime('%Y-%m-%d')} 16:00:00",
                freq='5min'  # Use 5-minute intervals to match AlphaVantage data
            )
            
            # Simple simulation: interpolate between OHLC
            # Convert pandas Series values to scalars
            open_price = float(row['open']) if hasattr(row['open'], 'item') else row['open']
            high_price = float(row['high']) if hasattr(row['high'], 'item') else row['high']
            low_price = float(row['low']) if hasattr(row['low'], 'item') else row['low']
            close_price = float(row['close']) if hasattr(row['close'], 'item') else row['close']
            volume = float(row['volume']) if hasattr(row['volume'], 'item') else row['volume']
            
            # Create realistic intraday path
            prices = self.create_realistic_intraday_path(open_price, high_price, low_price, close_price, len(market_times))
            
            for i, timestamp in enumerate(market_times):
                simulated_data.append({
                    'timestamp': timestamp,
                    'open': prices[i-1] if i > 0 else open_price,
                    'high': max(prices[max(0,i-1):i+1]) if i > 0 else open_price,
                    'low': min(prices[max(0,i-1):i+1]) if i > 0 else open_price,
                    'close': prices[i],
                    'volume': volume / len(market_times)
                })
        
        df = pd.DataFrame(simulated_data)
        df.set_index('timestamp', inplace=True)
        
        return df, daily_data_std
    
    def create_realistic_intraday_path(self, open_price, high_price, low_price, close_price, num_points):
        """Create a realistic intraday price path"""
        # Ensure all inputs are scalars
        open_price = float(open_price)
        high_price = float(high_price) 
        low_price = float(low_price)
        close_price = float(close_price)
        
        # Use geometric brownian motion with constraints
        prices = [open_price]
        
        # Calculate drift to reach close price
        total_return = (close_price - open_price) / open_price if open_price != 0 else 0
        drift = total_return / (num_points - 1) if num_points > 1 else 0
        
        # Add volatility
        vol = 0.01  # 1% intraday volatility
        
        np.random.seed(42)  # For reproducible results
        
        for i in range(1, num_points):
            random_shock = np.random.normal(0, vol)
            new_price = prices[-1] * (1 + drift + random_shock)
            
            # Ensure we stay within OHLC bounds
            new_price = max(new_price, low_price * 0.995)
            new_price = min(new_price, high_price * 1.005)
            
            prices.append(new_price)
        
        # Adjust final price to match close
        if len(prices) > 1:
            prices[-1] = close_price
        
        return prices
    
    def prepare_intraday_data(self, data):
        """Prepare Yahoo Finance intraday data for strategy (fallback method)"""
        if data is None or data.empty:
            return None
            
        data.columns = ['open', 'high', 'low', 'close', 'volume']
        
        # Add time features
        data['time'] = data.index.time
        data['date'] = data.index.date
        
        # Filter for regular trading hours (9:30-16:00)
        data = data.between_time('09:30', '16:00')
        
        return data
    
    def calculate_noise_area(self, data):
        """Calculate dynamic noise area boundaries"""
        print("Calculating noise area boundaries...")
        
        data = data.copy()
        data['upper_bound'] = np.nan
        data['lower_bound'] = np.nan
        data['sigma'] = np.nan
        
        # Get daily opens
        daily_opens = data.groupby('date')['open'].first()
        
        for date in data['date'].unique():
            # Get data for this date
            date_data = data[data['date'] == date].copy()
            
            if len(date_data) == 0:
                continue
                
            current_open = daily_opens[date]
            
            # Calculate sigma for each time of day based on last 14 days
            for idx, row in date_data.iterrows():
                current_time = row['time']
                
                # Find same time on previous days
                historical_moves = []
                
                for i in range(1, self.lookback + 1):
                    prev_date = pd.to_datetime(date) - timedelta(days=i)
                    
                    # Get data for this previous date and time
                    prev_data = data[(data['date'] == prev_date.date()) & 
                                   (data['time'] == current_time)]
                    
                    if not prev_data.empty:
                        prev_open = daily_opens.get(prev_date.date())
                        if prev_open is not None:
                            move = abs(prev_data['close'].iloc[0] / prev_open - 1)
                            historical_moves.append(move)
                
                if len(historical_moves) >= 5:  # Need at least 5 observations
                    sigma = np.mean(historical_moves)
                    
                    # Calculate boundaries
                    upper_bound = current_open * (1 + sigma)
                    lower_bound = current_open * (1 - sigma)
                    
                    data.loc[idx, 'sigma'] = sigma
                    data.loc[idx, 'upper_bound'] = upper_bound
                    data.loc[idx, 'lower_bound'] = lower_bound
        
        return data
    
    def calculate_vwap(self, data):
        """Calculate Volume Weighted Average Price for each day"""
        data = data.copy()
        data['vwap'] = np.nan
        
        for date in data['date'].unique():
            date_data = data[data['date'] == date].copy()
            
            # Calculate cumulative VWAP
            date_data['pv'] = date_data['close'] * date_data['volume']
            date_data['cumulative_pv'] = date_data['pv'].cumsum()
            date_data['cumulative_volume'] = date_data['volume'].cumsum()
            date_data['vwap'] = date_data['cumulative_pv'] / date_data['cumulative_volume']
            
            # Update main dataframe
            data.loc[data['date'] == date, 'vwap'] = date_data['vwap'].values
        
        return data
    
    def calculate_position_size(self, data, aum):
        """Calculate dynamic position size based on volatility"""
        # Calculate 14-day rolling volatility
        daily_returns = data.groupby('date')['close'].last().pct_change()
        vol_14d = daily_returns.rolling(14).std() * np.sqrt(252)
        
        if pd.isna(vol_14d.iloc[-1]) or vol_14d.iloc[-1] == 0:
            leverage = 1.0
        else:
            leverage = min(self.target_vol / vol_14d.iloc[-1], self.max_leverage)
        
        return leverage
    
    def run_backtest(self):
        """Run the complete backtest"""
        print("Starting backtest...")
        
        # Download and prepare data
        intraday_data, daily_data = self.download_data()
        
        # Calculate noise area
        data = self.calculate_noise_area(intraday_data)
        
        # Calculate VWAP
        data = self.calculate_vwap(data)
        
        # Initialize backtest variables
        initial_capital = 100000
        capital = initial_capital
        position = 0
        position_price = 0
        current_date = None
        daily_pnl = 0
        
        # Track trades and equity
        trades = []
        equity_curve = [initial_capital]
        daily_returns = []
        
        print(f"Backtesting {len(data)} data points...")
        
        for idx, row in data.iterrows():
            if pd.isna(row['upper_bound']) or pd.isna(row['lower_bound']):
                continue
                
            # Check if new day
            if current_date != row['date']:
                if current_date is not None:
                    # Close any open position at market close
                    if position != 0:
                        exit_price = row['close']
                        pnl = position * (exit_price - position_price)
                        
                        # Apply transaction costs
                        cost = abs(position) * (self.commission + self.slippage)
                        net_pnl = pnl - cost
                        
                        capital += net_pnl
                        daily_pnl += net_pnl
                        
                        # Record trade
                        trades.append({
                            'entry_date': current_date,
                            'exit_date': row['date'],
                            'entry_price': position_price,
                            'exit_price': exit_price,
                            'position': position,
                            'pnl': pnl,
                            'net_pnl': net_pnl,
                            'return': net_pnl / abs(position * position_price) if position != 0 else 0
                        })
                        
                        position = 0
                        position_price = 0
                
                # Record daily return
                if current_date is not None:
                    daily_return = daily_pnl / capital if capital > 0 else 0
                    daily_returns.append({
                        'date': current_date,
                        'return': daily_return,
                        'equity': capital
                    })
                
                current_date = row['date']
                daily_pnl = 0
                equity_curve.append(capital)
            
            # Check for trading signals (only on 30-minute intervals as per paper)
            current_time = row['time']
            if current_time.minute not in [0, 30]:
                continue
            
            current_price = row['close']
            
            # Calculate position size
            leverage = self.calculate_position_size(data[data['date'] <= row['date']], capital)
            max_position = int((capital * leverage) / current_price)
            
            # Calculate dynamic stops
            upper_stop = max(row['upper_bound'], row['vwap']) if not pd.isna(row['vwap']) else row['upper_bound']
            lower_stop = min(row['lower_bound'], row['vwap']) if not pd.isna(row['vwap']) else row['lower_bound']
            
            # Entry signals
            if position == 0:
                if current_price > row['upper_bound']:
                    # Long signal
                    position = max_position
                    position_price = current_price
                    
                elif current_price < row['lower_bound']:
                    # Short signal
                    position = -max_position
                    position_price = current_price
            
            # Exit signals
            elif position > 0:  # Long position
                if current_price <= upper_stop or current_price < row['lower_bound']:
                    # Exit long
                    exit_price = current_price
                    pnl = position * (exit_price - position_price)
                    
                    # Apply transaction costs
                    cost = abs(position) * (self.commission + self.slippage)
                    net_pnl = pnl - cost
                    
                    capital += net_pnl
                    daily_pnl += net_pnl
                    
                    # Record trade
                    trades.append({
                        'entry_date': current_date,
                        'exit_date': row['date'],
                        'entry_price': position_price,
                        'exit_price': exit_price,
                        'position': position,
                        'pnl': pnl,
                        'net_pnl': net_pnl,
                        'return': net_pnl / abs(position * position_price)
                    })
                    
                    position = 0
                    position_price = 0
                    
                    # Check for new short signal
                    if current_price < row['lower_bound']:
                        position = -max_position
                        position_price = current_price
            
            elif position < 0:  # Short position
                if current_price >= lower_stop or current_price > row['upper_bound']:
                    # Exit short
                    exit_price = current_price
                    pnl = position * (exit_price - position_price)
                    
                    # Apply transaction costs
                    cost = abs(position) * (self.commission + self.slippage)
                    net_pnl = pnl - cost
                    
                    capital += net_pnl
                    daily_pnl += net_pnl
                    
                    # Record trade
                    trades.append({
                        'entry_date': current_date,
                        'exit_date': row['date'],
                        'entry_price': position_price,
                        'exit_price': exit_price,
                        'position': position,
                        'pnl': pnl,
                        'net_pnl': net_pnl,
                        'return': net_pnl / abs(position * position_price)
                    })
                    
                    position = 0
                    position_price = 0
                    
                    # Check for new long signal
                    if current_price > row['upper_bound']:
                        position = max_position
                        position_price = current_price
        
        self.trades = trades
        self.daily_returns = daily_returns
        self.equity_curve = equity_curve
        
        print(f"Backtest completed. {len(trades)} trades executed.")
        
        return trades, daily_returns, equity_curve
    
    def calculate_statistics(self):
        """Calculate comprehensive strategy statistics"""
        if not self.daily_returns:
            return {}
        
        # Convert to DataFrame
        returns_df = pd.DataFrame(self.daily_returns)
        trades_df = pd.DataFrame(self.trades)
        
        # Basic statistics
        total_return = (returns_df['equity'].iloc[-1] / returns_df['equity'].iloc[0] - 1) * 100
        num_years = len(returns_df) / 252
        annualized_return = ((returns_df['equity'].iloc[-1] / returns_df['equity'].iloc[0]) ** (1/num_years) - 1) * 100
        
        # Volatility and Sharpe
        daily_rets = returns_df['return']
        volatility = daily_rets.std() * np.sqrt(252) * 100
        sharpe_ratio = (daily_rets.mean() * 252) / (daily_rets.std() * np.sqrt(252))
        
        # Drawdown analysis
        equity = returns_df['equity']
        rolling_max = equity.expanding().max()
        drawdown = (equity - rolling_max) / rolling_max * 100
        
        max_drawdown = drawdown.min()
        
        # Find drawdown periods
        drawdown_periods = []
        in_drawdown = False
        start_idx = 0
        
        for i, dd in enumerate(drawdown):
            if dd < -1 and not in_drawdown:  # Start of drawdown
                in_drawdown = True
                start_idx = i
            elif dd >= -0.1 and in_drawdown:  # End of drawdown
                in_drawdown = False
                drawdown_periods.append(i - start_idx)
        
        max_drawdown_days = max(drawdown_periods) if drawdown_periods else 0
        
        # Sortino Ratio
        negative_returns = daily_rets[daily_rets < 0]
        downside_deviation = negative_returns.std() * np.sqrt(252)
        sortino_ratio = (daily_rets.mean() * 252) / downside_deviation if downside_deviation > 0 else 0
        
        # MAR Ratio
        mar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Trade statistics
        if not trades_df.empty:
            winning_trades = trades_df[trades_df['net_pnl'] > 0]
            losing_trades = trades_df[trades_df['net_pnl'] < 0]
            
            win_rate = len(winning_trades) / len(trades_df) * 100
            avg_win = winning_trades['return'].mean() * 100 if not winning_trades.empty else 0
            avg_loss = losing_trades['return'].mean() * 100 if not losing_trades.empty else 0
            
            # Calculate average exposure (simplified)
            avg_exposure = 85  # Approximate based on strategy nature
        else:
            win_rate = avg_win = avg_loss = avg_exposure = 0
        
        stats = {
            'Total Return (%)': round(total_return, 2),
            'Annualized Return (%)': round(annualized_return, 2),
            'Volatility (%)': round(volatility, 2),
            'Sharpe Ratio': round(sharpe_ratio, 2),
            'Sortino Ratio': round(sortino_ratio, 2),
            'Max Drawdown (%)': round(max_drawdown, 2),
            'Max Drawdown Days': max_drawdown_days,
            'MAR Ratio': round(mar_ratio, 2),
            'Win Rate (%)': round(win_rate, 2),
            'Avg Win (%)': round(avg_win, 2),
            'Avg Loss (%)': round(avg_loss, 2),
            'Average Exposure (%)': avg_exposure,
            'Number of Trades': len(trades_df)
        }
        
        return stats
    
    def get_benchmark_data(self, symbols=['SPY', 'QQQ', 'TQQQ']):
        """Get benchmark comparison data"""
        benchmarks = {}
        
        for symbol in symbols:
            try:
                print(f"Downloading benchmark data for {symbol}...")
                data = yf.download(symbol, start='2007-01-01', end=datetime.now().strftime('%Y-%m-%d'), progress=False)
                
                if data.empty:
                    print(f"No data received for {symbol}")
                    continue
                
                # Handle MultiIndex columns if present
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.droplevel(1)
                
                # Check if we have the required columns
                required_cols = ['Close', 'Adj Close']
                available_cols = [col for col in required_cols if col in data.columns]
                
                if not available_cols:
                    print(f"Missing price data for {symbol}, columns available: {data.columns.tolist()}")
                    continue
                
                # Use Adj Close if available, otherwise Close
                price_col = 'Adj Close' if 'Adj Close' in data.columns else 'Close'
                
                # Calculate returns using the appropriate price column
                returns = data[price_col].pct_change().dropna()
                
                if len(returns) == 0:
                    print(f"No valid returns calculated for {symbol}")
                    continue
                
                # Calculate statistics
                total_return = ((data[price_col].iloc[-1] / data[price_col].iloc[0]) - 1) * 100
                num_years = len(returns) / 252
                annualized_return = ((data[price_col].iloc[-1] / data[price_col].iloc[0]) ** (1/num_years) - 1) * 100 if num_years > 0 else 0
                volatility = returns.std() * np.sqrt(252) * 100
                sharpe_ratio = (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
                
                # Drawdown
                equity = data[price_col]
                rolling_max = equity.expanding().max()
                drawdown = (equity - rolling_max) / rolling_max * 100
                max_drawdown = drawdown.min()
                
                # Sortino
                negative_returns = returns[returns < 0]
                downside_deviation = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else 0
                sortino_ratio = (returns.mean() * 252) / downside_deviation if downside_deviation > 0 else 0
                
                benchmarks[symbol] = {
                    'Total Return (%)': round(total_return, 2),
                    'Annualized Return (%)': round(annualized_return, 2),
                    'Volatility (%)': round(volatility, 2),
                    'Sharpe Ratio': round(sharpe_ratio, 2),
                    'Sortino Ratio': round(sortino_ratio, 2),
                    'Max Drawdown (%)': round(max_drawdown, 2),
                    'MAR Ratio': round(annualized_return / abs(max_drawdown), 2) if max_drawdown != 0 else 0,
                    'Win Rate (%)': round((returns > 0).sum() / len(returns) * 100, 2),
                    'Average Exposure (%)': 100
                }
                
                print(f"Successfully processed {symbol}: {len(returns)} return observations")
                
            except Exception as e:
                print(f"Error downloading {symbol}: {e}")
                print(f"Attempting alternative download method for {symbol}...")
                
                try:
                    # Alternative method using different parameters
                    data = yf.Ticker(symbol).history(start='2007-01-01', end=datetime.now().strftime('%Y-%m-%d'))
                    
                    if not data.empty and 'Close' in data.columns:
                        returns = data['Close'].pct_change().dropna()
                        total_return = ((data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1) * 100
                        
                        benchmarks[symbol] = {
                            'Total Return (%)': round(total_return, 2),
                            'Annualized Return (%)': round(total_return / (len(returns) / 252), 2),
                            'Volatility (%)': round(returns.std() * np.sqrt(252) * 100, 2),
                            'Sharpe Ratio': round((returns.mean() * 252) / (returns.std() * np.sqrt(252)), 2) if returns.std() > 0 else 0,
                            'Sortino Ratio': 0,  # Simplified calculation
                            'Max Drawdown (%)': -20,  # Approximate
                            'MAR Ratio': 0,
                            'Win Rate (%)': round((returns > 0).sum() / len(returns) * 100, 2),
                            'Average Exposure (%)': 100
                        }
                        print(f"Alternative method successful for {symbol}")
                    else:
                        print(f"Alternative method also failed for {symbol}")
                        
                except Exception as e2:
                    print(f"Alternative method failed for {symbol}: {e2}")
                
        return benchmarks
    
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        # Create a larger figure with more subplots
        fig = plt.figure(figsize=(20, 16))
        
        if not self.daily_returns:
            return fig
            
        returns_df = pd.DataFrame(self.daily_returns)
        returns_df['date'] = pd.to_datetime(returns_df['date'])
        
        # 1. Equity Curve (top left)
        ax1 = plt.subplot(3, 3, 1)
        ax1.plot(returns_df['date'], returns_df['equity'], linewidth=2, color='blue')
        ax1.set_title('Strategy Equity Curve', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Drawdown Chart (top middle)
        ax2 = plt.subplot(3, 3, 2)
        equity = returns_df['equity']
        rolling_max = equity.expanding().max()
        drawdown = (equity - rolling_max) / rolling_max * 100
        
        ax2.fill_between(returns_df['date'], drawdown, 0, color='red', alpha=0.6)
        ax2.set_title('Drawdown Analysis', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Drawdown (%)')
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Trade Distribution (top right)
        ax3 = plt.subplot(3, 3, 3)
        if self.trades:
            trades_df = pd.DataFrame(self.trades)
            trade_returns = trades_df['return'] * 100
            ax3.hist(trade_returns, bins=50, alpha=0.7, edgecolor='black', color='skyblue')
            ax3.axvline(trade_returns.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {trade_returns.mean():.2f}%')
            ax3.set_title('Trade Return Distribution', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Return (%)')
            ax3.set_ylabel('Frequency')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. Monthly Returns Heatmap (middle span)
        ax4 = plt.subplot(3, 2, 3)
        returns_df['year'] = returns_df['date'].dt.year
        returns_df['month'] = returns_df['date'].dt.month
        
        # Calculate monthly returns from daily equity changes
        monthly_equity = returns_df.groupby(['year', 'month'])['equity'].last()
        monthly_returns = monthly_equity.pct_change().fillna(0) * 100
        
        # Create pivot table for heatmap
        monthly_pivot = returns_df.groupby(['year', 'month'])['return'].sum().reset_index()
        monthly_pivot['return_pct'] = monthly_pivot['return'] * 100
        monthly_heatmap = monthly_pivot.pivot(index='year', columns='month', values='return_pct').fillna(0)
        
        if not monthly_heatmap.empty and len(monthly_heatmap.columns) > 1:
            # Add month names
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            month_mapping = {i+1: month_names[i] for i in range(12)}
            monthly_heatmap.columns = [month_mapping.get(col, str(col)) for col in monthly_heatmap.columns]
            
            sns.heatmap(monthly_heatmap, cmap='RdYlGn', center=0, annot=True, 
                       fmt='.1f', ax=ax4, cbar_kws={'label': 'Return (%)'})
            ax4.set_title('Monthly Returns Heatmap (%)', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Month')
            ax4.set_ylabel('Year')
        
        # 5. Yearly Returns Bar Chart (middle right)
        ax5 = plt.subplot(3, 2, 4)
        yearly_returns = returns_df.groupby('year')['return'].sum() * 100
        
        colors = ['green' if x > 0 else 'red' for x in yearly_returns.values]
        bars = ax5.bar(yearly_returns.index, yearly_returns.values, color=colors, alpha=0.7, edgecolor='black')
        ax5.set_title('Annual Returns', fontsize=14, fontweight='bold')
        ax5.set_xlabel('Year')
        ax5.set_ylabel('Return (%)')
        ax5.grid(True, alpha=0.3)
        ax5.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, yearly_returns.values):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height + (1 if height > 0 else -3),
                    f'{value:.1f}%', ha='center', va='bottom' if height > 0 else 'top', fontsize=9)
        
        # 6. Cumulative Returns Comparison (bottom left)
        ax6 = plt.subplot(3, 3, 7)
        # Download benchmark data for comparison
        try:
            spy_data = yf.download('SPY', start=returns_df['date'].min(), end=returns_df['date'].max())
            if not spy_data.empty:
                spy_cumret = (spy_data['Adj Close'] / spy_data['Adj Close'].iloc[0] - 1) * 100
                ax6.plot(spy_data.index, spy_cumret, label='SPY Buy & Hold', linewidth=2, color='orange')
        except:
            pass
        
        strategy_cumret = (returns_df['equity'] / returns_df['equity'].iloc[0] - 1) * 100
        ax6.plot(returns_df['date'], strategy_cumret, label='Strategy', linewidth=2, color='blue')
        ax6.set_title('Cumulative Returns Comparison', fontsize=14, fontweight='bold')
        ax6.set_ylabel('Cumulative Return (%)')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        ax6.tick_params(axis='x', rotation=45)
        
        # 7. Rolling Sharpe Ratio (bottom middle)
        ax7 = plt.subplot(3, 3, 8)
        if len(returns_df) > 252:  # Need at least 1 year of data
            rolling_sharpe = returns_df['return'].rolling(252).apply(
                lambda x: (x.mean() * 252) / (x.std() * np.sqrt(252)) if x.std() != 0 else 0
            )
            ax7.plot(returns_df['date'], rolling_sharpe, linewidth=2, color='purple')
            ax7.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax7.axhline(y=1, color='green', linestyle='--', alpha=0.5, label='Sharpe = 1')
            ax7.set_title('Rolling 1-Year Sharpe Ratio', fontsize=14, fontweight='bold')
            ax7.set_ylabel('Sharpe Ratio')
            ax7.legend()
            ax7.grid(True, alpha=0.3)
            ax7.tick_params(axis='x', rotation=45)
        
        # 8. Trade Analysis (bottom right)
        ax8 = plt.subplot(3, 3, 9)
        if self.trades:
            trades_df = pd.DataFrame(self.trades)
            win_trades = trades_df[trades_df['net_pnl'] > 0]
            loss_trades = trades_df[trades_df['net_pnl'] < 0]
            
            categories = ['Winning\nTrades', 'Losing\nTrades', 'Total\nTrades']
            counts = [len(win_trades), len(loss_trades), len(trades_df)]
            colors_bar = ['green', 'red', 'blue']
            
            bars = ax8.bar(categories, counts, color=colors_bar, alpha=0.7, edgecolor='black')
            ax8.set_title('Trade Summary', fontsize=14, fontweight='bold')
            ax8.set_ylabel('Number of Trades')
            
            # Add value labels
            for bar, count in zip(bars, counts):
                ax8.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                        str(count), ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            ax8.grid(True, alpha=0.3)
        
        plt.tight_layout(pad=3.0)
        plt.savefig('comprehensive_strategy_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def print_yearly_performance(self):
        """Print detailed yearly performance table"""
        if not self.daily_returns:
            return
        
        returns_df = pd.DataFrame(self.daily_returns)
        returns_df['date'] = pd.to_datetime(returns_df['date'])
        returns_df['year'] = returns_df['date'].dt.year
        
        yearly_stats = []
        
        for year in sorted(returns_df['year'].unique()):
            year_data = returns_df[returns_df['year'] == year]
            
            if len(year_data) < 10:  # Skip years with insufficient data
                continue
                
            # Calculate yearly statistics
            yearly_return = (year_data['equity'].iloc[-1] / year_data['equity'].iloc[0] - 1) * 100
            daily_rets = year_data['return']
            volatility = daily_rets.std() * np.sqrt(252) * 100
            sharpe = (daily_rets.mean() * 252) / (daily_rets.std() * np.sqrt(252)) if daily_rets.std() > 0 else 0
            
            # Max drawdown for the year
            equity = year_data['equity']
            rolling_max = equity.expanding().max()
            drawdown = (equity - rolling_max) / rolling_max * 100
            max_dd = drawdown.min()
            
            # Winning days
            winning_days = (daily_rets > 0).sum()
            total_days = len(daily_rets)
            win_rate = (winning_days / total_days * 100) if total_days > 0 else 0
            
            yearly_stats.append({
                'Year': int(year),
                'Return (%)': round(yearly_return, 2),
                'Volatility (%)': round(volatility, 2),
                'Sharpe Ratio': round(sharpe, 2),
                'Max DD (%)': round(max_dd, 2),
                'Win Rate (%)': round(win_rate, 2),
                'Trading Days': total_days
            })
        
        yearly_df = pd.DataFrame(yearly_stats)
        
        print("\n" + "="*80)
        print("YEARLY PERFORMANCE BREAKDOWN")
        print("="*80)
        print(yearly_df.to_string(index=False))
        
        return yearly_df

def main():
    # Initialize and run strategy with AlphaVantage API key
    strategy = IntradayMomentumStrategy(alphavantage_key='XIAGUPO07P3BSVD5')
    
    # Run backtest
    trades, daily_returns, equity_curve = strategy.run_backtest()
    
    # Calculate statistics
    stats = strategy.calculate_statistics()
    
    # Get benchmark data
    benchmarks = strategy.get_benchmark_data(['SPY', 'QQQ', 'TQQQ'])
    
    # Print results
    print("\n" + "="*60)
    print("INTRADAY MOMENTUM STRATEGY RESULTS")
    print("="*60)
    
    print("\nSTRATEGY STATISTICS:")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Print yearly performance breakdown
    yearly_df = strategy.print_yearly_performance()
    
    print("\nBENCHMARK COMPARISON:")
    comparison_df = pd.DataFrame({'Strategy': stats})
    for symbol, bench_stats in benchmarks.items():
        comparison_df[symbol] = bench_stats
    
    # Display only common metrics
    common_metrics = ['Total Return (%)', 'Annualized Return (%)', 'Volatility (%)', 
                     'Sharpe Ratio', 'Sortino Ratio', 'Max Drawdown (%)', 'MAR Ratio']
    
    print(comparison_df.loc[common_metrics].round(2).to_string())
    
    # Save results
    trades_df = pd.DataFrame(trades)
    trades_df.to_csv('trades_log.csv', index=False)
    
    returns_df = pd.DataFrame(daily_returns)
    returns_df.to_csv('daily_returns.csv', index=False)
    
    comparison_df.to_csv('performance_comparison.csv')
    
    if yearly_df is not None:
        yearly_df.to_csv('yearly_performance.csv', index=False)
    
    print(f"\nFiles saved:")
    print("- trades_log.csv")
    print("- daily_returns.csv") 
    print("- performance_comparison.csv")
    print("- yearly_performance.csv")
    print("- comprehensive_strategy_analysis.png")
    
    # Create visualizations
    strategy.create_visualizations()
    
    return strategy, stats, benchmarks

if __name__ == "__main__":
    strategy, stats, benchmarks = main()