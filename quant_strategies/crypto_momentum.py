import yfinance as yf
import pandas as pd
import numpy as np
from quant_strategies.strategy_base import Strategy

class CryptoMomentumStrategy(Strategy):
    """
    Implements a crypto momentum breakout strategy.

    This strategy inherits from the base Strategy class and implements the
    `generate_signals` method.

    The strategy enters a position when the price breaks above a recent
    highest high, subject to several filters (liquidity, trend, market regime).
    The exit signal is based on a short-term moving average crossover.
    """

    def __init__(self, config):
        """
        Initializes the strategy with a configuration dictionary.

        Args:
            config (dict): A dictionary containing all strategy parameters.
        """
        super().__init__(config)

    def _download_data(self):
        """Downloads historical data for all tickers in the universe."""
        print("Downloading data...")
        self.data = yf.download(self.tickers, start=self.start_date, end=self.end_date, group_by='ticker')
        print("Data download complete.")

    def _calculate_signals(self, ticker):
        """
        Calculates the entry and exit signals for a single ticker.

        Returns:
            pd.DataFrame: A dataframe with the original data and the calculated 'Signal' column.
        """
        # Ensure the ticker data exists and is not a DataFrame of NaNs
        if ticker not in self.data or self.data[ticker].isnull().all().all():
            return pd.DataFrame()

        df = self.data[ticker].copy().dropna()
        if df.empty:
            return pd.DataFrame()

        # --- Indicator and Filter Calculation ---

        # 1. Highest High (HH): The core of the breakout signal.
        df['HH'] = df['Close'].rolling(self.params['hh_period']).max().shift(1)

        # 2. Exponential Moving Average (EMA): Used to determine the short-term trend.
        df['EMA'] = df['Close'].ewm(span=self.params['ema_period'], adjust=False).mean()
        df['EMA_p'] = df['EMA'].shift(1) # Previous day's EMA for comparison.

        # 3. Bitcoin Risk-On Filter: A market regime filter.
        btc_data = self.data.get('BTC-USD', pd.DataFrame()).copy().dropna()
        if not btc_data.empty:
            btc_ema = btc_data['Close'].ewm(span=self.params['risk_on_btc_ema_period'], adjust=False).mean()
            # Align BTC data with the current ticker's dates
            df['RiskOn'] = btc_ema.reindex(df.index, method='ffill') < btc_data['Close'].reindex(df.index, method='ffill')
            df['RiskOn'] = df['RiskOn'].fillna(False)
        else:
            df['RiskOn'] = True # Default to risk-on if BTC data is unavailable

        # --- Signal Generation ---
        df['Signal'] = 0

        # Entry Signal: All conditions must be met simultaneously.
        entry_conditions = (
            (df['Close'] > df['HH']) &
            (df['Close'] > self.params['min_price']) &
            (df['Volume'] > self.params['min_volume']) &
            (df['EMA'] > df['EMA_p']) &
            (df['RiskOn'])
        )

        # Exit Signal: The short-term trend has turned down.
        exit_conditions = (df['EMA'] < df['EMA_p'])

        df.loc[entry_conditions, 'Signal'] = 1
        df.loc[exit_conditions, 'Signal'] = -1

        return df

    def generate_signals(self):
        """
        Orchestrates the data download and signal generation for all tickers.

        This method implements the abstract method from the Strategy base class.

        Returns:
            dict: A dictionary where keys are tickers and values are DataFrames with signals.
        """
        self._download_data()

        if 'BTC-USD' not in self.tickers:
            raise ValueError("BTC-USD must be in the ticker list for the risk filter.")

        for ticker in self.tickers:
            print(f"Calculating signals for {ticker}...")
            self.signals[ticker] = self._calculate_signals(ticker)

        print("Signal generation complete.")
        return self.signals

if __name__ == '__main__':
    # Example usage:
    strategy_config = {
        'tickers': [
            "BTC-USD", "ETH-USD", "XRP-USD", "BNB-USD", "SOL-USD", "ADA-USD",
            "DOGE-USD", "LTC-USD", "LINK-USD", "BCH-USD", "AVAX-USD"
        ],
        'start_date': "2020-01-01",
        'parameters': {
            'hh_period': 20,
            'ema_period': 10,
            'risk_on_btc_ema_period': 200,
            'min_price': 0.1,
            'min_volume': 100000
        }
    }
    strategy = CryptoMomentumStrategy(config=strategy_config)
    signals = strategy.generate_signals()

    # Display the latest signal for a specific ticker
    if "ETH-USD" in signals and not signals["ETH-USD"].empty:
        print("\nLatest signals for ETH-USD:")
        print(signals["ETH-USD"].tail())
