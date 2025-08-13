import yfinance as yf
import pandas as pd
import numpy as np

class CryptoMomentumStrategy:
    """
    Implements a crypto momentum breakout strategy.

    The strategy enters a position when the price breaks above a recent
    highest high, subject to several filters (liquidity, trend, market regime).
    The exit signal is based on a short-term moving average crossover.
    """

    def __init__(self, config):
        """
        Initializes the strategy with a configuration dictionary.

        Args:
            config (dict): A dictionary containing all strategy parameters,
                           typically loaded from a YAML or JSON file.
                           Expected keys: 'tickers', 'start_date', 'end_date',
                           'parameters'.
        """
        self.tickers = config['tickers']
        self.start_date = config['start_date']
        self.end_date = pd.to_datetime('today').strftime('%Y-%m-%d') # Always use current date as end
        self.params = config['parameters']
        self.data = None
        self.signals = {}

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
        df = self.data[ticker].copy().dropna()
        if df.empty:
            return pd.DataFrame()

        # Calculate technical indicators and filters
        df['HH'] = df['Close'].rolling(self.params['hh_period']).max().shift(1)
        df['EMA'] = df['Close'].ewm(span=self.params['ema_period'], adjust=False).mean()
        df['EMA_p'] = df['EMA'].shift(1)

        # Bitcoin risk-on filter
        btc_data = self.data['BTC-USD'].copy().dropna()
        if not btc_data.empty:
            btc_ema = btc_data['Close'].ewm(span=self.params['risk_on_btc_ema_period'], adjust=False).mean()
            df['RiskOn'] = btc_ema.reindex(df.index, method='ffill') < btc_data['Close'].reindex(df.index, method='ffill')
            df['RiskOn'] = df['RiskOn'].fillna(False)
        else:
            df['RiskOn'] = True # Default to risk-on if BTC data is unavailable

        # Generate trading signal: 1=entry, -1=exit, 0=flat
        df['Signal'] = 0
        entry_conditions = (
            (df['Close'] > df['HH']) &
            (df['Close'] > self.params['min_price']) &
            (df['Volume'] > self.params['min_volume']) &
            (df['EMA'] > df['EMA_p']) &
            (df['RiskOn'])
        )
        exit_conditions = (df['EMA'] < df['EMA_p'])

        df.loc[entry_conditions, 'Signal'] = 1
        df.loc[exit_conditions, 'Signal'] = -1

        return df

    def run_signal_generation(self):
        """
        Orchestrates the data download and signal generation for all tickers.

        Returns:
            dict: A dictionary where keys are tickers and values are DataFrames with signals.
        """
        self._download_data()

        # Ensure BTC is processed first for the risk filter
        if 'BTC-USD' not in self.tickers:
            raise ValueError("BTC-USD must be in the ticker list for the risk filter.")

        for ticker in self.tickers:
            if ticker in self.data:
                print(f"Calculating signals for {ticker}...")
                self.signals[ticker] = self._calculate_signals(ticker)
            else:
                print(f"No data for {ticker}, skipping.")

        print("Signal generation complete.")
        return self.signals

if __name__ == '__main__':
    # Example usage:
    tickers = [
        "BTC-USD", "ETH-USD", "XRP-USD", "BNB-USD", "SOL-USD", "ADA-USD",
        "DOGE-USD", "LTC-USD", "LINK-USD", "BCH-USD", "AVAX-USD"
    ]
    strategy = CryptoMomentumStrategy(
        tickers=tickers,
        start_date="2020-01-01",
        end_date=pd.to_datetime('today').strftime('%Y-%m-%d')
    )
    signals = strategy.run_signal_generation()

    # Display the latest signal for a specific ticker
    if "ETH-USD" in signals and not signals["ETH-USD"].empty:
        print("\nLatest signals for ETH-USD:")
        print(signals["ETH-USD"].tail())
