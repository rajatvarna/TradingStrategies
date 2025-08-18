import pandas as pd
import numpy as np
from .strategy_base import Strategy
from .data_manager import DataManager

class CryptoMomentumStrategy(Strategy):
    """
    A strategy that implements a crypto momentum strategy based on breakouts,
    EMAs, and a Bitcoin risk-on filter. This version is optimized for performance
    by performing vectorized operations on a consolidated DataFrame.
    """
    def __init__(self, tickers, start_date, end_date, params, data=None):
        """
        Initializes the strategy with a configuration dictionary.
        Args:
            tickers (list): A list of ticker symbols.
            start_date (str): The start date for the backtest.
            end_date (str): The end date for the backtest.
            params (dict): A dictionary containing all strategy parameters.
            data (pd.DataFrame, optional): Pre-loaded market data.
        """
        super().__init__(tickers, start_date, end_date, params, data)
        self.strategy_type = 'signal'

    def generate_signals(self, data=None):
        """
        Generates trading signals for the crypto momentum strategy.
        This method expects a multi-index DataFrame with ('Feature', 'Ticker') columns.
        """
        print("Generating crypto momentum signals...")
        if data is None:
            dm = DataManager()
            data = dm.download_data(self.tickers, self.start_date, self.end_date)

        if 'BTC-USD' not in [t[1] for t in data.columns if isinstance(t, tuple)]:
            raise ValueError("BTC-USD data is required for the RiskOn filter but not found in data.")

        # --- Vectorized Indicator Calculation ---
        close_prices = data['Close']
        volume = data['Volume']

        # 1. Highest High (HH) for all tickers
        hh = close_prices.rolling(window=self.params['hh_period'], min_periods=1).max().shift(1)

        # 2. Exponential Moving Average (EMA) for all tickers
        ema = close_prices.ewm(span=self.params['ema_period'], adjust=False).mean()
        ema_p = ema.shift(1)

        # 3. Bitcoin Risk-On Filter
        btc_close = close_prices['BTC-USD']
        btc_ema = btc_close.ewm(span=self.params['risk_on_btc_ema_period'], adjust=False).mean()
        risk_on = (btc_close > btc_ema).reindex(close_prices.index, method='ffill').fillna(False)

        # 4. Position Score
        position_score = (close_prices * volume).rolling(window=21).mean().fillna(0)

        # --- Signal Generation for each Ticker ---
        signals = {}
        for t in self.tickers:
            if t == 'BTC-USD': continue

            # Create a DataFrame for the current ticker's signals
            signal_df = pd.DataFrame(index=data.index)

            # Entry conditions
            entry_conditions = (
                (close_prices[t] > hh[t]) &
                (close_prices[t] > self.params['min_price']) &
                (volume[t] > self.params['min_volume']) &
                (ema[t] > ema_p[t]) &
                risk_on
            )

            # Exit conditions
            exit_conditions = (ema[t] < ema_p[t])

            signal_df['Signal'] = 0
            signal_df.loc[entry_conditions, 'Signal'] = 1
            signal_df.loc[exit_conditions, 'Signal'] = -1

            # Add other necessary columns for the backtester
            signal_df['Close'] = close_prices[t]
            signal_df['PositionScore'] = position_score[t]

            signals[t] = signal_df.dropna() # Drop rows where signals could not be computed

        print("Signal generation complete.")
        return signals

if __name__ == '__main__':
    # Example usage:
    strategy_params = {
        'hh_period': 20,
        'ema_period': 5,
        'risk_on_btc_ema_period': 20,
        'min_price': 0.50,
        'min_volume': 100000
    }
    tickers = [
        "BTC-USD", "ETH-USD", "XRP-USD", "BNB-USD", "SOL-USD", "ADA-USD",
        "DOGE-USD", "LTC-USD", "LINK-USD", "BCH-USD", "AVAX-USD"
    ]

    strategy = CryptoMomentumStrategy(
        tickers=tickers,
        start_date="2021-01-01",
        end_date="2023-12-31",
        params=strategy_params
    )

    signals = strategy.generate_signals()

    # Display the latest signal for a specific ticker
    if "ETH-USD" in signals and not signals["ETH-USD"].empty:
        print("\nLatest signals for ETH-USD:")
        print(signals["ETH-USD"].tail())
