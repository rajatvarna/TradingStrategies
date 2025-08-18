import pandas as pd
import numpy as np
from .strategy_base import Strategy
from .data_manager import DataManager

class OpeningRangeBreakoutStrategy(Strategy):
    """
    A strategy that implements a daily version of the Opening Range Breakout.
    It enters on a breakout of the previous day's high and exits the next day.
    It also uses an ATR-based stop-loss.
    """
    def __init__(self, tickers, start_date, end_date, params, data=None):
        super().__init__(tickers, start_date, end_date, params, data)
        self.strategy_type = 'signal' # This strategy generates signals, not orders

    def _calculate_atr(self, df, period):
        """Calculates the Average True Range (ATR)."""
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.ewm(span=period, adjust=False).mean()
        return atr

    def generate_signals(self, data=None):
        """
        Generates trading signals for the daily ORB strategy.
        """
        if data is None:
            dm = DataManager()
            data = dm.download_data(self.tickers, self.start_date, self.end_date)

        signals = {}
        for t in self.tickers:
            df = data.loc[:, pd.IndexSlice[:, t]].copy()
            df.columns = df.columns.droplevel(1)

            if df.empty or 'Close' not in df.columns:
                continue

            df.dropna(inplace=True)
            if df.empty:
                continue

            # --- Indicator and Filter Calculation ---
            prev_high = df['High'].shift(1)
            atr = self._calculate_atr(df, self.params.get('atr_period', 14))

            # --- Signal Generation ---
            signal_df = pd.DataFrame(index=df.index)
            signal_df['Signal'] = 0

            # Entry Signal: Close breaks above previous day's high
            entry_conditions = (df['Close'] > prev_high)
            signal_df.loc[entry_conditions, 'Signal'] = 1

            # Exit Signal: Exit on the next bar (1-day hold)
            # We can handle this in the backtester, or by creating a -1 signal
            # For simplicity, we'll let the backtester handle the exit after 1 day.
            # A more robust implementation could generate explicit -1 signals.

            # Add Stop Loss info to the signal DataFrame
            # The backtester will need to be modified to use this
            signal_df['stop_loss'] = prev_high - (self.params.get('atr_multiplier', 1) * atr)

            # Add other necessary columns
            signal_df['Close'] = df['Close']
            # This strategy doesn't have a sophisticated ranking, so we can use a constant score
            signal_df['PositionScore'] = 1.0

            signals[t] = signal_df.dropna()

        return signals
