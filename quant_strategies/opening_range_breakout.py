import pandas as pd
import numpy as np

class OpeningRangeBreakoutStrategy:
    """
    Implements a portfolio-level Opening Range Breakout (ORB) strategy for US equities.

    This strategy identifies stocks with unusually high relative volume in the first
    few minutes of the trading day and attempts to trade a breakout of that opening range.
    It incorporates risk management by sizing positions based on a fixed percentage of
    the portfolio and using an ATR-based stop loss.
    """

    def __init__(self, config):
        """
        Initializes the ORB strategy with a configuration dictionary.

        Args:
            config (dict): A dictionary containing all strategy parameters,
                           typically loaded from a YAML or JSON file.
                           Expected keys: 'default_tickers', 'parameters'.
        """
        self.params = config['parameters']
        self.default_tickers = config['default_tickers']
        self.signals = {}

    def select_universe(self, fundamental_data):
        """
        Selects the initial universe of stocks based on dollar volume.
        This method simulates the QuantConnect universe selection.

        Args:
            fundamental_data (pd.DataFrame): A DataFrame with fundamental data,
                                             requiring 'symbol' and 'dollar_volume' columns.

        Returns:
            list: A list of symbols for the tradable universe.
        """
        print(f"Selecting universe of top {self.params['universe_size']} stocks by dollar volume.")
        # Sort by dollar volume and take the top N stocks
        sorted_by_volume = fundamental_data.sort_values(by='dollar_volume', ascending=False)
        universe = sorted_by_volume.head(self.params['universe_size'])['symbol'].tolist()
        return universe

    def generate_signals(self, market_data, universe_symbols, portfolio_value):
        """
        Generates entry and exit signals for a given day.

        This is the core logic, simulating the '_scan_for_entries' and '_exit' methods
        from the notebook.

        Args:
            market_data (dict): A dictionary where keys are symbols and values are DataFrames
                                of market data (both daily for ATR/volume and intraday for ORB).
                                Expected columns: 'Open', 'High', 'Low', 'Close', 'Volume', 'ATR'.
            universe_symbols (list): The list of symbols to scan for entries.
            portfolio_value (float): The current total value of the portfolio.

        Returns:
            dict: A dictionary of trading orders for the day.
                  e.g., {'AAPL': {'action': 'BUY', 'quantity': 100, 'entry_price': 150.5, 'stop_price': 149.0}}
        """
        print("Scanning for opening range breakout entries...")

        # --- 1. Filter for stocks with high relative volume ("in play" stocks) ---
        volume_candidates = []
        for symbol in universe_symbols:
            # Requires both daily data (for avg volume and ATR) and minute data (for opening range)
            if symbol not in market_data or 'daily' not in market_data[symbol] or 'minute' not in market_data[symbol]:
                continue
            daily_data = market_data[symbol]['daily']
            minute_data = market_data[symbol]['minute']

            if daily_data.empty or minute_data.empty or len(daily_data) < self.params['atr_period']:
                continue

            # Calculate 14-day average daily volume for the denominator of the relative volume calc.
            avg_daily_volume = daily_data['Volume'].rolling(self.params['atr_period']).mean().iloc[-1]

            # Calculate the actual volume in the specified opening range (e.g., first 5 minutes).
            opening_range_volume = minute_data.head(self.params['opening_range_minutes'])['Volume'].sum()

            # Relative Volume is a key indicator for identifying unusual interest.
            # A value > 1 means the stock has traded more in the first 5 mins than its recent daily average.
            relative_volume = opening_range_volume / avg_daily_volume if avg_daily_volume > 0 else 0

            if relative_volume > 1.0:
                 volume_candidates.append({'symbol': symbol, 'relative_volume': relative_volume})

        if not volume_candidates:
            return {}

        # --- 2. Select top N candidates and generate entry/stop orders ---
        # We focus our capital on the stocks with the most significant interest.
        top_candidates = sorted(volume_candidates, key=lambda x: x['relative_volume'], reverse=True)
        top_candidates = top_candidates[:self.params['max_positions']]

        orders = {}
        for candidate in top_candidates:
            symbol = candidate['symbol']
            minute_data = market_data[symbol]['minute']
            daily_data = market_data[symbol]['daily']

            # Define the opening range based on the first N minutes.
            opening_range = minute_data.head(self.params['opening_range_minutes'])
            orb_high = opening_range['High'].max()
            orb_low = opening_range['Low'].min()
            orb_close = opening_range['Close'].iloc[-1]
            orb_open = opening_range['Open'].iloc[0]

            # ATR is used for setting a volatility-adjusted stop loss.
            atr = daily_data['ATR'].iloc[-1]

            action = None
            entry_price = 0
            stop_price = 0

            # If the opening bar is green, we prepare a long entry on a breakout of the high.
            if orb_close > orb_open:
                action = 'BUY'
                entry_price = orb_high
                stop_price = entry_price - (self.params['stop_loss_atr_multiplier'] * atr)
            # If red, we prepare a short entry on a breakdown of the low.
            else:
                action = 'SELL_SHORT'
                entry_price = orb_low
                stop_price = entry_price + (self.params['stop_loss_atr_multiplier'] * atr)

            # --- 3. Calculate Position Size based on Risk ---
            risk_per_share = abs(entry_price - stop_price)
            if risk_per_share == 0:
                continue

            # This is a key risk management rule: size the position so that a hit to the
            # stop-loss results in a predefined, acceptable loss to the portfolio.
            amount_to_risk = portfolio_value * self.params['portfolio_risk_per_trade']

            quantity = int(amount_to_risk / risk_per_share)

            if quantity > 0:
                orders[symbol] = {
                    'action': action,
                    'quantity': quantity,
                    'entry_price': entry_price,
                    'stop_price': stop_price,
                    'exit_time': 'EOD' # Signal to exit at End of Day
                }

        return orders
