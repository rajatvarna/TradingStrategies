from abc import ABC
import pandas as pd

class Strategy(ABC):
    """
    Abstract base class for all trading strategies.

    A strategy can be either signal-based (implementing `generate_signals`)
    or order-based (implementing `generate_orders`).
    """

    def __init__(self, config: dict):
        """
        Initializes the strategy with a configuration dictionary.

        Args:
            config (dict): A dictionary containing all strategy parameters.
        """
        self.config = config
        self.tickers = config.get('tickers', [])
        self.start_date = config.get('start_date')
        self.end_date = config.get('end_date', pd.to_datetime('today').strftime('%Y-%m-%d'))
        self.params = config.get('parameters', {})
        self.signals = {}
        self.data = None

    @property
    def strategy_type(self) -> str:
        """
        Defines the type of the strategy, 'signal' or 'order'.
        This can be overridden by subclasses.
        """
        return 'signal'

    def generate_signals(self) -> dict:
        """
        (For Signal-Based Strategies)
        Generates trading signals (1 for buy, -1 for sell, 0 for hold).

        Returns:
            dict: A dictionary where keys are tickers and values are pandas
                  DataFrames containing a 'Signal' column.
        """
        # Default implementation for strategies that are not signal-based.
        return {}

    def generate_orders(self, portfolio_state: dict) -> list:
        """
        (For Order-Based Strategies)
        Generates a list of Order objects for the current day.

        Args:
            portfolio_state (dict): A dictionary containing the current state of the
                                    portfolio (e.g., date, capital, positions).

        Returns:
            list: A list of Order objects to be executed.
        """
        # Default implementation for strategies that are not order-based.
        return []
