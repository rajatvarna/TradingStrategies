from abc import ABC, abstractmethod
import pandas as pd

class Block(ABC):
    """Abstract base class for a strategy logic block."""
    @abstractmethod
    def evaluate(self, data: pd.DataFrame) -> pd.Series:
        """
        Evaluates the block's logic on the given data.

        Args:
            data (pd.DataFrame): A DataFrame containing market data for a single ticker.
                                 It must have a DatetimeIndex.

        Returns:
            pd.Series: The result of the evaluation. This could be a series of
                       floats (for indicators) or booleans (for conditions).
        """
        pass

# --- Value Blocks ---

class ValueBlock(Block):
    """Represents a constant numerical value."""
    def __init__(self, value: float):
        self.value = value

    def evaluate(self, data: pd.DataFrame) -> pd.Series:
        return pd.Series(self.value, index=data.index)

class PriceBlock(Block):
    """Represents a column from the input market data (e.g., 'Close', 'Volume')."""
    def __init__(self, column_name: str):
        if not isinstance(column_name, str):
            raise TypeError("column_name must be a string.")
        self.column_name = column_name

    def evaluate(self, data: pd.DataFrame) -> pd.Series:
        if self.column_name not in data.columns:
            raise ValueError(f"Column '{self.column_name}' not found in data.")
        return data[self.column_name]

# --- Indicator Blocks ---

class MovingAverageBlock(Block):
    """Calculates the simple moving average of an input block's output."""
    def __init__(self, input_block: Block, period: int):
        if not isinstance(input_block, Block):
            raise TypeError("input_block must be a Block instance.")
        if not isinstance(period, int) or period <= 0:
            raise ValueError("period must be a positive integer.")
        self.input_block = input_block
        self.period = period

    def evaluate(self, data: pd.DataFrame) -> pd.Series:
        series = self.input_block.evaluate(data)
        return series.rolling(window=self.period).mean()

# --- Comparison Blocks ---

class CrossesAboveBlock(Block):
    """
    Returns True on the day that the left block's value crosses above the right block's value.
    """
    def __init__(self, left_block: Block, right_block: Block):
        if not isinstance(left_block, Block) or not isinstance(right_block, Block):
            raise TypeError("Inputs must be Block instances.")
        self.left_block = left_block
        self.right_block = right_block

    def evaluate(self, data: pd.DataFrame) -> pd.Series:
        left_series = self.left_block.evaluate(data)
        right_series = self.right_block.evaluate(data)

        # Condition 1: Left is above right today
        condition1 = left_series > right_series
        # Condition 2: Left was below or equal to right yesterday
        condition2 = left_series.shift(1) <= right_series.shift(1)

        return (condition1 & condition2).fillna(False)

class GreaterThanBlock(Block):
    """Returns True if the left block's value is greater than the right block's value."""
    def __init__(self, left_block: Block, right_block: Block):
        if not isinstance(left_block, Block) or not isinstance(right_block, Block):
            raise TypeError("Inputs must be Block instances.")
        self.left_block = left_block
        self.right_block = right_block

    def evaluate(self, data: pd.DataFrame) -> pd.Series:
        left_series = self.left_block.evaluate(data)
        right_series = self.right_block.evaluate(data)
        return (left_series > right_series).fillna(False)

# --- Logical Blocks ---

class AndBlock(Block):
    """Performs a logical AND operation on the boolean outputs of two blocks."""
    def __init__(self, left_block: Block, right_block: Block):
        if not isinstance(left_block, Block) or not isinstance(right_block, Block):
            raise TypeError("Inputs must be Block instances.")
        self.left_block = left_block
        self.right_block = right_block

    def evaluate(self, data: pd.DataFrame) -> pd.Series:
        left_series = self.left_block.evaluate(data)
        right_series = self.right_block.evaluate(data)
        if left_series.dtype != 'bool' or right_series.dtype != 'bool':
            raise TypeError("Inputs to AndBlock must evaluate to boolean Series.")
        return (left_series & right_series).fillna(False)


import yfinance as yf
from quant_strategies.strategy_base import Strategy

# --- Strategy Execution ---

BLOCK_CLASS_MAP = {
    'ValueBlock': ValueBlock,
    'PriceBlock': PriceBlock,
    'MovingAverageBlock': MovingAverageBlock,
    'CrossesAboveBlock': CrossesAboveBlock,
    'GreaterThanBlock': GreaterThanBlock,
    'AndBlock': AndBlock,
}

class CustomStrategy(Strategy):
    """
    A strategy that executes logic defined by a configuration of Blocks.
    """
    def __init__(self, config: dict):
        super().__init__(config)
        self.entry_rule = self._parse_block_config(self.params['entry_rule'])
        self.exit_rule = self._parse_block_config(self.params['exit_rule'])

    def _parse_block_config(self, block_config: dict) -> Block:
        """
        Recursively parses a dictionary configuration and instantiates Block objects.
        """
        class_name = block_config.get('class')
        if not class_name:
            raise ValueError("Block configuration must have a 'class' key.")

        block_class = BLOCK_CLASS_MAP.get(class_name)
        if not block_class:
            raise ValueError(f"Unknown block class: {class_name}")

        # Recursively parse child blocks
        args = {}
        for key, value in block_config.items():
            if key == 'class':
                continue
            if isinstance(value, dict) and 'class' in value:
                args[key] = self._parse_block_config(value)
            else:
                args[key] = value

        return block_class(**args)

    def _download_data(self):
        """Downloads historical data for all tickers."""
        print("Downloading data for CustomStrategy...")
        self.data = yf.download(self.tickers, start=self.start_date, end=self.end_date, group_by='ticker')
        print("Data download complete.")

    def generate_signals(self) -> dict:
        """
        Generates signals by evaluating the entry and exit rule blocks for each ticker.
        """
        self._download_data()

        for ticker in self.tickers:
            print(f"Calculating signals for {ticker}...")

            if ticker not in self.data or self.data[ticker].isnull().all().all():
                print(f"No data for {ticker}, skipping.")
                self.signals[ticker] = pd.DataFrame()
                continue

            ticker_data = self.data[ticker].copy().dropna()
            if ticker_data.empty:
                self.signals[ticker] = pd.DataFrame()
                continue

            # Evaluate entry and exit rules
            entry_signals = self.entry_rule.evaluate(ticker_data)
            exit_signals = self.exit_rule.evaluate(ticker_data)

            # Create Signal column
            df = ticker_data.copy()
            df['Signal'] = 0
            df.loc[entry_signals, 'Signal'] = 1
            df.loc[exit_signals, 'Signal'] = -1

            self.signals[ticker] = df

        print("Signal generation complete for CustomStrategy.")
        return self.signals
