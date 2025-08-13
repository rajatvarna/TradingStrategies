from dataclasses import dataclass

@dataclass
class Order:
    """
    Represents a trade order to be sent to the backtesting engine.
    """
    symbol: str
    action: str  # e.g., 'BUY', 'SELL_SHORT', 'SELL', 'COVER'
    quantity: float
