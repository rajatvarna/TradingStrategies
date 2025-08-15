import unittest
import pandas as pd
import numpy as np
from quant_strategies.strategy_base import Strategy
from quant_strategies.backtester import Backtester

class MockStrategy(Strategy):
    """A simple mock strategy for testing."""
    def __init__(self, config, signal_data):
        super().__init__(config)
        self._signal_data = signal_data

    def generate_signals(self, data=None):
        return self._signal_data

class TestBacktester(unittest.TestCase):
    """Unit tests for the Backtester class."""

    def setUp(self):
        """Set up a mock strategy and data for testing."""
        self.tickers = ['AAPL', 'GOOG']
        dates = pd.to_datetime(pd.date_range(start='2023-01-01', periods=10, freq='D'))

        # Mock data with Close and Signal columns
        aapl_data = pd.DataFrame({
            'AAPL_Close': np.linspace(100, 110, 10),
            'AAPL_Signal': [1, 0, 0, -1, 1, 0, 0, 0, -1, 0]
        }, index=dates)

        goog_data = pd.DataFrame({
            'GOOG_Close': np.linspace(200, 220, 10),
            'GOOG_Signal': [0, 1, 0, 0, -1, 0, 1, 0, 0, -1]
        }, index=dates)

        # The backtester expects signals in a dictionary of DataFrames
        self.mock_signals = {
            'AAPL': aapl_data[['AAPL_Signal']].rename(columns={'AAPL_Signal': 'Signal'}),
            'GOOG': goog_data[['GOOG_Signal']].rename(columns={'GOOG_Signal': 'Signal'})
        }

        # The backtester also needs the price data for calculations
        # In the real code, the strategy provides the data, but here we pass it directly
        self.mock_data = pd.concat([aapl_data, goog_data], axis=1)

        strategy_config = {
            'tickers': self.tickers,
            'start_date': '2023-01-01',
            'end_date': '2023-01-10'
        }
        self.strategy = MockStrategy(strategy_config, self.mock_signals)

    def test_run_signal_based_backtest(self):
        """Test that the signal-based backtest runs and produces output."""
        # We need to provide the data with 'Close' columns to the backtester directly
        # because the MockStrategy doesn't download it.
        backtester = Backtester(self.strategy, initial_capital=100000)

        # The backtester's run method calls the strategy's generate_signals,
        # which in our mock returns self.mock_signals.
        # However, the backtester logic expects the data to be part of the signals dict DataFrames.
        # Let's adjust the mock signals to include the close price.

        aapl_signal_df = self.mock_signals['AAPL'].copy()
        aapl_signal_df['Close'] = self.mock_data['AAPL_Close']

        goog_signal_df = self.mock_signals['GOOG'].copy()
        goog_signal_df['Close'] = self.mock_data['GOOG_Close']

        # The strategy should return a dict of dataframes, each with a 'Signal' and 'Close' column
        self.strategy.generate_signals = lambda data=None: {'AAPL': aapl_signal_df, 'GOOG': goog_signal_df}

        equity_curve, trade_log = backtester.run()

        # 1. Check if the outputs are of the correct type
        self.assertIsInstance(equity_curve, pd.Series)
        self.assertIsInstance(trade_log, pd.DataFrame)

        # 2. Check if the equity curve is not empty
        self.assertFalse(equity_curve.empty)
        self.assertEqual(len(equity_curve), 10)

        # 3. Check if the trade log is not empty and has the correct columns
        self.assertFalse(trade_log.empty)
        expected_columns = ['Ticker', 'Buy Date', 'Buy Price', 'Sell Date', 'Sell Price', 'Return', 'Status']
        self.assertListEqual(list(trade_log.columns), expected_columns)

        # 4. Check if the initial capital is correct
        self.assertEqual(equity_curve.iloc[0], 100000)

    def test_calculate_statistics(self):
        """Test the performance statistics calculation with a predictable equity curve."""
        # 1. Create a simple, predictable equity curve
        dates = pd.to_datetime(pd.date_range(start='2022-01-01', end='2023-01-01', periods=252))
        equity = np.linspace(100000, 120000, 252)
        equity_curve = pd.Series(equity, index=dates)

        # 2. Manually calculate expected values
        # CAGR: (120000/100000)^(1/1) - 1 = 0.2 = 20%
        expected_cagr = 20.0
        # Max Drawdown: There is no drawdown in a linearly increasing curve
        expected_max_drawdown = 0.0

        # Create a backtester instance and set its equity curve
        backtester = Backtester(strategy=self.strategy, initial_capital=100000)
        backtester.equity_curve = equity_curve

        # 3. Calculate statistics
        # We need a benchmark to avoid errors, but for this test, alpha/beta are not important.
        stats = backtester.calculate_statistics(benchmark_ticker='SPY')

        # 4. Assert that the calculated values are close to the expected values
        self.assertAlmostEqual(stats['CAGR (%)'], expected_cagr, places=1)
        self.assertAlmostEqual(stats['Max Drawdown (%)'], expected_max_drawdown, places=1)

        # Volatility and Sharpe should be calculated, but their exact value is less intuitive.
        # We can assert they are not zero.
        self.assertNotEqual(stats['Volatility (%)'], 0)
        self.assertNotEqual(stats['Sharpe Ratio'], 0)


if __name__ == '__main__':
    unittest.main()
