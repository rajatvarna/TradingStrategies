import json
import pandas as pd
import numpy as np
import copy

from web_app.models import Portfolio
from quant_strategies.strategy_blocks import CustomStrategy
from quant_strategies.backtester import Backtester

class PortfolioBacktester:
    """
    Runs a backtest on a portfolio of multiple strategies.
    """
    def __init__(self, portfolio_obj: Portfolio, initial_capital=1000000):
        if not portfolio_obj:
            raise ValueError("A valid portfolio object must be provided.")
        self.portfolio = portfolio_obj
        self.initial_capital = initial_capital
        self.strategies = []
        self._load_strategies()

    def _load_strategies(self):
        """Instantiates the strategies from the portfolio object."""
        if not self.portfolio.strategies:
            raise ValueError("Portfolio contains no strategies.")

        for strategy_obj in self.portfolio.strategies:
            config = json.loads(strategy_obj.config_json)
            self.strategies.append(CustomStrategy(config=config))

    def _combine_signals(self):
        """
        Generates signals for all strategies and combines them.

        A simple combination logic is used:
        - Signals are summed. A positive sum is a BUY (1).
        - A negative sum is a SELL (-1).
        - A sum of zero is a HOLD (0).
        - This allows strategies to vote on the final decision.
        """
        print("Generating signals for all strategies in the portfolio...")
        all_signals = []
        for i, strategy in enumerate(self.strategies):
            print(f"  - Generating signals for strategy {i+1}/{len(self.strategies)}...")
            signals = strategy.generate_signals()
            all_signals.append(signals)

        print("Combining signals...")

        # Use the first strategy's signal dict as a base to get the structure and index
        if not all_signals:
            return {}

        # Get all unique tickers and a master index
        all_tickers = set(t for sig_dict in all_signals for t in sig_dict.keys())
        master_index = pd.Index([])
        for sig_dict in all_signals:
            for df in sig_dict.values():
                master_index = master_index.union(df.index)

        combined_signals = {}
        for ticker in all_tickers:
            # Create a base DataFrame for each ticker with a Signal column of zeros
            combined_df = pd.DataFrame(index=master_index)
            combined_df['Signal'] = 0

            # Add signals from all strategies for this ticker
            for signals_dict in all_signals:
                if ticker in signals_dict and not signals_dict[ticker].empty:
                    s2 = signals_dict[ticker]['Signal']
                    combined_df['Signal'] = combined_df['Signal'].add(s2, fill_value=0)

            # Copy other columns from the first available DataFrame for this ticker
            for signals_dict in all_signals:
                 if ticker in signals_dict and not signals_dict[ticker].empty:
                    for col in signals_dict[ticker].columns:
                        if col != 'Signal':
                            combined_df[col] = signals_dict[ticker][col]
                    break # copy columns only once

            combined_signals[ticker] = combined_df

        # Normalize the summed signals back to -1, 0, 1
        for ticker in combined_signals:
            if not combined_signals[ticker].empty:
                combined_signals[ticker]['Signal'] = np.sign(combined_signals[ticker]['Signal']).astype(int)

        return combined_signals

    def run(self):
        """
        Runs the full portfolio backtest.
        """
        self._load_portfolio_and_strategies()

        # Generate the combined signals first
        combined_signals = self._combine_signals()

        # Create a simple dummy object that behaves like a Strategy for the Backtester
        class DummyMasterStrategy:
            def __init__(self, strategies):
                self.tickers = list(set(t for s in strategies for t in s.tickers))
                self.strategy_type = 'signal'

            def generate_signals(self, data=None):
                # Return the pre-combined signals
                return combined_signals

        master_strategy = DummyMasterStrategy(self.strategies)

        # Use the existing Backtester with the master strategy
        print("\nRunning portfolio backtest with combined signals...")
        portfolio_backtester = Backtester(strategy=master_strategy, initial_capital=self.initial_capital)

        equity_curve, trade_log = portfolio_backtester.run()

        if equity_curve is None:
            raise ValueError("Portfolio backtest failed to produce results.")

        stats = portfolio_backtester.calculate_statistics()

        return {
            'portfolio_name': self.portfolio.name,
            'performance_metrics': stats,
            'equity_curve': {k.strftime('%Y-%m-%d'): v for k, v in equity_curve.to_dict().items()}
        }
