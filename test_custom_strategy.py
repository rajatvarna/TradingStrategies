import pandas as pd
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend
import matplotlib.pyplot as plt

from quant_strategies.strategy_blocks import CustomStrategy
from quant_strategies.backtester import Backtester

def run_test():
    """
    Defines and runs a test of the CustomStrategy with the Backtester.
    """
    print("--- Starting Custom Strategy Test ---")

    # 1. Define the strategy using the Block system configuration
    strategy_config = {
        'tickers': ["AAPL", "MSFT", "GOOG"],
        'start_date': "2021-01-01",
        'end_date': "2023-12-31",
        'parameters': {
            'entry_rule': {
                'class': 'CrossesAboveBlock',
                'left_block': {
                    'class': 'MovingAverageBlock',
                    'input_block': {'class': 'PriceBlock', 'column_name': 'Close'},
                    'period': 20
                },
                'right_block': {
                    'class': 'MovingAverageBlock',
                    'input_block': {'class': 'PriceBlock', 'column_name': 'Close'},
                    'period': 50
                }
            },
            'exit_rule': {
                # Exit when the short MA crosses back below the long MA,
                # which is equivalent to the long MA crossing above the short MA.
                'class': 'CrossesAboveBlock',
                'left_block': {
                    'class': 'MovingAverageBlock',
                    'input_block': {'class': 'PriceBlock', 'column_name': 'Close'},
                    'period': 50
                },
                'right_block': {
                    'class': 'MovingAverageBlock',
                    'input_block': {'class': 'PriceBlock', 'column_name': 'Close'},
                    'period': 20
                }
            }
        }
    }

    # 2. Instantiate the CustomStrategy
    print("\n[Step 1] Initializing CustomStrategy...")
    try:
        custom_strategy = CustomStrategy(config=strategy_config)
        print("CustomStrategy initialized successfully.")
    except Exception as e:
        print(f"Error initializing CustomStrategy: {e}")
        return

    # 3. Instantiate the Backtester
    print("\n[Step 2] Initializing Backtester...")
    backtester = Backtester(strategy=custom_strategy, initial_capital=100000)
    print("Backtester initialized successfully.")

    # 4. Run the backtest
    print("\n[Step 3] Running backtest...")
    equity_curve, trade_log = backtester.run()

    if equity_curve is None:
        print("Backtest did not produce an equity curve. Exiting.")
        return

    print("Backtest run completed.")
    print("\n--- Trade Log ---")
    print(trade_log.head())
    print("-----------------\n")

    # 5. Calculate and print statistics
    print("\n[Step 4] Calculating statistics...")
    stats = backtester.calculate_statistics(benchmark_ticker='SPY')
    print("\n--- Performance Stats ---")
    for key, value in stats.items():
        print(f"{key}: {value:.2f}")
    print("-----------------------\n")

    # 6. Plot and save the equity curve
    print("\n[Step 5] Plotting equity curve...")
    try:
        plt.figure(figsize=(12, 6))
        equity_curve.plot(title='MA Crossover Strategy - Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value')
        plt.grid(True)
        plt.savefig('equity_curve.png')
        print("Equity curve plot saved to 'equity_curve.png'")
    except Exception as e:
        print(f"Error plotting equity curve: {e}")

    print("\n--- Custom Strategy Test Finished ---")

if __name__ == '__main__':
    run_test()
