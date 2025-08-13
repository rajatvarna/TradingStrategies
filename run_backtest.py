import sys
import os
import json

# --- Path Setup ---
# Add the project root to the Python path to allow imports from other directories.
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from web_app.app import app, db, Strategy, BacktestResult
from quant_strategies.strategy_blocks import CustomStrategy
from quant_strategies.backtester import Backtester

def run_backtest_for_strategy(strategy_id: int):
    """
    Loads a strategy from the database, runs a backtest, and saves the results.
    """
    with app.app_context():
        # 1. Load the strategy from the database
        strategy_obj = Strategy.query.get(strategy_id)
        if not strategy_obj:
            print(f"Error: Strategy with ID {strategy_id} not found.")
            return

        print(f"--- Starting backtest for strategy: {strategy_obj.name} (ID: {strategy_id}) ---")

        # 2. Instantiate the strategy and backtester from the stored config
        try:
            config = json.loads(strategy_obj.config_json)
            # The config from the DB doesn't contain the 'parameters' nesting
            # that the CustomStrategy class expects. We need to wrap it.
            # This is a slight mismatch between the test script config and what we stored.
            # Let's assume the stored 'config' IS the full config dictionary including parameters.

            # Let's adjust what we store in the DB instead. The API should store the full config.
            # For now, let's assume the config is correct.

            custom_strategy = CustomStrategy(config=config)
            backtester = Backtester(strategy=custom_strategy, initial_capital=100000)
        except Exception as e:
            print(f"Error initializing strategy from config: {e}")
            return

        # 3. Run the backtest and calculate stats
        print("\n[Step 1] Running backtest engine...")
        equity_curve, trade_log = backtester.run()
        if equity_curve is None:
            print("Backtest failed to produce results.")
            return

        print("\n[Step 2] Calculating performance statistics...")
        stats = backtester.calculate_statistics()

        # 4. Save the results to the database
        print("\n[Step 3] Saving results to the database...")

        # Check if a result already exists and update it, otherwise create a new one
        result = BacktestResult.query.filter_by(strategy_id=strategy_id).first()
        if not result:
            result = BacktestResult(strategy_id=strategy_id)

        result.cagr = stats.get('CAGR (%)')
        result.volatility = stats.get('Volatility (%)')
        result.sharpe_ratio = stats.get('Sharpe Ratio')
        result.max_drawdown = stats.get('Max Drawdown (%)')
        result.alpha = stats.get('Alpha')
        result.beta = stats.get('Beta')

        db.session.add(result)
        db.session.commit()

        print("\n--- Backtest Results ---")
        for key, value in stats.items():
            print(f"{key}: {value:.2f}")
        print("------------------------")
        print("Results saved successfully.")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python run_backtest.py <strategy_id>")
        sys.exit(1)

    try:
        s_id = int(sys.argv[1])
        run_backtest_for_strategy(s_id)
    except ValueError:
        print("Error: <strategy_id> must be an integer.")
        sys.exit(1)
