import sys
import os
import json
import numpy as np
import copy

# --- Path Setup ---
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from web_app.app import app, db, Strategy
from quant_strategies.strategy_blocks import CustomStrategy
from quant_strategies.backtester import Backtester

def set_nested_param(config: dict, param_path: str, value):
    """
    Sets a nested parameter in a dictionary using a dot-separated path.
    Example: set_nested_param(config, 'parameters.entry_rule.period', 20)
    """
    keys = param_path.split('.')
    current_level = config
    for key in keys[:-1]:
        current_level = current_level[key]
    current_level[keys[-1]] = value

def run_optimization(strategy_id: int, optimization_params: dict):
    """
    Runs a parameter optimization for a given strategy and returns the results.
    """
    with app.app_context():
        # 1. Load the base strategy from the database
        strategy_obj = Strategy.query.get(strategy_id)
        if not strategy_obj:
            raise ValueError(f"Strategy with ID {strategy_id} not found.")

        print(f"--- Starting optimization for strategy: {strategy_obj.name} ---")
        base_config = json.loads(strategy_obj.config_json)

        # 2. Get optimization parameters
        param_path = optimization_params['parameter_name']
        start = optimization_params['start']
        end = optimization_params['end']
        step = optimization_params['step']

        param_range = np.arange(start, end + step, step)
        all_results = []

        # 3. Loop through each parameter value and run a backtest
        for param_value in param_range:
            print(f"\nTesting {param_path} = {param_value}...")

            # Create a deep copy to avoid modifying the base config
            run_config = copy.deepcopy(base_config)

            # Update the nested parameter
            try:
                # The parameter value from arange might be a numpy type, convert to standard Python type
                py_param_value = param_value.item() if hasattr(param_value, 'item') else param_value
                set_nested_param(run_config, param_path, py_param_value)
            except KeyError:
                raise ValueError(f"Parameter path '{param_path}' not found in strategy config.")

            # Run the backtest for this parameter set
            try:
                strategy = CustomStrategy(config=run_config)
                backtester = Backtester(strategy=strategy, initial_capital=100000)

                # Suppress verbose output from backtester during optimization
                # (This would be a good future enhancement to the Backtester class)
                equity_curve, _ = backtester.run()

                if equity_curve is not None and not equity_curve.empty:
                    stats = backtester.calculate_statistics()
                    all_results.append({
                        'parameters': {param_path: py_param_value},
                        'performance': stats
                    })
                else:
                    all_results.append({
                        'parameters': {param_path: py_param_value},
                        'performance': {'error': 'Backtest failed to produce results.'}
                    })
            except Exception as e:
                print(f"  - Backtest failed for {param_path}={param_value} with error: {e}")
                all_results.append({
                    'parameters': {param_path: py_param_value},
                    'performance': {'error': str(e)}
                })

        print("\n--- Optimization Finished ---")
        return all_results
