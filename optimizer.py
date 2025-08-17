import json
import numpy as np
import copy

from quant_strategies.strategy_blocks import CustomStrategy
from quant_strategies.backtester import Backtester
from web_app.models import Strategy

def set_nested_param(config: dict, param_name: str, value):
    """
    Sets a parameter's value in the 'parameters' dictionary of a config.
    """
    if 'parameters' not in config or param_name not in config['parameters']:
        raise KeyError(f"Parameter '{param_name}' not found in config['parameters'].")
    config['parameters'][param_name]['value'] = value

def run_optimization(strategy_obj: Strategy, optimization_params: dict, data=None):
    """
    Runs a parameter optimization for a given strategy and returns the results.
    Can accept a pre-loaded DataFrame to run the optimizations on, in which case
    it will use that data instead of downloading it.
    """
    if not strategy_obj:
        raise ValueError("A valid strategy object must be provided.")

    print(f"--- Starting optimization for strategy: {strategy_obj.name} ---")
    base_config = json.loads(strategy_obj.config_json)

    param_name = optimization_params['parameter_name']
    if param_name not in base_config.get('parameters', {}):
        raise ValueError(f"Parameter '{param_name}' not found in strategy config.")

    param_def = base_config['parameters'][param_name]
    start = param_def.get('min', optimization_params.get('start'))
    end = param_def.get('max', optimization_params.get('end'))
    step = param_def.get('step', optimization_params.get('step', 1))

    if start is None or end is None:
        raise ValueError(f"Optimization range (min/max) not defined for parameter '{param_name}'.")

    param_range = np.arange(start, end + step, step)
    all_results = []

    for param_value in param_range:
        print(f"\nTesting {param_name} = {param_value}...")

        run_config = copy.deepcopy(base_config)

        try:
            py_param_value = param_value.item() if hasattr(param_value, 'item') else param_value
            set_nested_param(run_config, param_name, py_param_value)
        except KeyError:
            raise ValueError(f"Parameter '{param_name}' not found in strategy config.")

        try:
            strategy = CustomStrategy(config=run_config)
            # Pass the pre-loaded data to the backtester, if it exists.
            backtester = Backtester(strategy=strategy, initial_capital=100000, data=data)

            equity_curve, _ = backtester.run()

            if equity_curve is not None and not equity_curve.empty:
                stats = backtester.calculate_statistics()
                all_results.append({
                    'parameters': {param_name: py_param_value},
                    'performance': stats
                })
            else:
                 all_results.append({
                    'parameters': {param_name: py_param_value},
                    'performance': {'error': 'Backtest failed to produce results.'}
                })
        except Exception as e:
            print(f"  - Backtest failed for {param_name}={param_value} with error: {e}")
            all_results.append({
                'parameters': {param_name: py_param_value},
                'performance': {'error': str(e)}
            })

    print("\n--- Optimization Finished ---")
    return all_results
