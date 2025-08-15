import json
import pandas as pd
import numpy as np
import copy
import yfinance as yf

from web_app.models import Strategy
from quant_strategies.strategy_blocks import CustomStrategy
from quant_strategies.backtester import Backtester
from optimizer import run_optimization

def run_walk_forward_analysis(strategy_obj: Strategy, optimization_params: dict, in_sample_days: int, out_of_sample_days: int):
    """
    Performs a walk-forward analysis on a given strategy.
    """
    if not strategy_obj:
        raise ValueError("A valid strategy object must be provided.")

    print(f"--- Starting Walk-Forward Analysis for strategy: {strategy_obj.name} ---")
    base_config = json.loads(strategy_obj.config_json)

    # 1. Download the full dataset for the entire period
    print("Downloading full dataset...")
    full_data = yf.download(base_config['tickers'], start=base_config['start_date'], end=base_config['end_date'], group_by='ticker')

    # 2. Define walk-forward windows
    all_dates = full_data.index
    window_size = in_sample_days + out_of_sample_days
    step_size = out_of_sample_days
    num_windows = (len(all_dates) - window_size) // step_size + 1

    print(f"Total days: {len(all_dates)}. Windows to run: {num_windows}")

    all_oos_equity_curves = []

    # 3. Loop through each window
    for i in range(num_windows):
        start_idx = i * step_size

        in_sample_start_idx = start_idx
        in_sample_end_idx = start_idx + in_sample_days

        out_of_sample_start_idx = in_sample_end_idx
        out_of_sample_end_idx = out_of_sample_start_idx + out_of_sample_days

        if out_of_sample_end_idx > len(all_dates):
            break

        in_sample_data = full_data.iloc[in_sample_start_idx:in_sample_end_idx]
        out_of_sample_data = full_data.iloc[out_of_sample_start_idx:out_of_sample_end_idx]

        print(f"\n--- Window {i+1}/{num_windows} ---")
        print(f"In-sample: {in_sample_data.index[0].date()} to {in_sample_data.index[-1].date()}")
        print(f"Out-of-sample: {out_of_sample_data.index[0].date()} to {out_of_sample_data.index[-1].date()}")

        # 4. Run optimization on the in-sample data
        optimization_results = run_optimization(strategy_id, optimization_params, data=in_sample_data)

        # Find the best parameters (e.g., maximizing Sharpe Ratio)
        best_run = max(optimization_results, key=lambda r: r['performance'].get('Sharpe Ratio', -np.inf))
        best_params_config = best_run['parameters']

        print(f"Best parameter for this window: {best_params_config}")

        # 5. Run a backtest on the out-of-sample data with the best parameters
        oos_config = copy.deepcopy(base_config)

        # The 'parameters' dict contains the full path as the key
        param_path, param_value = list(best_params_config.items())[0]

        from optimizer import set_nested_param
        set_nested_param(oos_config, param_path, param_value)

        oos_strategy = CustomStrategy(config=oos_config)
        oos_backtester = Backtester(strategy=oos_strategy, initial_capital=100000, data=out_of_sample_data)
        oos_equity_curve, _ = oos_backtester.run()

        if oos_equity_curve is not None:
            all_oos_equity_curves.append(oos_equity_curve)

    # 6. Stitch the out-of-sample equity curves together
    if not all_oos_equity_curves:
        raise ValueError("Walk-forward analysis produced no valid out-of-sample runs.")

    # To stitch, we need to chain the returns, not the values.
    stitched_returns = pd.concat([curve.pct_change() for curve in all_oos_equity_curves]).dropna()
    initial_capital = 100000 # Standard initial capital
    stitched_equity_curve = initial_capital * (1 + stitched_returns).cumprod()

    # 7. Calculate final performance statistics on the stitched curve
    # This is a bit circular. We need a way to calculate stats without a full backtester object.
    # Let's create a temporary strategy object to pass to the backtester.
    temp_strategy_for_stats = CustomStrategy(config=base_config)
    final_stats_calculator = Backtester(strategy=temp_strategy_for_stats, initial_capital=initial_capital)
    final_stats_calculator.equity_curve = stitched_equity_curve
    final_stats = final_stats_calculator.calculate_statistics()

    print("\n--- Walk-Forward Analysis Finished ---")

    # Convert Timestamp keys to strings for JSON compatibility
    equity_curve_dict = {k.strftime('%Y-%m-%d'): v for k, v in stitched_equity_curve.to_dict().items()}

    return {
        'stitched_equity_curve': equity_curve_dict,
        'performance_metrics': final_stats
    }
