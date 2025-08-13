import yaml
import pandas as pd
from quant_strategies.crypto_momentum import CryptoMomentumStrategy
from quant_strategies.data_manager import DataManager

def main():
    """
    Test script to verify the functionality of the quant_strategies library.
    """
    print("--- Starting Library Test ---")

    # 1. Load Configuration
    print("\n1. Loading configuration...")
    try:
        with open('quant_strategies/config.yml', 'r') as f:
            config = yaml.safe_load(f)
        crypto_config = config['crypto_momentum']
        print("Configuration loaded successfully.")
    except Exception as e:
        print(f"Error loading config file: {e}")
        return

    # 2. Initialize and use DataManager
    # For this test, we'll let the strategy class handle the data manager
    # to test its internal data handling. A more advanced setup would
    # pass a data manager instance to the strategy.

    # 3. Initialize the Strategy
    print("\n2. Initializing Crypto Momentum Strategy...")
    # Use a smaller set of tickers and a shorter date range for a quick test
    test_config = crypto_config.copy()
    test_config['tickers'] = ["BTC-USD", "ETH-USD", "SOL-USD"]
    test_config['start_date'] = "2022-01-01"

    strategy = CryptoMomentumStrategy(test_config)
    print("Strategy initialized.")

    # 4. Run Signal Generation
    print("\n3. Running signal generation...")
    try:
        signals = strategy.run_signal_generation()
        print("Signal generation completed.")
    except Exception as e:
        print(f"An error occurred during signal generation: {e}")
        return

    # 5. Verify Output
    print("\n4. Verifying output...")
    if signals and "ETH-USD" in signals and not signals["ETH-USD"].empty:
        print("\nSuccessfully generated signals for ETH-USD. Displaying latest signals:")
        print(signals["ETH-USD"][['Close', 'Signal']].tail())
        print("\n--- Library Test Passed ---")
    else:
        print("\n--- Library Test Failed: No signals were generated for ETH-USD. ---")

if __name__ == '__main__':
    main()
