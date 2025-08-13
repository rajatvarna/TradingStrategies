import yfinance as yf
import pandas as pd
import os

class DataManager:
    """
    Handles sourcing, storing, and retrieving market data.
    """

    def __init__(self, data_dir='data'):
        """
        Initializes the DataManager.

        Args:
            data_dir (str): The directory to store data files.
        """
        self.data_dir = data_dir
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def download_data(self, tickers, start_date, end_date, file_name='market_data.parquet'):
        """
        Downloads historical data for a list of tickers and saves it.

        Args:
            tickers (list): List of ticker symbols.
            start_date (str): Start date in 'YYYY-MM-DD' format.
            end_date (str): End date in 'YYYY-MM-DD' format.
            file_name (str): The name of the file to save the data to.

        Returns:
            pd.DataFrame: A DataFrame containing the downloaded data.
        """
        print(f"Downloading data for {len(tickers)} tickers from {start_date} to {end_date}...")
        data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker')
        print("Download complete.")

        self.save_data(data, file_name)
        return data

    def save_data(self, data, file_name):
        """
        Saves a DataFrame to a Parquet file.

        Args:
            data (pd.DataFrame): The data to save.
            file_name (str): The name of the file.
        """
        file_path = os.path.join(self.data_dir, file_name)
        print(f"Saving data to {file_path}...")
        try:
            # Parquet requires a flat column index
            data.columns = ['_'.join(col).strip() for col in data.columns.values]
            data.to_parquet(file_path)
            print("Data saved successfully.")
        except Exception as e:
            print(f"Error saving data to Parquet: {e}")

    def load_data(self, file_name='market_data.parquet'):
        """
        Loads data from a Parquet file.

        Args:
            file_name (str): The name of the file to load.

        Returns:
            pd.DataFrame: The loaded data, or None if the file doesn't exist.
        """
        file_path = os.path.join(self.data_dir, file_name)
        if not os.path.exists(file_path):
            print(f"Data file not found at {file_path}. Please download the data first.")
            return None

        print(f"Loading data from {file_path}...")
        data = pd.read_parquet(file_path)

        # Restore the MultiIndex columns
        data.columns = pd.MultiIndex.from_tuples([tuple(c.split('_')) for c in data.columns])

        print("Data loaded successfully.")
        return data

if __name__ == '__main__':
    # Example Usage
    dm = DataManager()

    tickers = ["AAPL", "MSFT", "GOOG"]
    start = "2022-01-01"
    end = "2023-01-01"

    # Download and save the data
    fresh_data = dm.download_data(tickers, start, end, 'tech_stocks.parquet')

    # Load the data from the file
    loaded_data = dm.load_data('tech_stocks.parquet')

    if loaded_data is not None:
        print("\nLoaded data for AAPL:")
        print(loaded_data['AAPL'].head())
