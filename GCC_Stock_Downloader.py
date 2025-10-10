"""
Gulf Stock Market Data Downloader - All-in-One Solution
Downloads historical stock data from ADX, DFM, and Tadawul exchanges
Saves data in separate folders with proper organization

Author: Quant Developer
Date: 2025
"""

import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import time
import warnings
import json
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Base directory for all data
BASE_DATA_DIR = 'gulf_stocks_data'

# Delay between downloads (seconds) - be respectful to servers
DOWNLOAD_DELAY = 1

# Download period: 'max' gets all available data from earliest to latest
DOWNLOAD_PERIOD = 'max'

# ============================================================================
# TICKER LISTS
# ============================================================================

# DFM (Dubai Financial Market) - Suffix: .AE
DFM_TICKERS = [
    'EMAAR.AE',      # Emaar Properties
    'DFM.AE',        # Dubai Financial Market
    'DEWA.AE',       # Dubai Electricity & Water Authority
    'AMANAT.AE',     # Amanat Holdings
    'DIB.AE',        # Dubai Islamic Bank
    'TABREED.AE',    # National Central Cooling
    'TECOM.AE',      # Tecom Group
    'SHUAA.AE',      # Shuaa Capital
    'GFH.AE',        # GFH Financial Group
    'PARKIN.AE',     # Parkin Company
    'SALIK.AE',      # Salik Company
    'EMAARDV.AE',    # Emaar Development
    'UNION.AE',      # Union Properties
    'DAMAC.AE',      # DAMAC Properties
    'EKTTITAB.AE',   # Ekttitab Holding
    'GGICO.AE',      # Green Emirates Insurance
    'TAKAFUL.AE',    # Dubai Islamic Insurance
    'DUIND.AE',      # Dubai Investments
    'ALRAMZ.AE',     # Al Ramz Corporation
    'AIRARABIA.AE',  # Air Arabia
]

# ADX (Abu Dhabi Securities Exchange) - Suffix: .AE or .AD
ADX_TICKERS = [
    'FAB.AE',        # First Abu Dhabi Bank
    'ADCB.AE',       # Abu Dhabi Commercial Bank
    'ADIB.AE',       # Abu Dhabi Islamic Bank
    'ADNOCDIST.AE',  # ADNOC Distribution
    'ADNOCGAS.AE',   # ADNOC Gas
    'TAQA.AE',       # Abu Dhabi National Energy (TAQA)
    'ALDAR.AE',      # Aldar Properties
    'DANA.AE',       # Dana Gas
    'ESHRAQ.AE',     # Eshraq Investments
    'ADNH.AE',       # ADNOC Drilling
    'SCI.AE',        # Sharjah Cement & Industrial Development
    'RAKWCT.AE',     # RAK Ceramics
    'ARKAN.AE',      # Arkan Building Materials
    'UNB.AE',        # Union National Bank
    'ADNIC.AE',      # Abu Dhabi National Insurance
    'BURJEEL.AE',    # Burjeel Holdings
    'PUREHEALTH.AE', # Pure Health
    'ADPORTS.AE',    # AD Ports Group
    'MULTIPLY.AE',   # Multiply Group
    'ORIENT.AE',     # Orient Insurance
]

# Tadawul (Saudi Stock Exchange) - Suffix: .SR
TADAWUL_TICKERS = [
    # Banking Sector
    '1120.SR',  # Al Rajhi Bank
    '1180.SR',  # Saudi National Bank
    '1050.SR',  # Banque Saudi Fransi
    '1060.SR',  # Riyad Bank
    '1150.SR',  # Alinma Bank
    '1010.SR',  # National Commercial Bank
    '1080.SR',  # Arab National Bank
    
    # Energy & Petrochemicals
    '2222.SR',  # Saudi Aramco
    '2010.SR',  # SABIC
    '2020.SR',  # SABIC Innovative Plastics
    '2030.SR',  # Saudi Kayan Petrochemical
    '2060.SR',  # Saudi Paper Manufacturing
    '2090.SR',  # JPMC
    '1211.SR',  # Maaden (Saudi Arabian Mining)
    '2001.SR',  # SABIC Agri-Nutrients
    
    # Telecommunications
    '4030.SR',  # Saudi Telecom Company (STC)
    '4061.SR',  # Etihad Etisalat (Mobily)
    '4040.SR',  # Zain KSA
    
    # Utilities
    '4280.SR',  # Saudi Electricity Company
    '4081.SR',  # Marafiq
    
    # Healthcare
    '4004.SR',  # Dr. Sulaiman Al Habib Medical Services
    '4003.SR',  # Saudi German Health
    '4006.SR',  # Mouwasat Medical Services
    '4005.SR',  # Care (Bupa Arabia)
    
    # Retail & Consumer
    '4001.SR',  # Extra Stores
    '4008.SR',  # Jarir Marketing
    '1301.SR',  # Savola Group
    '2280.SR',  # Almarai Company
    
    # Real Estate
    '4020.SR',  # Dar Al Arkan Real Estate
    '4090.SR',  # Taiba Holding
    '4140.SR',  # Jabal Omar Development
    
    # Insurance
    '8010.SR',  # Tawuniya (The Company for Cooperative Insurance)
    '8012.SR',  # Bupa Arabia for Cooperative Insurance
    '8020.SR',  # Al Alamiya for Cooperative Insurance
    
    # Industrial
    '2110.SR',  # Saudi Cement Company
    '2050.SR',  # Safco (Saudi Arabian Fertilizer)
    '3010.SR',  # Saudi Cable Company
    
    # Transport
    '4031.SR',  # Saudi Airlines Catering
    '4260.SR',  # Bahri (National Shipping Company)
    
    # Agriculture & Food
    '6001.SR',  # Halwani Bros
    '6010.SR',  # Nadec (National Agricultural Development)
    
    # Hotels & Tourism
    '4170.SR',  # Dur Hospitality Company
    '4320.SR',  # Shams (Yanbu National Petrochemical)
]

# ============================================================================
# MAIN DOWNLOADER CLASS
# ============================================================================

class GulfStockDownloader:
    """Complete stock data downloader for Gulf exchanges"""
    
    def __init__(self, base_path=BASE_DATA_DIR):
        self.base_path = Path(base_path)
        self.exchanges = {
            'ADX': {'path': self.base_path / 'ADX', 'tickers': ADX_TICKERS},
            'DFM': {'path': self.base_path / 'DFM', 'tickers': DFM_TICKERS},
            'Tadawul': {'path': self.base_path / 'Tadawul', 'tickers': TADAWUL_TICKERS}
        }
        self._create_directories()
        self.download_stats = {
            'total_attempted': 0,
            'successful': 0,
            'failed': 0,
            'start_time': None,
            'end_time': None
        }
    
    def _create_directories(self):
        """Create directory structure for each exchange"""
        print(f"\n{'='*70}")
        print("Setting up directory structure...")
        print(f"{'='*70}")
        
        for exchange_name, exchange_info in self.exchanges.items():
            exchange_path = exchange_info['path']
            exchange_path.mkdir(parents=True, exist_ok=True)
            (exchange_path / 'daily').mkdir(exist_ok=True)
            (exchange_path / 'metadata').mkdir(exist_ok=True)
            print(f"✓ Created: {exchange_path}")
    
    def download_stock_data(self, ticker, exchange_name):
        """
        Download historical data for a single stock
        Gets all available data from earliest to latest
        """
        try:
            print(f"  {ticker:20s} ", end='', flush=True)
            
            # Download maximum available data
            data = yf.download(
                ticker, 
                period=DOWNLOAD_PERIOD,
                progress=False
            )
            
            if data.empty:
                print("❌ No data available")
                return None
            
            # Sort by date (earliest to latest)
            data = data.sort_index()
            
            # Clean ticker name for filename
            clean_ticker = ticker.replace('.', '_')
            
            # Save to CSV
            output_path = self.exchanges[exchange_name]['path'] / 'daily' / f'{clean_ticker}.csv'
            data.to_csv(output_path)
            
            # Get date range
            start_date = data.index[0].strftime('%Y-%m-%d')
            end_date = data.index[-1].strftime('%Y-%m-%d')
            
            print(f"✓ {len(data):5d} rows [{start_date} to {end_date}]")
            
            # Get additional stock info
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                
                metadata = {
                    'Ticker': ticker,
                    'Name': info.get('longName', info.get('shortName', 'N/A')),
                    'Currency': info.get('currency', 'N/A'),
                    'Exchange': info.get('exchange', exchange_name),
                    'Sector': info.get('sector', 'N/A'),
                    'Industry': info.get('industry', 'N/A'),
                    'Market_Cap': info.get('marketCap', 'N/A'),
                    'Start_Date': start_date,
                    'End_Date': end_date,
                    'Data_Points': len(data),
                    'Download_Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                return metadata
                
            except Exception as e:
                # Return basic metadata if detailed info fails
                return {
                    'Ticker': ticker,
                    'Name': 'N/A',
                    'Start_Date': start_date,
                    'End_Date': end_date,
                    'Data_Points': len(data),
                    'Download_Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
        except Exception as e:
            print(f"❌ Error: {str(e)[:40]}")
            return None
    
    def download_exchange_data(self, exchange_name):
        """Download data for all tickers in an exchange"""
        print(f"\n{'='*70}")
        print(f"Downloading {exchange_name} Stocks")
        print(f"{'='*70}")
        
        tickers = self.exchanges[exchange_name]['tickers']
        metadata_list = []
        
        successful = 0
        failed = 0
        
        for i, ticker in enumerate(tickers, 1):
            print(f"[{i:3d}/{len(tickers):3d}] ", end='')
            
            metadata = self.download_stock_data(ticker, exchange_name)
            
            if metadata:
                metadata_list.append(metadata)
                successful += 1
            else:
                failed += 1
            
            self.download_stats['total_attempted'] += 1
            
            # Delay between requests
            if i < len(tickers):
                time.sleep(DOWNLOAD_DELAY)
        
        # Save metadata summary
        if metadata_list:
            df_metadata = pd.DataFrame(metadata_list)
            metadata_path = self.exchanges[exchange_name]['path'] / 'metadata' / 'stock_info.csv'
            df_metadata.to_csv(metadata_path, index=False)
            
            print(f"\n{'='*70}")
            print(f"✓ {exchange_name} Summary: {successful} successful, {failed} failed")
            print(f"✓ Metadata saved: {metadata_path}")
            print(f"{'='*70}")
        
        return metadata_list
    
    def download_all_exchanges(self):
        """Download data from all three exchanges"""
        print("\n" + "="*70)
        print("GULF STOCK MARKET DATA DOWNLOADER")
        print("="*70)
        print(f"Base Directory: {self.base_path.absolute()}")
        print(f"Download Period: {DOWNLOAD_PERIOD} (earliest to latest available)")
        print(f"Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)
        
        self.download_stats['start_time'] = datetime.now()
        all_results = {}
        
        # Download each exchange
        for exchange_name in ['DFM', 'ADX', 'Tadawul']:
            all_results[exchange_name] = self.download_exchange_data(exchange_name)
        
        self.download_stats['end_time'] = datetime.now()
        self.download_stats['successful'] = sum(
            len([m for m in results if m]) 
            for results in all_results.values()
        )
        self.download_stats['failed'] = (
            self.download_stats['total_attempted'] - 
            self.download_stats['successful']
        )
        
        # Create summary
        self._create_summary(all_results)
        
        return all_results
    
    def _create_summary(self, results):
        """Create comprehensive summary report"""
        summary_path = self.base_path / 'DOWNLOAD_SUMMARY.txt'
        
        duration = (
            self.download_stats['end_time'] - 
            self.download_stats['start_time']
        )
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("GULF STOCK MARKET DOWNLOAD SUMMARY\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Download Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Duration: {duration}\n")
            f.write(f"Period: {DOWNLOAD_PERIOD} (all available historical data)\n\n")
            
            f.write("="*70 + "\n")
            f.write("EXCHANGE STATISTICS\n")
            f.write("="*70 + "\n\n")
            
            total_stocks = 0
            total_successful = 0
            
            for exchange, data in results.items():
                successful = len([d for d in data if d])
                attempted = len(data)
                total_stocks += attempted
                total_successful += successful
                
                # Protect against division by zero when no tickers attempted
                if attempted > 0:
                    success_rate = successful / attempted * 100
                else:
                    success_rate = 0.0
                
                f.write(f"{exchange}:\n")
                f.write(f"  Stocks Attempted:     {attempted}\n")
                f.write(f"  Successfully Downloaded: {successful}\n")
                f.write(f"  Failed:                {attempted - successful}\n")
                f.write(f"  Success Rate:          {success_rate:.1f}%\n\n")
            
            f.write("="*70 + "\n")
            f.write("OVERALL STATISTICS\n")
            f.write("="*70 + "\n\n")
            # Protect against division by zero for overall totals
            if total_stocks > 0:
                overall_rate = total_successful / total_stocks * 100
            else:
                overall_rate = 0.0
            f.write(f"Total Stocks Attempted:     {total_stocks}\n")
            f.write(f"Total Successfully Downloaded: {total_successful}\n")
            f.write(f"Total Failed:                {total_stocks - total_successful}\n")
            f.write(f"Overall Success Rate:        {overall_rate:.1f}%\n\n")
            
            f.write("="*70 + "\n")
            f.write("DIRECTORY STRUCTURE\n")
            f.write("="*70 + "\n\n")
            f.write(f"{self.base_path}/\n")
            f.write("├── ADX/\n")
            f.write("│   ├── daily/           (CSV files with OHLCV data)\n")
            f.write("│   └── metadata/        (Stock information)\n")
            f.write("├── DFM/\n")
            f.write("│   ├── daily/\n")
            f.write("│   └── metadata/\n")
            f.write("├── Tadawul/\n")
            f.write("│   ├── daily/\n")
            f.write("│   └── metadata/\n")
            f.write("└── DOWNLOAD_SUMMARY.txt (This file)\n\n")
            
            f.write("="*70 + "\n")
            f.write("DATA FORMAT\n")
            f.write("="*70 + "\n\n")
            f.write("Each CSV file contains:\n")
            f.write("  - Date (index): Trading date\n")
            f.write("  - Open: Opening price\n")
            f.write("  - High: Highest price\n")
            f.write("  - Low: Lowest price\n")
            f.write("  - Close: Closing price\n")
            f.write("  - Adj Close: Adjusted closing price\n")
            f.write("  - Volume: Trading volume\n\n")
            
            f.write("Data is sorted from earliest to latest date.\n\n")
            
            f.write("="*70 + "\n")
            f.write("USAGE NOTES\n")
            f.write("="*70 + "\n\n")
            f.write("1. To load data in Python:\n")
            f.write("   df = pd.read_csv('gulf_stocks_data/DFM/daily/EMAAR_AE.csv',\n")
            f.write("                    index_col=0, parse_dates=True)\n\n")
            f.write("2. To update data: Simply run this script again\n\n")
            f.write("3. Data source: Yahoo Finance\n\n")
            f.write("4. For questions or issues, check ticker symbols on exchange websites:\n")
            f.write("   - ADX: https://www.adx.ae\n")
            f.write("   - DFM: https://www.dfm.ae\n")
            f.write("   - Tadawul: https://www.saudiexchange.sa\n\n")
        
        # Print summary to console
        print(f"\n{'='*70}")
        print("DOWNLOAD COMPLETE!")
        print(f"{'='*70}")
        print(f"Total Stocks: {total_stocks}")
        print(f"Successful: {total_successful}")
        print(f"Failed: {total_stocks - total_successful}")
        if total_stocks > 0:
            print(f"Success Rate: {total_successful/total_stocks*100:.1f}%")
        else:
            print("Success Rate: 0.0%")
        print(f"Duration: {duration}")
        print(f"\n✓ Summary saved to: {summary_path}")
        print(f"✓ Data saved to: {self.base_path.absolute()}")
        print(f"{'='*70}\n")


# ============================================================================
# SIMPLE ANALYSIS FUNCTIONS
# ============================================================================

def load_stock_data(ticker, exchange='DFM'):
    """
    Quick function to load a stock's data
    
    Example:
        df = load_stock_data('EMAAR.AE', 'DFM')
    """
    clean_ticker = ticker.replace('.', '_')
    file_path = Path(BASE_DATA_DIR) / exchange / 'daily' / f'{clean_ticker}.csv'
    
    if file_path.exists():
        return pd.read_csv(file_path, index_col=0, parse_dates=True)
    else:
        print(f"File not found: {file_path}")
        return None


def get_basic_stats(ticker, exchange='DFM'):
    """
    Get basic statistics for a stock
    
    Example:
        stats = get_basic_stats('EMAAR.AE', 'DFM')
    """
    df = load_stock_data(ticker, exchange)
    
    if df is None or df.empty:
        return None
    
    returns = df['Close'].pct_change()
    
    stats = {
        'Ticker': ticker,
        'Start Date': df.index[0].strftime('%Y-%m-%d'),
        'End Date': df.index[-1].strftime('%Y-%m-%d'),
        'Total Days': len(df),
        'Current Price': df['Close'].iloc[-1],
        'Min Price': df['Close'].min(),
        'Max Price': df['Close'].max(),
        'Avg Volume': df['Volume'].mean(),
        'Total Return %': (df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100,
        'Avg Daily Return %': returns.mean() * 100,
        'Volatility (Annual) %': returns.std() * np.sqrt(252) * 100,
    }
    
    return stats


def list_available_stocks(exchange='DFM'):
    """
    List all downloaded stocks for an exchange
    
    Example:
        stocks = list_available_stocks('DFM')
    """
    exchange_path = Path(BASE_DATA_DIR) / exchange / 'daily'
    
    if not exchange_path.exists():
        print(f"No data found for {exchange}")
        return []
    
    stocks = []
    for csv_file in sorted(exchange_path.glob('*.csv')):
        ticker = csv_file.stem.replace('_', '.')
        stocks.append(ticker)
    
    return stocks


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n")
    print("*" * 70)
    print("*" + " " * 68 + "*")
    print("*" + "  GULF STOCK MARKET DATA DOWNLOADER - ALL-IN-ONE SOLUTION  ".center(68) + "*")
    print("*" + " " * 68 + "*")
    print("*" * 70)
    
    # Initialize and run downloader
    downloader = GulfStockDownloader(base_path=BASE_DATA_DIR)
    
    # Download all data
    results = downloader.download_all_exchanges()
    
    # Show example usage
    print("\n" + "="*70)
    print("QUICK START EXAMPLES")
    print("="*70)
    print("\nTo use the downloaded data in Python:\n")
    print("1. Load a stock:")
    print("   >>> df = load_stock_data('EMAAR.AE', 'DFM')")
    print("   >>> print(df.head())")
    print("\n2. Get basic statistics:")
    print("   >>> stats = get_basic_stats('EMAAR.AE', 'DFM')")
    print("   >>> print(stats)")
    print("\n3. List all available stocks:")
    print("   >>> stocks = list_available_stocks('DFM')")
    print("   >>> print(stocks)")
    print("\n4. List stocks from all exchanges:")
    print("   >>> for exchange in ['ADX', 'DFM', 'Tadawul']:")
    print("   >>>     print(f'{exchange}: {list_available_stocks(exchange)}')")
    print("\n" + "="*70)
    print("\n✓ All done! Check the 'gulf_stocks_data' folder for your data.")
    print("✓ Run this script anytime to download latest data.\n")