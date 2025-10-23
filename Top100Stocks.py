"""
Top 100 Stocks Data Download and Chart Generator - IMPROVED VERSION
====================================================================
Enhanced with better scraping, multiple fallback methods, and robust error handling

Requirements:
    pip install yfinance pandas matplotlib seaborn beautifulsoup4 requests lxml html5lib

Author: Claude (Anthropic)
Date: October 2025
"""

import requests
from bs4 import BeautifulSoup
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import time
import warnings
import json
import re
warnings.filterwarnings('ignore')

# Set style for professional-looking charts
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class StockDataDownloader:
    """Main class for downloading and processing stock data"""
    
    def __init__(self, output_dir='stock_analysis'):
        self.output_dir = output_dir
        self.data_dir = os.path.join(output_dir, 'data')
        self.charts_dir = os.path.join(output_dir, 'charts')
        self.setup_directories()
        
    def setup_directories(self):
        """Create necessary directory structure"""
        dirs = [
            self.data_dir,
            os.path.join(self.charts_dir, 'daily'),
            os.path.join(self.charts_dir, 'weekly'),
            os.path.join(self.charts_dir, 'monthly')
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)
        print(f"✓ Created directory structure in: {self.output_dir}")
    
    def get_sp500_tickers(self):
        """
        Fallback Method 1: Get S&P 500 tickers from Wikipedia
        This is more reliable than scraping Barchart
        """
        try:
            print("Fetching S&P 500 tickers from Wikipedia...")
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            
            # Read tables from Wikipedia
            tables = pd.read_html(url)
            df = tables[0]
            
            tickers = df['Symbol'].tolist()
            # Clean tickers (remove dots for class shares)
            tickers = [ticker.replace('.', '-') for ticker in tickers]
            
            print(f"✓ Successfully retrieved {len(tickers)} S&P 500 tickers")
            return tickers[:100]  # Return top 100
            
        except Exception as e:
            print(f"✗ Error fetching S&P 500 tickers: {e}")
            return None
    
    def get_nasdaq100_tickers(self):
        """
        Fallback Method 2: Get NASDAQ 100 tickers
        """
        try:
            print("Fetching NASDAQ 100 tickers from Wikipedia...")
            url = "https://en.wikipedia.org/wiki/Nasdaq-100"
            
            tables = pd.read_html(url)
            # Usually the first table contains the tickers
            for table in tables:
                if 'Ticker' in table.columns or 'Symbol' in table.columns:
                    ticker_col = 'Ticker' if 'Ticker' in table.columns else 'Symbol'
                    tickers = table[ticker_col].tolist()
                    # Clean up
                    tickers = [str(t).strip() for t in tickers if pd.notna(t)]
                    print(f"✓ Successfully retrieved {len(tickers)} NASDAQ 100 tickers")
                    return tickers
            
            return None
        except Exception as e:
            print(f"✗ Error fetching NASDAQ 100 tickers: {e}")
            return None
    
    def get_top_100_tickers_barchart(self):
        """
        Primary Method: Scrape top 100 tickers from Barchart
        """
        url = "https://www.barchart.com/stocks/top-100-stocks?viewName=main&orderBy=weightedAlpha&orderDir=desc"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        
        try:
            print("Attempting to fetch top 100 tickers from Barchart.com...")
            session = requests.Session()
            response = session.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            # Try multiple parsers
            for parser in ['lxml', 'html.parser', 'html5lib']:
                try:
                    soup = BeautifulSoup(response.content, parser)
                    tickers = []
                    
                    # Method 1: Look for symbol links
                    for link in soup.find_all('a', class_='symbol'):
                        ticker = link.get_text().strip()
                        if ticker and ticker not in tickers:
                            tickers.append(ticker)
                    
                    # Method 2: Look for data-symbol attributes
                    if len(tickers) < 50:
                        for tag in soup.find_all(attrs={'data-symbol': True}):
                            ticker = tag['data-symbol'].strip()
                            if ticker and ticker not in tickers:
                                tickers.append(ticker)
                    
                    # Method 3: Look in table cells
                    if len(tickers) < 50:
                        for row in soup.find_all('tr'):
                            cells = row.find_all('td')
                            if cells:
                                # First cell usually contains ticker
                                text = cells[0].get_text().strip()
                                # Check if it looks like a ticker (1-5 letters, uppercase)
                                if text and len(text) <= 5 and text.replace('.', '').replace('-', '').isalpha():
                                    if text not in tickers:
                                        tickers.append(text)
                    
                    if tickers:
                        tickers = tickers[:100]
                        print(f"✓ Successfully retrieved {len(tickers)} tickers from Barchart")
                        return tickers
                        
                except Exception as e:
                    continue
            
            print("✗ Could not parse tickers from Barchart")
            return None
            
        except Exception as e:
            print(f"✗ Error accessing Barchart: {e}")
            return None
    
    def get_fallback_tickers(self):
        """
        Ultimate Fallback: Curated list of top 100 liquid US stocks
        """
        print("Using curated list of top 100 liquid US stocks...")
        return [
            # Mega Cap Tech
            'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'META', 'TSLA', 'AVGO', 'ORCL',
            'ADBE', 'CRM', 'CSCO', 'ACN', 'AMD', 'INTC', 'IBM', 'NOW', 'INTU', 'QCOM',
            'TXN', 'AMAT', 'ADI', 'LRCX', 'KLAC', 'SNPS', 'CDNS', 'MRVL', 'PANW', 'FTNT',
            
            # Finance
            'BRK.B', 'JPM', 'V', 'MA', 'BAC', 'WFC', 'MS', 'GS', 'SCHW', 'AXP',
            'BLK', 'C', 'SPGI', 'CME', 'ICE', 'AON', 'MMC', 'PGR', 'TFC', 'USB',
            
            # Healthcare
            'UNH', 'JNJ', 'LLY', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'PFE', 'BMY',
            'AMGN', 'GILD', 'CVS', 'CI', 'ELV', 'BSX', 'ISRG', 'VRTX', 'REGN', 'ZTS',
            
            # Consumer
            'WMT', 'HD', 'PG', 'COST', 'KO', 'PEP', 'MCD', 'NKE', 'SBUX', 'TGT',
            'LOW', 'TJX', 'EL', 'PM', 'MO', 'CL', 'MDLZ', 'GIS', 'KHC', 'MNST',
            
            # Energy & Industrials
            'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'PSX', 'MPC', 'VLO', 'OXY', 'HES',
            'CAT', 'BA', 'HON', 'UPS', 'RTX', 'LMT', 'GE', 'MMM', 'DE', 'EMR'
        ]
    
    def get_tickers(self):
        """
        Master method to get tickers using multiple methods with fallbacks
        """
        print("\n" + "="*80)
        print("FETCHING TOP 100 TICKERS")
        print("="*80 + "\n")
        
        # Try Method 1: Barchart (original request)
        tickers = self.get_top_100_tickers_barchart()
        if tickers and len(tickers) >= 50:
            return tickers
        
        # Try Method 2: S&P 500 (most reliable)
        print("\nTrying fallback method: S&P 500 tickers...")
        tickers = self.get_sp500_tickers()
        if tickers and len(tickers) >= 50:
            return tickers
        
        # Try Method 3: NASDAQ 100
        print("\nTrying fallback method: NASDAQ 100 tickers...")
        tickers = self.get_nasdaq100_tickers()
        if tickers and len(tickers) >= 50:
            return tickers
        
        # Method 4: Use curated fallback list
        print("\nUsing curated fallback list...")
        return self.get_fallback_tickers()
    
    def download_stock_data(self, ticker, period="2y"):
        """
        Download historical data from Yahoo Finance
        """
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period)
            
            if df.empty:
                return None
                
            # Clean data
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            df = df.dropna()
            
            return df
            
        except Exception as e:
            return None
    
    def resample_to_weekly(self, df):
        """Convert daily data to weekly timeframe"""
        weekly = df.resample('W').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
        return weekly
    
    def resample_to_monthly(self, df):
        """Convert daily data to monthly timeframe"""
        monthly = df.resample('ME').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
        return monthly
    
    def calculate_indicators(self, df):
        """Calculate technical indicators"""
        # Moving averages
        if len(df) > 20:
            df['MA20'] = df['Close'].rolling(window=20).mean()
        if len(df) > 50:
            df['MA50'] = df['Close'].rolling(window=50).mean()
        if len(df) > 200:
            df['MA200'] = df['Close'].rolling(window=200).mean()
        
        # RSI
        if len(df) > 14:
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
        
        return df
    
    def create_professional_chart(self, df, ticker, timeframe, save_path):
        """
        Create professional multi-panel chart with price and volume
        """
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 1], hspace=0.1)
        
        # Price Chart
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(df.index, df['Close'], linewidth=2.5, label='Close Price', 
                color='#2E86AB', alpha=0.9)
        
        # Add candlestick-like visualization
        ax1.fill_between(df.index, df['Low'], df['High'], 
                         alpha=0.15, color='#A23B72', label='High-Low Range')
        
        # Moving averages
        if 'MA20' in df.columns and not df['MA20'].isna().all():
            ax1.plot(df.index, df['MA20'], linewidth=1.8, label='MA20', 
                    color='#F18F01', linestyle='--', alpha=0.8)
        
        if 'MA50' in df.columns and not df['MA50'].isna().all():
            ax1.plot(df.index, df['MA50'], linewidth=1.8, label='MA50', 
                    color='#C73E1D', linestyle='--', alpha=0.8)
        
        if 'MA200' in df.columns and not df['MA200'].isna().all():
            ax1.plot(df.index, df['MA200'], linewidth=1.8, label='MA200', 
                    color='#6A4C93', linestyle='--', alpha=0.7)
        
        # Formatting
        ax1.set_title(f'{ticker} - {timeframe} Price Chart', 
                     fontsize=18, fontweight='bold', pad=20)
        ax1.set_ylabel('Price ($)', fontsize=13, fontweight='bold')
        ax1.legend(loc='upper left', fontsize=10, framealpha=0.9)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_xticklabels([])
        
        # Add price statistics
        current_price = df['Close'].iloc[-1]
        price_change = df['Close'].iloc[-1] - df['Close'].iloc[0]
        pct_change = (price_change / df['Close'].iloc[0]) * 100
        
        stats_text = f'Current: ${current_price:.2f} | Change: {pct_change:+.2f}%'
        ax1.text(0.02, 0.97, stats_text, transform=ax1.transAxes,
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Volume Chart
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        colors = ['#26A96C' if df['Close'].iloc[i] >= df['Open'].iloc[i] 
                 else '#E63946' for i in range(len(df))]
        ax2.bar(df.index, df['Volume'], color=colors, alpha=0.7, width=0.8)
        ax2.set_ylabel('Volume', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_xticklabels([])
        
        # RSI Chart
        if 'RSI' in df.columns and not df['RSI'].isna().all():
            ax3 = fig.add_subplot(gs[2], sharex=ax1)
            ax3.plot(df.index, df['RSI'], linewidth=2, color='#7209B7', alpha=0.8)
            ax3.axhline(y=70, color='red', linestyle='--', alpha=0.5, linewidth=1)
            ax3.axhline(y=30, color='green', linestyle='--', alpha=0.5, linewidth=1)
            ax3.fill_between(df.index, 30, 70, alpha=0.1, color='gray')
            ax3.set_ylabel('RSI', fontsize=12, fontweight='bold')
            ax3.set_ylim(0, 100)
            ax3.grid(True, alpha=0.3, linestyle='--')
            ax3.set_xlabel('Date', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def process_all_tickers(self, tickers):
        """
        Main processing loop for all tickers
        """
        print(f"\n{'='*80}")
        print(f"PROCESSING {len(tickers)} TICKERS")
        print(f"{'='*80}\n")
        
        results = {
            'successful': [],
            'failed': [],
            'summary': {}
        }
        
        for i, ticker in enumerate(tickers, 1):
            print(f"[{i:3d}/{len(tickers)}] Processing {ticker:6s}...", end=' ', flush=True)
            
            try:
                # Download daily data
                df_daily = self.download_stock_data(ticker, period="2y")
                
                if df_daily is None or df_daily.empty:
                    print(f"✗ No data")
                    results['failed'].append(ticker)
                    continue
                
                # Calculate indicators
                df_daily = self.calculate_indicators(df_daily)
                
                # Save daily data
                df_daily.to_csv(os.path.join(self.data_dir, f'{ticker}_daily.csv'))
                
                # Create and save weekly data
                df_weekly = self.resample_to_weekly(df_daily)
                df_weekly = self.calculate_indicators(df_weekly)
                df_weekly.to_csv(os.path.join(self.data_dir, f'{ticker}_weekly.csv'))
                
                # Create and save monthly data
                df_monthly = self.resample_to_monthly(df_daily)
                df_monthly = self.calculate_indicators(df_monthly)
                df_monthly.to_csv(os.path.join(self.data_dir, f'{ticker}_monthly.csv'))
                
                # Create charts
                self.create_professional_chart(
                    df_daily, ticker, 'Daily',
                    os.path.join(self.charts_dir, 'daily', f'{ticker}_daily.png')
                )
                
                self.create_professional_chart(
                    df_weekly, ticker, 'Weekly',
                    os.path.join(self.charts_dir, 'weekly', f'{ticker}_weekly.png')
                )
                
                self.create_professional_chart(
                    df_monthly, ticker, 'Monthly',
                    os.path.join(self.charts_dir, 'monthly', f'{ticker}_monthly.png')
                )
                
                results['successful'].append(ticker)
                print(f"✓ Complete (D:{len(df_daily)} W:{len(df_weekly)} M:{len(df_monthly)})")
                
                # Rate limiting
                time.sleep(0.3)
                
            except Exception as e:
                print(f"✗ Error: {str(e)[:50]}")
                results['failed'].append(ticker)
        
        return results
    
    def generate_summary_report(self, tickers, results):
        """Generate a summary report of the analysis"""
        report = []
        report.append("\n" + "="*80)
        report.append("STOCK DATA ANALYSIS - SUMMARY REPORT")
        report.append("="*80)
        report.append(f"\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"\nTotal Tickers Requested: {len(tickers)}")
        report.append(f"Successfully Processed: {len(results['successful'])}")
        report.append(f"Failed: {len(results['failed'])}")
        report.append(f"Success Rate: {len(results['successful'])/len(tickers)*100:.1f}%")
        
        report.append(f"\n{'='*80}")
        report.append("OUTPUT STRUCTURE")
        report.append("="*80)
        report.append(f"\nData Directory: {self.data_dir}")
        report.append(f"  - Daily CSV files: {len(results['successful'])} files")
        report.append(f"  - Weekly CSV files: {len(results['successful'])} files")
        report.append(f"  - Monthly CSV files: {len(results['successful'])} files")
        
        report.append(f"\nCharts Directory: {self.charts_dir}")
        report.append(f"  - Daily charts: {os.path.join(self.charts_dir, 'daily')}")
        report.append(f"  - Weekly charts: {os.path.join(self.charts_dir, 'weekly')}")
        report.append(f"  - Monthly charts: {os.path.join(self.charts_dir, 'monthly')}")
        
        if results['failed']:
            report.append(f"\n{'='*80}")
            report.append("FAILED TICKERS")
            report.append("="*80)
            for ticker in results['failed']:
                report.append(f"  - {ticker}")
        
        report.append(f"\n{'='*80}")
        report.append("ANALYSIS COMPLETE")
        report.append("="*80 + "\n")
        
        report_text = '\n'.join(report)
        
        # Save report
        report_path = os.path.join(self.output_dir, 'summary_report.txt')
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        print(report_text)
        
        # Save ticker lists
        with open(os.path.join(self.data_dir, 'successful_tickers.txt'), 'w') as f:
            f.write('\n'.join(results['successful']))
        
        if results['failed']:
            with open(os.path.join(self.data_dir, 'failed_tickers.txt'), 'w') as f:
                f.write('\n'.join(results['failed']))


def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("TOP 100 STOCKS - DATA DOWNLOAD & CHART GENERATOR")
    print("="*80 + "\n")
    
    # Initialize downloader
    downloader = StockDataDownloader(output_dir='top_100_stocks_analysis')
    
    # Get top 100 tickers (with multiple fallback methods)
    tickers = downloader.get_tickers()
    
    if not tickers or len(tickers) == 0:
        print("\nERROR: Could not retrieve any tickers.")
        print("Please check your internet connection and try again.")
        return
    
    print(f"\n✓ Retrieved {len(tickers)} tickers")
    print(f"First 10 tickers: {', '.join(tickers[:10])}")
    
    # Process all tickers
    results = downloader.process_all_tickers(tickers)
    
    # Generate summary report
    downloader.generate_summary_report(tickers, results)
    
    print("\n✓ All processing complete!")
    print(f"  Check '{downloader.output_dir}' for all data and charts\n")


if __name__ == "__main__":
    main()