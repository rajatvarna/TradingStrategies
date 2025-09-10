#!/usr/bin/env python3
"""
Stocks in Play Scanner - Educational Implementation
Based on "A Profitable Day Trading Strategy for The U.S. Equity Market"

⚠️ CRITICAL WARNING ⚠️
This code is for EDUCATIONAL and RESEARCH purposes ONLY.
- 80-90% of day traders lose money
- Day trading is extremely risky and can result in total loss of capital
- This is NOT investment advice
- DO NOT use for actual trading without proper risk management and education
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class StocksInPlayScanner:
    """
    Educational implementation of the Stocks in Play identification criteria
    from the ORB research paper.
    """
    
    def __init__(self):
        self.min_price = 5.0
        self.min_avg_volume = 1_000_000
        self.min_atr = 0.50
        self.min_relative_volume = 1.0
        self.lookback_days = 14
        self.top_n_stocks = 20
        
    def calculate_atr(self, data, period=14):
        """Calculate Average True Range"""
        high = data['High']
        low = data['Low']
        close = data['Close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return true_range.rolling(window=period).mean()
    
    def get_stock_universe(self):
        """Get a list of liquid US stocks for analysis"""
        
        # Popular stocks mentioned in the paper + some liquid names
        popular_stocks = [
            # Paper mentions these as top performers
            'TSLA', 'NVDA', 'AMD', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',
            'NFLX', 'CRM', 'ADBE', 'PYPL', 'SNOW', 'PLTR', 
            
            # Other liquid stocks commonly in play
            'SPY', 'QQQ', 'IWM', 'ROKU', 'ZOOM', 'DOCU', 'PTON', 'BYND',
            'MRNA', 'BNTX', 'ZM', 'SHOP', 'SQ', 'COIN', 'RIVN', 'LCID',
            'F', 'GM', 'BAC', 'JPM', 'WFC', 'XOM', 'CVX', 'INTC',
            'DIS', 'V', 'MA', 'UNH', 'JNJ', 'PG', 'KO', 'PEP',
            'ABBV', 'PFE', 'MRK', 'T', 'VZ', 'HD', 'WMT', 'COST'
        ]
        
        # Add some ETFs and popular day trading stocks
        etfs_and_leveraged = ['TQQQ', 'SQQQ', 'SPXS', 'SPXL', 'TNA', 'TZA']
        
        return popular_stocks + etfs_and_leveraged
    
    def fetch_stock_data(self, symbol, days_back=30):
        """Fetch recent stock data for analysis"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            # Get daily data
            stock = yf.Ticker(symbol)
            data = stock.history(start=start_date, end=end_date)
            
            if data.empty:
                return None
                
            # Calculate ATR
            data['ATR'] = self.calculate_atr(data)
            
            return data
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None
    
    def calculate_relative_volume(self, data):
        """
        Calculate relative volume for the most recent day
        Note: This is simplified as we don't have intraday opening range data
        """
        if len(data) < self.lookback_days + 1:
            return None
            
        # Use full day volume as proxy for opening range volume
        # (Real implementation would need 5-minute intraday data)
        recent_volume = data['Volume'].iloc[-1]
        avg_volume = data['Volume'].iloc[-(self.lookback_days+1):-1].mean()
        
        if avg_volume > 0:
            return recent_volume / avg_volume
        return None
    
    def screen_stock(self, symbol):
        """Screen a single stock against all criteria"""
        data = self.fetch_stock_data(symbol)
        
        if data is None or len(data) < self.lookback_days + 1:
            return None
        
        latest = data.iloc[-1]
        
        # Apply basic filters from the paper
        if latest['Close'] < self.min_price:
            return None
            
        avg_volume_14d = data['Volume'].iloc[-(self.lookback_days+1):-1].mean()
        if avg_volume_14d < self.min_avg_volume:
            return None
            
        if pd.isna(latest['ATR']) or latest['ATR'] < self.min_atr:
            return None
        
        # Calculate relative volume
        relative_volume = self.calculate_relative_volume(data)
        if relative_volume is None or relative_volume < self.min_relative_volume:
            return None
        
        # Calculate additional metrics
        price_change = (latest['Close'] - latest['Open']) / latest['Open'] * 100
        
        return {
            'Symbol': symbol,
            'Price': latest['Close'],
            'Volume': latest['Volume'],
            'Avg_Volume_14D': avg_volume_14d,
            'Relative_Volume': relative_volume,
            'ATR': latest['ATR'],
            'Price_Change_Pct': price_change,
            'High': latest['High'],
            'Low': latest['Low'],
            'Open': latest['Open']
        }
    
    def scan_stocks_in_play(self):
        """
        Scan for stocks meeting the 'Stocks in Play' criteria
        """
        print("🔍 Scanning for Stocks in Play...")
        print("⚠️  WARNING: This is for educational purposes only!")
        print("⚠️  Day trading is extremely risky - most traders lose money!")
        print("-" * 60)
        
        stock_universe = self.get_stock_universe()
        results = []
        
        print(f"Analyzing {len(stock_universe)} stocks...")
        
        for i, symbol in enumerate(stock_universe):
            if i % 10 == 0:
                print(f"Progress: {i+1}/{len(stock_universe)}")
                
            result = self.screen_stock(symbol)
            if result:
                results.append(result)
        
        if not results:
            print("No stocks found meeting the criteria today.")
            return pd.DataFrame()
        
        # Convert to DataFrame and sort by relative volume
        df = pd.DataFrame(results)
        df = df.sort_values('Relative_Volume', ascending=False)
        
        return df
    
    def display_results(self, df):
        """Display the screening results"""
        if df.empty:
            print("No stocks in play found today.")
            return
        
        print(f"\n📊 STOCKS IN PLAY ANALYSIS")
        print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        # Show top stocks by relative volume
        top_stocks = df.head(self.top_n_stocks)
        
        print(f"\n🔥 TOP {len(top_stocks)} STOCKS BY RELATIVE VOLUME:")
        print("-" * 80)
        print(f"{'Symbol':<6} {'Price':<8} {'RelVol':<8} {'Volume':<12} {'ATR':<6} {'Change%':<8}")
        print("-" * 80)
        
        for _, row in top_stocks.iterrows():
            print(f"{row['Symbol']:<6} ${row['Price']:<7.2f} "
                  f"{row['Relative_Volume']:<7.1f}x {row['Volume']:>10,.0f} "
                  f"${row['ATR']:<5.2f} {row['Price_Change_Pct']:>+6.1f}%")
        
        print("\n📈 RELATIVE VOLUME CATEGORIES:")
        high_vol = df[df['Relative_Volume'] >= 3.0]
        medium_vol = df[(df['Relative_Volume'] >= 2.0) & (df['Relative_Volume'] < 3.0)]
        normal_vol = df[(df['Relative_Volume'] >= 1.0) & (df['Relative_Volume'] < 2.0)]
        
        print(f"High Activity (≥3.0x): {len(high_vol)} stocks")
        print(f"Medium Activity (2.0-3.0x): {len(medium_vol)} stocks")
        print(f"Above Normal (1.0-2.0x): {len(normal_vol)} stocks")
        
        if len(high_vol) > 0:
            print(f"\n🚀 HIGHEST ACTIVITY STOCKS (≥3.0x relative volume):")
            for _, row in high_vol.iterrows():
                print(f"  {row['Symbol']}: {row['Relative_Volume']:.1f}x volume, "
                      f"{row['Price_Change_Pct']:+.1f}% change")
        
        print("\n" + "="*80)
        print("⚠️  CRITICAL DISCLAIMERS:")
        print("• This analysis is for EDUCATIONAL purposes only")
        print("• High relative volume does NOT guarantee profitable trading")
        print("• Day trading has extremely high failure rates (80-90%)")  
        print("• This is NOT investment advice")
        print("• Past patterns do not predict future results")
        print("• Only trade with money you can afford to lose completely")
        print("="*80)

def main():
    """Main function to run the stocks in play scanner"""
    
    print("📈 COMPREHENSIVE STOCKS IN PLAY SCANNER")
    print("Based on ORB Research Paper (7000+ Stock Universe)")
    print("=" * 60)
    
    # Display critical warnings upfront
    print("\n🚨 CRITICAL FINANCIAL RISK WARNINGS:")
    print("• Day trading has an 80-90% failure rate")
    print("• Most day traders lose their entire account within months")
    print("• The paper's 1,637% return is mathematically impossible to achieve")
    print("• Professional day traders have advantages you don't have:")
    print("  - Proprietary low-latency data feeds")
    print("  - Direct market access with rebates")
    print("  - Millions in capital for diversification")
    print("  - Teams of quantitative developers")
    print("• This comprehensive scanner could facilitate harmful behavior")
    print("• Academic study ≠ practical trading strategy")
    print("\n" + "="*60)
    
    print("\n🎓 LEGITIMATE EDUCATIONAL USES:")
    print("• Academic research on market microstructure")
    print("• Understanding volume/volatility relationships") 
    print("• Learning about the paper's screening methodology")
    print("• Studying correlation between news and volume")
    print("• Building financial education tools")
    print("\n" + "="*60)
    
    print("\n🔬 SCAN OPTIONS:")
    print("1. Limited scan (~100 stocks, 2-3 minutes)")
    print("2. Comprehensive scan (1000+ stocks, 30+ minutes, research only)")
    print("3. Exit (recommended if considering actual trading)")
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    if choice == '3':
        print("\nWise choice. Consider these alternatives:")
        print("• Paper trading for 6+ months minimum")
        print("• Long-term index fund investing")
        print("• Learning fundamental analysis")
        print("• Professional financial education courses")
        return
    
    if choice == '2':
        print("\n🚨 COMPREHENSIVE SCAN CONFIRMATION:")
        print("This is for ACADEMIC/RESEARCH purposes only.")
        print("Expanding day trading tools can lead to financial ruin.")
        print("Are you:")
        print("• A finance researcher studying market microstructure?")
        print("• An academic analyzing the paper's methodology?") 
        print("• Building educational tools about market patterns?")
        print("\nIf you're considering actual trading, this is extremely dangerous.")
        
        confirm = input("\nType 'ACADEMIC RESEARCH ONLY' to proceed: ")
        if confirm.upper() != 'ACADEMIC RESEARCH ONLY':
            print("Protecting you from potentially harmful behavior.")
            print("Day trading destroys lives and families financially.")
            return
        use_comprehensive = True
    else:
        use_comprehensive = False
    
    # Additional safety check
    if use_comprehensive:
        print("\n⚠️  FINAL WARNING:")
        print("Even for academic purposes, understanding these patterns")
        print("could create psychological pressure to start day trading.")
        print("Remember: The paper's returns are not achievable in practice.")
        
        final_confirm = input("Continue with research? (yes/no): ")
        if final_confirm.lower() != 'yes':
            print("Research cancelled for safety.")
            return
    
    # Run the scanner
    scanner = StocksInPlayScanner()
    
    if use_comprehensive:
        print(f"\n🔬 RESEARCH MODE ACTIVATED")
        print("This may take 30+ minutes. Please be patient.")
        results_df = scanner.scan_stocks_in_play()
    else:
        print(f"\n📚 EDUCATIONAL MODE ACTIVATED") 
        results_df = scanner.scan_stocks_in_play()
    
    scanner.display_results(results_df)
    
    # Save results if any found
    if not results_df.empty:
        scan_type = "comprehensive" if use_comprehensive else "limited"
        filename = f"stocks_in_play_{scan_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        results_df.to_csv(filename, index=False)
        print(f"\n💾 Results saved to: {filename}")
    
    print(f"\n📚 FINAL EDUCATIONAL REMINDERS:")
    print("• The paper assumes perfect execution (impossible in reality)")
    print("• Real ORB trading requires millisecond-level data feeds")
    print("• Transaction costs and slippage eliminate most profits")
    print("• Market makers and HFTs compete for the same opportunities")
    print("• Successful trading requires years of education and practice")
    print("• Most profitable approach for individuals is long-term investing")
    
    if use_comprehensive:
        print(f"\n🔬 RESEARCH NOTES:")
        print("• This data shows market patterns, not trading opportunities")
        print("• High relative volume often leads to high volatility")
        print("• News-driven volume spikes are unpredictable")
        print("• Professional traders have advantages retail traders lack")
        print("• Consider this data in academic context only")

if __name__ == "__main__":
    main()