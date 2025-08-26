"""
# Earnings Volatility Calculator with Trade Ideas
### **1. Specific Trade Ideas Generation**
- **Short Straddle/Iron Condor**: When IV is overpriced vs realized volatility
- **Calendar Spreads**: When term structure favors selling front month
- **Long Strangle**: When expecting large moves
- **Conservative/Avoid**: When conditions aren't optimal

### **2. Detailed Strategy Information**
- **Setup**: Specific strikes and expirations
- **Rationale**: Why this strategy fits the analysis
- **Max Profit/Risk**: Clear risk/reward metrics
- **Entry/Exit Targets**: Specific profit-taking levels
- **Stop Loss**: Risk management guidelines

### **3. Enhanced GUI**
- **"View Trade Ideas" button** in results window
- **Scrollable popup** showing all strategies
- **Color-coded information** for easy reading

## **Example Trade Ideas Output:**

**For High IV + Negative Slope (RECOMMENDED):**
```
Strategy 1: Short Straddle
Rationale: High IV/RV ratio + negative term structure suggests IV crush opportunity
Setup: Sell 180 Call + Sell 180 Put
Max Profit: $8.50 (straddle premium)
Target: Close at 25-50% profit or before earnings
Stop Loss: 200% of credit received
```

**For Mixed Signals:**
```
Strategy 1: Calendar Spread  
Setup: Sell Sept 180 Call, Buy Oct 180 Call
Rationale: Profit from time decay differential
Target: At ATM strike at front expiration
```

## **Key Benefits:**
- **Actionable trades** instead of just analysis
- **Specific strikes and dates** 
- **Risk management built-in**
- **Multiple strategy options** based on market conditions
- **Educational rationale** for each trade

The trade ideas are **dynamically generated** based on your analysis criteria and provide **real trading strategies** that professionals use for earnings plays. Each recommendation includes the "why" behind the trade and specific execution details.
"""

import FreeSimpleGUI as sg
import yfinance as yf
from datetime import datetime, timedelta
from scipy.interpolate import interp1d
import numpy as np
import pandas as pd
import threading
import time

# Add retry logic for yfinance calls
def safe_yfinance_call(func, max_retries=3, delay=2):
    """Wrapper for yfinance calls with retry logic"""
    for attempt in range(max_retries):
        try:
            result = func()
            return result
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                raise e

def get_current_price(ticker):
    try:
        def get_price():
            todays_data = ticker.history(period='1d')
            if todays_data.empty:
                todays_data = ticker.history(period='5d')
                if todays_data.empty:
                    raise ValueError("No price data available")
            return todays_data['Close'].iloc[-1]
        
        return safe_yfinance_call(get_price)
    except Exception as e:
        print(f"Error getting current price: {e}")
        return None

def filter_dates(dates):
    today = datetime.today().date()
    cutoff_date = today + timedelta(days=45)
    
    sorted_dates = sorted(datetime.strptime(date, "%Y-%m-%d").date() for date in dates)

    arr = []
    for i, date in enumerate(sorted_dates):
        if date >= cutoff_date:
            arr = [d.strftime("%Y-%m-%d") for d in sorted_dates[:i+1]]  
            break
    
    if len(arr) > 0:
        if arr[0] == today.strftime("%Y-%m-%d"):
            return arr[1:]
        return arr

    raise ValueError("No date 45 days or more in the future found.")

def yang_zhang(price_data, window=30, trading_periods=252, return_last_only=True):
    log_ho = (price_data['High'] / price_data['Open']).apply(np.log)
    log_lo = (price_data['Low'] / price_data['Open']).apply(np.log)
    log_co = (price_data['Close'] / price_data['Open']).apply(np.log)
    
    log_oc = (price_data['Open'] / price_data['Close'].shift(1)).apply(np.log)
    log_oc_sq = log_oc**2
    
    log_cc = (price_data['Close'] / price_data['Close'].shift(1)).apply(np.log)
    log_cc_sq = log_cc**2
    
    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
    
    close_vol = log_cc_sq.rolling(
        window=window,
        center=False
    ).sum() * (1.0 / (window - 1.0))

    open_vol = log_oc_sq.rolling(
        window=window,
        center=False
    ).sum() * (1.0 / (window - 1.0))

    window_rs = rs.rolling(
        window=window,
        center=False
    ).sum() * (1.0 / (window - 1.0))

    k = 0.34 / (1.34 + ((window + 1) / (window - 1)) )
    result = (open_vol + k * close_vol + (1 - k) * window_rs).apply(np.sqrt) * np.sqrt(trading_periods)

    if return_last_only:
        return result.iloc[-1]
    else:
        return result.dropna()

def build_term_structure(days, ivs):
    days = np.array(days)
    ivs = np.array(ivs)

    sort_idx = days.argsort()
    days = days[sort_idx]
    ivs = ivs[sort_idx]

    spline = interp1d(days, ivs, kind='linear', fill_value="extrapolate")

    def term_spline(dte):
        if dte < days[0]:  
            return ivs[0]
        elif dte > days[-1]:
            return ivs[-1]
        else:  
            return float(spline(dte))

    return term_spline

def generate_trade_ideas(ticker, underlying_price, options_chains, iv30_rv30, ts_slope_0_45, avg_volume, straddle_price, exp_dates):
    """Generate specific trade ideas based on the analysis"""
    
    trade_ideas = []
    
    # Get the first expiration data for specific recommendations
    first_exp = exp_dates[0] if exp_dates else None
    if first_exp and first_exp in options_chains:
        chain = options_chains[first_exp]
        calls = chain.calls
        puts = chain.puts
        
        if not calls.empty and not puts.empty:
            # Find ATM strikes
            atm_call_idx = (calls['strike'] - underlying_price).abs().idxmin()
            atm_put_idx = (puts['strike'] - underlying_price).abs().idxmin()
            atm_strike = calls.loc[atm_call_idx, 'strike']
            
            # Find strikes for different strategies
            otm_call_strike = round(underlying_price * 1.05, 0)  # 5% OTM
            otm_put_strike = round(underlying_price * 0.95, 0)   # 5% OTM
            
            # Get prices for specific options
            atm_call_price = (calls.loc[atm_call_idx, 'bid'] + calls.loc[atm_call_idx, 'ask']) / 2 if not pd.isna(calls.loc[atm_call_idx, 'bid']) else None
            atm_put_price = (puts.loc[atm_put_idx, 'bid'] + puts.loc[atm_put_idx, 'ask']) / 2 if not pd.isna(puts.loc[atm_put_idx, 'bid']) else None
    
    # High IV + Negative Slope = Short Volatility Strategies
    if iv30_rv30 >= 1.25 and ts_slope_0_45 <= -0.00406:
        # Calendar spread as primary strategy
        if len(exp_dates) >= 2:
            trade_ideas.append({
                'strategy': 'Calendar Spread (RECOMMENDED)',
                'rationale': 'High front month IV + negative slope = sell expensive front, buy cheaper back month',
                'setup': f'Sell {first_exp} {atm_strike} Call, Buy {exp_dates[1]} {atm_strike} Call (or Put version)',
                'expiration': f'{first_exp} (short) / {exp_dates[1]} (long)',
                'max_profit': 'Maximum at ATM strike at front expiration',
                'risk': 'Limited to net debit paid',
                'target': 'Stock stays near current price through front expiration',
                'stop_loss': 'Close if stock moves >7% from strike or 50% loss'
            })
        
        trade_ideas.append({
            'strategy': 'Short Straddle',
            'rationale': 'High IV/RV ratio + negative term structure suggests IV crush opportunity',
            'setup': f'Sell {atm_strike} Call + Sell {atm_strike} Put',
            'expiration': first_exp,
            'max_profit': f'${straddle_price:.2f}' if straddle_price else 'N/A',
            'risk': 'Unlimited',
            'target': 'Close at 25-50% profit or before earnings',
            'stop_loss': '200% of credit received'
        })
        
        trade_ideas.append({
            'strategy': 'Iron Condor',
            'rationale': 'High IV with range-bound expectation',
            'setup': f'Sell {otm_put_strike} Put, Buy {otm_put_strike-5} Put, Sell {otm_call_strike} Call, Buy {otm_call_strike+5} Call',
            'expiration': first_exp,
            'max_profit': 'Net credit received',
            'risk': 'Limited to spread width minus credit',
            'target': 'Stock stays between short strikes',
            'stop_loss': '50% of max loss'
        })
    
    # Negative term structure but lower IV/RV = Calendar spreads
    elif ts_slope_0_45 <= -0.00406 and iv30_rv30 < 1.25:
        if len(exp_dates) >= 2:
            trade_ideas.append({
                'strategy': 'Calendar Spread',
                'rationale': 'Negative term structure - sell front month, buy back month',
                'setup': f'Sell {first_exp} {atm_strike} Call/Put, Buy {exp_dates[1]} {atm_strike} Call/Put',
                'expiration': f'{first_exp} (short) / {exp_dates[1]} (long)',
                'max_profit': 'At ATM strike at front expiration',
                'risk': 'Limited to net debit paid',
                'target': 'Profit from time decay differential',
                'stop_loss': '50% of debit paid'
            })
    
    # High volume + high IV but positive slope = Directional plays
    elif avg_volume >= 1500000 and iv30_rv30 >= 1.25 and ts_slope_0_45 > -0.00406:
        trade_ideas.append({
            'strategy': 'Long Strangle',
            'rationale': 'High IV but positive slope suggests big move expected',
            'setup': f'Buy {otm_call_strike} Call + Buy {otm_put_strike} Put',
            'expiration': first_exp,
            'max_profit': 'Unlimited',
            'risk': 'Limited to premium paid',
            'target': f'Stock moves beyond {otm_call_strike} or {otm_put_strike}',
            'stop_loss': '50% of premium paid'
        })
    
    # Low volume = avoid
    elif avg_volume < 1500000:
        trade_ideas.append({
            'strategy': 'AVOID TRADING',
            'rationale': 'Insufficient volume for reliable options pricing',
            'setup': 'Wait for higher volume or choose different underlying',
            'expiration': 'N/A',
            'max_profit': 'N/A',
            'risk': 'High slippage and poor fills',
            'target': 'Find more liquid alternatives',
            'stop_loss': 'N/A'
        })
    
    # Default conservative approach
    else:
        trade_ideas.append({
            'strategy': 'Conservative Approach',
            'rationale': 'Mixed signals - consider paper trading or smaller position',
            'setup': f'Small position in {atm_strike} straddle or wait for clearer setup',
            'expiration': first_exp,
            'max_profit': 'Limited',
            'risk': 'Manage position size carefully',
            'target': 'Quick scalp or wait for better opportunity',
            'stop_loss': '25% of premium'
        })
    
    return trade_ideas

def compute_recommendation(ticker):
    try:
        ticker = ticker.strip().upper()
        if not ticker:
            return {"error": "No stock symbol provided."}
        
        print(f"Processing ticker: {ticker}")
        
        # Test basic ticker validity first
        try:
            def get_stock_info():
                stock = yf.Ticker(ticker)
                info = stock.info
                if not info or 'symbol' not in info:
                    recent_data = stock.history(period='5d')
                    if recent_data.empty:
                        raise KeyError(f"No data found for ticker {ticker}")
                return stock
            
            stock = safe_yfinance_call(get_stock_info)
            print("✓ Ticker validated successfully")
            
        except Exception as e:
            print(f"Error validating ticker: {e}")
            return {"error": f"Invalid ticker symbol '{ticker}' or no data available"}
        
        # Get options data
        try:
            def get_options():
                options_list = stock.options
                if len(options_list) == 0:
                    raise KeyError("No options data")
                return list(options_list)
            
            exp_dates = safe_yfinance_call(get_options)
            print(f"Found {len(exp_dates)} option expiration dates")
            
        except Exception as e:
            print(f"Error getting options data: {e}")
            return {"error": f"No options found for ticker '{ticker}'"}
        
        # Filter dates
        try:
            exp_dates = filter_dates(exp_dates)
            print(f"Filtered to {len(exp_dates)} dates: {exp_dates}")
        except Exception as e:
            print(f"Error filtering dates: {e}")
            return {"error": "Not enough future option expiration dates (need dates 45+ days out)"}
        
        # Get current price
        print("Getting current stock price...")
        underlying_price = get_current_price(stock)
        if underlying_price is None:
            return {"error": "Unable to retrieve current stock price"}
        print(f"Current price: ${underlying_price:.2f}")
        
        # Get options chains
        print("Getting options chains...")
        options_chains = {}
        for i, exp_date in enumerate(exp_dates):
            try:
                def get_chain():
                    return stock.option_chain(exp_date)
                
                chain = safe_yfinance_call(get_chain)
                options_chains[exp_date] = chain
                print(f"✓ Got options chain for {exp_date} ({i+1}/{len(exp_dates)})")
                
            except Exception as e:
                print(f"✗ Error getting options chain for {exp_date}: {e}")
                continue
        
        if not options_chains:
            return {"error": "Could not retrieve any options chains"}
        
        # Calculate ATM IVs
        print("Calculating ATM implied volatilities...")
        atm_iv = {}
        straddle = None 
        i = 0
        for exp_date, chain in options_chains.items():
            try:
                calls = chain.calls
                puts = chain.puts

                if calls.empty or puts.empty:
                    print(f"Empty chain for {exp_date}, skipping")
                    continue

                # Find ATM call and put IVs
                call_diffs = (calls['strike'] - underlying_price).abs()
                call_idx = call_diffs.idxmin()
                
                put_diffs = (puts['strike'] - underlying_price).abs()
                put_idx = put_diffs.idxmin()
                
                call_iv = calls.loc[call_idx, 'impliedVolatility']
                put_iv = puts.loc[put_idx, 'impliedVolatility']
                
                # Check for valid IV values
                if pd.isna(call_iv) or pd.isna(put_iv) or call_iv <= 0 or put_iv <= 0:
                    print(f"Invalid IV data for {exp_date}, skipping")
                    continue

                atm_iv_value = (call_iv + put_iv) / 2.0
                atm_iv[exp_date] = atm_iv_value
                print(f"ATM IV for {exp_date}: {atm_iv_value:.3f}")

                # Calculate straddle price for first expiration
                if i == 0:
                    call_bid = calls.loc[call_idx, 'bid']
                    call_ask = calls.loc[call_idx, 'ask']
                    put_bid = puts.loc[put_idx, 'bid']
                    put_ask = puts.loc[put_idx, 'ask']
                    
                    call_mid = None
                    put_mid = None
                    
                    if (call_bid is not None and call_ask is not None and 
                        not pd.isna(call_bid) and not pd.isna(call_ask) and 
                        call_bid > 0 and call_ask > 0):
                        call_mid = (call_bid + call_ask) / 2.0

                    if (put_bid is not None and put_ask is not None and 
                        not pd.isna(put_bid) and not pd.isna(put_ask) and 
                        put_bid > 0 and put_ask > 0):
                        put_mid = (put_bid + put_ask) / 2.0

                    if call_mid is not None and put_mid is not None:
                        straddle = call_mid + put_mid
                        print(f"Straddle price: ${straddle:.2f}")

                i += 1
            except Exception as e:
                print(f"Error processing {exp_date}: {e}")
                continue
        
        if not atm_iv:
            return {"error": "Could not determine ATM IV for any expiration dates"}
        
        # Build term structure
        print("Building term structure...")
        today = datetime.today().date()
        dtes = []
        ivs = []
        for exp_date, iv in atm_iv.items():
            exp_date_obj = datetime.strptime(exp_date, "%Y-%m-%d").date()
            days_to_expiry = (exp_date_obj - today).days
            dtes.append(days_to_expiry)
            ivs.append(iv)
        
        print(f"DTEs: {dtes}")
        print(f"IVs: {ivs}")
        
        if len(dtes) < 2:
            return {"error": "Need at least 2 expiration dates for term structure"}
        
        try:
            term_spline = build_term_structure(dtes, ivs)
            ts_slope_0_45 = (term_spline(45) - term_spline(dtes[0])) / (45 - dtes[0])
            print(f"Term structure slope: {ts_slope_0_45:.6f}")
        except Exception as e:
            print(f"Error building term structure: {e}")
            return {"error": f"Failed to build term structure: {str(e)}"}
        
        # Get price history and calculate realized volatility
        print("Getting price history and calculating realized volatility...")
        try:
            def get_history():
                price_history = stock.history(period='3mo')
                if price_history.empty:
                    price_history = stock.history(period='2mo')
                    if price_history.empty:
                        raise ValueError("No price history available")
                return price_history
                
            price_history = safe_yfinance_call(get_history)
            print(f"Price history shape: {price_history.shape}")
            
            if len(price_history) < 30:
                return {"error": "Insufficient price history for calculations (need 30+ days)"}
            
            # Calculate realized volatility
            rv30 = yang_zhang(price_history)
            if pd.isna(rv30) or rv30 <= 0:
                return {"error": "Could not calculate realized volatility"}
                
            iv30 = term_spline(30)
            iv30_rv30 = iv30 / rv30
            print(f"IV30: {iv30:.3f}, RV30: {rv30:.3f}, IV/RV ratio: {iv30_rv30:.3f}")

            # Calculate average volume
            volume_data = price_history['Volume'].rolling(30).mean().dropna()
            if volume_data.empty:
                return {"error": "No volume data available"}
            avg_volume = volume_data.iloc[-1]
            print(f"Average volume: {avg_volume:,.0f}")

        except Exception as e:
            print(f"Error with price history/volatility calculation: {e}")
            return {"error": f"Failed to calculate historical metrics: {str(e)}"}

        # Calculate expected move
        expected_move = None
        if straddle and underlying_price:
            expected_move = str(round(straddle / underlying_price * 100, 2)) + "%"

        # Generate trade ideas
        trade_ideas = generate_trade_ideas(
            ticker, underlying_price, options_chains, iv30_rv30, 
            ts_slope_0_45, avg_volume, straddle, exp_dates
        )

        # Return results dictionary
        result = {
            'avg_volume': avg_volume >= 1500000, 
            'iv30_rv30': iv30_rv30 >= 1.25, 
            'ts_slope_0_45': ts_slope_0_45 <= -0.00406, 
            'expected_move': expected_move,
            'trade_ideas': trade_ideas,
            'underlying_price': underlying_price,
            'error': None
        }
        
        print(f"Final result: {result}")
        return result
        
    except Exception as e:
        print(f"UNEXPECTED ERROR: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": f"Unexpected error: {type(e).__name__}: {str(e)}"}

def show_trade_ideas_window(trade_ideas, ticker, underlying_price):
    """Show detailed trade ideas in a separate window"""
    
    if not trade_ideas:
        sg.popup("No trade ideas generated", title="Trade Ideas")
        return
    
    # Create layout for trade ideas
    layout = [
        [sg.Text(f"Trade Ideas for {ticker} @ ${underlying_price:.2f}", 
                font=("Helvetica", 14, "bold"), justification="center")],
        [sg.Text("_" * 80)],
    ]
    
    for i, idea in enumerate(trade_ideas):
        layout.extend([
            [sg.Text(f"Strategy {i+1}: {idea['strategy']}", 
                    font=("Helvetica", 12, "bold"), text_color="blue")],
            [sg.Text(f"Rationale: {idea['rationale']}", size=(80, None), 
                    font=("Helvetica", 9))],
            [sg.Text(f"Setup: {idea['setup']}", text_color="darkgreen")],
            [sg.Text(f"Expiration: {idea['expiration']}")],
            [sg.Text(f"Max Profit: {idea['max_profit']}")],
            [sg.Text(f"Risk: {idea['risk']}")],
            [sg.Text(f"Target: {idea['target']}", text_color="navy")],
            [sg.Text(f"Stop Loss: {idea['stop_loss']}", text_color="darkred")],
            [sg.Text("_" * 80)],
        ])
    
    layout.append([sg.Button("Close", size=(10, 1))])
    
    # Create scrollable window
    window = sg.Window(
        f"Trade Ideas - {ticker}", 
        [[sg.Column(layout, scrollable=True, vertical_scroll_only=True, size=(700, 500))]],
        modal=True, finalize=True, resizable=True
    )
    
    while True:
        event, _ = window.read()
        if event in (sg.WINDOW_CLOSED, "Close"):
            break
    
    window.close()

def main_gui():
    main_layout = [
        [sg.Text("Enter Stock Symbol:"), sg.Input(key="stock", size=(20, 1), focus=True)],
        [sg.Button("Submit", bind_return_key=True), sg.Button("Exit")],
        [sg.Text("", key="recommendation", size=(50, 1))]
    ]
    
    window = sg.Window("Earnings Position Checker with Trade Ideas", main_layout)
    
    while True:
        event, values = window.read()
        if event in (sg.WINDOW_CLOSED, "Exit"):
            break

        if event == "Submit":
            window["recommendation"].update("")
            stock = values.get("stock", "")

            # Show loading window while processing
            loading_layout = [[sg.Text("Loading...", key="loading", justification="center")]]
            loading_window = sg.Window("Loading", loading_layout, modal=True, finalize=True, size=(275, 200))

            result_holder = {}

            def worker():
                try:
                    result = compute_recommendation(stock)
                    result_holder['result'] = result
                except Exception as e:
                    result_holder['error'] = str(e)

            thread = threading.Thread(target=worker, daemon=True)
            thread.start()

            while thread.is_alive():
                event_load, _ = loading_window.read(timeout=100)
                if event_load == sg.WINDOW_CLOSED:
                    break
            thread.join(timeout=1)

            loading_window.close()

            # Handle results with proper error checking
            if 'error' in result_holder:
                window["recommendation"].update(f"Error: {result_holder['error']}")
            elif 'result' in result_holder:
                result = result_holder['result']
                
                # Check if result is an error dictionary
                if isinstance(result, dict) and result.get('error'):
                    window["recommendation"].update(f"Error: {result['error']}")
                elif isinstance(result, dict) and 'avg_volume' in result:
                    # Valid result dictionary
                    avg_volume_bool = result['avg_volume']
                    iv30_rv30_bool = result['iv30_rv30']
                    ts_slope_bool = result['ts_slope_0_45']
                    expected_move = result['expected_move']
                    trade_ideas = result.get('trade_ideas', [])
                    underlying_price = result.get('underlying_price', 0)
                    
                    # Determine recommendation
                    if avg_volume_bool and iv30_rv30_bool and ts_slope_bool:
                        title = "Recommended"
                        title_color = "#006600"
                    elif ts_slope_bool and ((avg_volume_bool and not iv30_rv30_bool) or (iv30_rv30_bool and not avg_volume_bool)):
                        title = "Consider"
                        title_color = "#ff9900"
                    else:
                        title = "Avoid"
                        title_color = "#800000"
                    
                    # Show result window with trade ideas button
                    result_layout = [
                        [sg.Text(title, text_color=title_color, font=("Helvetica", 16))],
                        [sg.Text(f"avg_volume: {'PASS' if avg_volume_bool else 'FAIL'}", 
                                text_color="#006600" if avg_volume_bool else "#800000")],
                        [sg.Text(f"iv30_rv30: {'PASS' if iv30_rv30_bool else 'FAIL'}", 
                                text_color="#006600" if iv30_rv30_bool else "#800000")],
                        [sg.Text(f"ts_slope_0_45: {'PASS' if ts_slope_bool else 'FAIL'}", 
                                text_color="#006600" if ts_slope_bool else "#800000")],
                        [sg.Text(f"Expected Move: {expected_move or 'N/A'}", text_color="blue")],
                        [sg.Text("_" * 30)],
                        [sg.Button("View Trade Ideas", size=(15, 1)), sg.Button("OK")]
                    ]
                    
                    result_window = sg.Window("Recommendation", result_layout, modal=True, finalize=True)
                    while True:
                        event_result, _ = result_window.read()
                        if event_result == "View Trade Ideas":
                            show_trade_ideas_window(trade_ideas, stock.upper(), underlying_price)
                        elif event_result in (sg.WINDOW_CLOSED, "OK"):
                            break
                    result_window.close()
                else:
                    window["recommendation"].update(f"Error: Unexpected result format")
            else:
                window["recommendation"].update("Error: No result returned")
    
    window.close()

def gui():
    main_gui()

if __name__ == "__main__":
    gui()