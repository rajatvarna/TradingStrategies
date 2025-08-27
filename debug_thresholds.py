# ADJUSTED THRESHOLDS - Replace in your main code

def analyze_single_ticker_fixed_thresholds(ticker_symbol, earnings_date=None, progress_callback=None):
    """Analyze a single ticker with REALISTIC thresholds"""
    
    # ... [keep all the existing code until the final checks] ...
    
    # At the end, replace these threshold checks:
    
    # ADJUSTED THRESHOLDS (more realistic):
    VOLUME_THRESHOLD = 500000      # Reduced from 1,500,000  
    IV_RV_THRESHOLD = 0.9          # Reduced from 1.25
    SLOPE_THRESHOLD = -0.001       # Relaxed from -0.00406
    
    # Calculate expected move
    expected_move = None
    expected_move_pct = None
    if straddle and underlying_price:
        expected_move_pct = straddle / underlying_price * 100
        expected_move = f"{expected_move_pct:.2f}%"

    # Generate trade ideas
    trade_ideas = generate_trade_ideas(
        ticker_symbol, underlying_price, options_chains, iv30_rv30, 
        ts_slope_0_45, avg_volume, straddle, list(exp_dates)
    )

    # Determine recommendation with ADJUSTED THRESHOLDS
    avg_volume_bool = avg_volume >= VOLUME_THRESHOLD      # Now 500k
    iv30_rv30_bool = iv30_rv30 >= IV_RV_THRESHOLD         # Now 0.9  
    ts_slope_bool = ts_slope_0_45 <= SLOPE_THRESHOLD      # Now -0.001

    if avg_volume_bool and iv30_rv30_bool and ts_slope_bool:
        recommendation = "RECOMMENDED"
    elif ts_slope_bool and ((avg_volume_bool and not iv30_rv30_bool) or (iv30_rv30_bool and not avg_volume_bool)):
        recommendation = "CONSIDER"
    else:
        recommendation = "AVOID"

    # Add debug printing
    print(f"DEBUG {ticker_symbol}:")
    print(f"  Volume: {avg_volume:,.0f} (need {VOLUME_THRESHOLD:,.0f}) -> {'PASS' if avg_volume_bool else 'FAIL'}")
    print(f"  IV/RV: {iv30_rv30:.3f} (need {IV_RV_THRESHOLD:.2f}) -> {'PASS' if iv30_rv30_bool else 'FAIL'}")  
    print(f"  Slope: {ts_slope_0_45:.6f} (need <= {SLOPE_THRESHOLD:.3f}) -> {'PASS' if ts_slope_bool else 'FAIL'}")
    print(f"  Final: {recommendation}")

    # Return comprehensive results
    return {
        'ticker': ticker_symbol,
        'earnings_date': earnings_date,
        'recommendation': recommendation,
        'underlying_price': underlying_price,
        'avg_volume': avg_volume,
        'avg_volume_pass': avg_volume_bool,
        'iv30': iv30,
        'rv30': rv30,
        'iv30_rv30': iv30_rv30,
        'iv30_rv30_pass': iv30_rv30_bool,
        'ts_slope_0_45': ts_slope_0_45,
        'ts_slope_pass': ts_slope_bool,
        'expected_move': expected_move,
        'expected_move_pct': expected_move_pct,
        'straddle_price': straddle,
        'trade_ideas': trade_ideas,
        'error': None
    }

# ALSO UPDATE THE TRADE IDEAS FUNCTION:
def generate_trade_ideas_fixed(ticker, underlying_price, options_chains, iv30_rv30, ts_slope_0_45, avg_volume, straddle_price, exp_dates):
    """Generate trade ideas with ADJUSTED thresholds"""
    
    trade_ideas = []
    
    # Get the first expiration data
    first_exp = exp_dates[0] if exp_dates else None
    if first_exp and first_exp in options_chains:
        chain = options_chains[first_exp]
        calls = chain.calls
        puts = chain.puts
        
        if not calls.empty and not puts.empty:
            atm_call_idx = (calls['strike'] - underlying_price).abs().idxmin()
            atm_strike = calls.loc[atm_call_idx, 'strike']
            otm_call_strike = round(underlying_price * 1.05, 0)
            otm_put_strike = round(underlying_price * 0.95, 0)
    
    # ADJUSTED THRESHOLDS IN LOGIC:
    
    # High IV + Negative Slope (ADJUSTED: IV/RV >= 0.9, slope <= -0.001)
    if iv30_rv30 >= 0.9 and ts_slope_0_45 <= -0.001:
        if len(exp_dates) >= 2:
            trade_ideas.append({
                'strategy': 'Calendar Spread (RECOMMENDED)',
                'rationale': f'IV/RV ratio {iv30_rv30:.2f} + negative slope {ts_slope_0_45:.4f} = sell expensive front, buy cheaper back',
                'setup': f'Sell {first_exp} {atm_strike} Call, Buy {exp_dates[1]} {atm_strike} Call',
                'expiration': f'{first_exp} (short) / {exp_dates[1]} (long)',
                'max_profit': 'Maximum at ATM strike at front expiration',
                'risk': 'Limited to net debit paid',
                'target': 'Stock stays near current price through front expiration',
                'stop_loss': 'Close if stock moves >7% from strike or 50% loss'
            })
        
        trade_ideas.append({
            'strategy': 'Short Straddle',
            'rationale': f'IV overpriced vs RV (ratio: {iv30_rv30:.2f}) + negative slope suggests IV crush',
            'setup': f'Sell {atm_strike} Call + Sell {atm_strike} Put',
            'expiration': first_exp,
            'max_profit': f'${straddle_price:.2f}' if straddle_price else 'N/A',
            'risk': 'Unlimited',
            'target': 'Close at 25-50% profit or before earnings',
            'stop_loss': '200% of credit received'
        })
    
    # Negative slope but lower IV/RV
    elif ts_slope_0_45 <= -0.001 and iv30_rv30 < 0.9:
        if len(exp_dates) >= 2:
            trade_ideas.append({
                'strategy': 'Calendar Spread',
                'rationale': f'Negative slope {ts_slope_0_45:.4f} favors selling front month time decay',
                'setup': f'Sell {first_exp} {atm_strike} Call/Put, Buy {exp_dates[1]} {atm_strike} Call/Put',
                'expiration': f'{first_exp} (short) / {exp_dates[1]} (long)',
                'max_profit': 'At ATM strike at front expiration',
                'risk': 'Limited to net debit paid',
                'target': 'Profit from time decay differential',
                'stop_loss': '50% of debit paid'
            })
    
    # High volume + high IV but positive slope (ADJUSTED: volume >= 500k, IV/RV >= 0.9)  
    elif avg_volume >= 500000 and iv30_rv30 >= 0.9 and ts_slope_0_45 > -0.001:
        trade_ideas.append({
            'strategy': 'Long Strangle', 
            'rationale': f'High IV ({iv30_rv30:.2f} ratio) but positive slope suggests big move expected',
            'setup': f'Buy {otm_call_strike} Call + Buy {otm_put_strike} Put',
            'expiration': first_exp,
            'max_profit': 'Unlimited',
            'risk': 'Limited to premium paid',
            'target': f'Stock moves beyond {otm_call_strike} or {otm_put_strike}',
            'stop_loss': '50% of premium paid'
        })
    
    # Low volume = avoid (ADJUSTED: < 500k)
    elif avg_volume < 500000:
        trade_ideas.append({
            'strategy': 'AVOID TRADING',
            'rationale': f'Volume {avg_volume:,.0f} too low for reliable options pricing',
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
            'rationale': f'Mixed signals: IV/RV={iv30_rv30:.2f}, Slope={ts_slope_0_45:.4f}, Vol={avg_volume:,.0f}',
            'setup': f'Small position or wait for clearer setup',
            'expiration': first_exp,
            'max_profit': 'Limited',
            'risk': 'Manage position size carefully', 
            'target': 'Quick scalp or wait for better opportunity',
            'stop_loss': '25% of premium'
        })
    
    return trade_ideas