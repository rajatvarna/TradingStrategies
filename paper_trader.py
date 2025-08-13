import sys
import os
import json
from datetime import datetime, timedelta
import yfinance as yf

# --- Path Setup ---
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from web_app.app import app, db, Strategy, PaperAccount, PaperPosition, PaperTrade
from quant_strategies.strategy_blocks import CustomStrategy

def execute_buy(account, symbol, quantity, price):
    """Simulates executing a buy order."""
    cost = quantity * price
    if account.buying_power < cost:
        print(f"  - Insufficient buying power to buy {quantity} of {symbol}. Have {account.buying_power}, need {cost}.")
        return

    # Check for existing position to average down/up
    existing_position = PaperPosition.query.filter_by(account_id=account.id, symbol=symbol).first()
    if existing_position:
        # Update existing position
        new_total_quantity = existing_position.quantity + quantity
        new_total_cost = (existing_position.average_entry_price * existing_position.quantity) + cost
        existing_position.average_entry_price = new_total_cost / new_total_quantity
        existing_position.quantity = new_total_quantity
    else:
        # Create new position
        new_position = PaperPosition(account_id=account.id, symbol=symbol, quantity=quantity, average_entry_price=price)
        db.session.add(new_position)

    # Update account
    account.buying_power -= cost

    # Log the trade
    trade = PaperTrade(account_id=account.id, symbol=symbol, action='BUY', quantity=quantity, price=price)
    db.session.add(trade)
    print(f"  - Executed BUY of {quantity} {symbol} at ${price:.2f}")

def execute_sell(account, symbol, quantity_to_sell, price):
    """Simulates executing a sell order."""
    position = PaperPosition.query.filter_by(account_id=account.id, symbol=symbol).first()
    if not position:
        print(f"  - No position in {symbol} to sell.")
        return

    if quantity_to_sell > position.quantity:
        print(f"  - Cannot sell {quantity_to_sell} of {symbol}, only have {position.quantity}.")
        quantity_to_sell = position.quantity

    proceeds = quantity_to_sell * price

    # Update position
    position.quantity -= quantity_to_sell
    if position.quantity <= 0:
        db.session.delete(position)

    # Update account
    account.buying_power += proceeds
    account.balance += proceeds - (position.average_entry_price * quantity_to_sell) # Realize PnL

    # Log the trade
    trade = PaperTrade(account_id=account.id, symbol=symbol, action='SELL', quantity=quantity_to_sell, price=price)
    db.session.add(trade)
    print(f"  - Executed SELL of {quantity_to_sell} {symbol} at ${price:.2f}")


def run_paper_trading_cycle():
    """
    Runs one cycle of the paper trading engine.
    """
    with app.app_context():
        print(f"\n--- Running Paper Trading Cycle at {datetime.now()} ---")

        # 1. Get all deployed strategies
        deployed_strategies = Strategy.query.filter_by(is_paper_deployed=True).all()
        if not deployed_strategies:
            print("No strategies are currently deployed for paper trading. Exiting.")
            return

        # 2. Gather all unique tickers from deployed strategies
        all_tickers = set()
        for s in deployed_strategies:
            config = json.loads(s.config_json)
            all_tickers.update(config.get('tickers', []))

        if not all_tickers:
            print("No tickers found in deployed strategies. Exiting.")
            return

        # 3. Fetch latest prices for all tickers
        print(f"Fetching latest prices for: {', '.join(all_tickers)}")
        try:
            # Using '1d' to get the most recent closing price available
            data = yf.download(list(all_tickers), period='1d', progress=False)
            if data.empty:
                print("Could not fetch price data. Exiting.")
                return
            latest_prices = data['Close'].iloc[-1].to_dict()
        except Exception as e:
            print(f"Error fetching price data: {e}")
            return

        # 4. Loop through each strategy and execute trades based on signals
        for strategy in deployed_strategies:
            print(f"\nProcessing strategy: '{strategy.name}' for user '{strategy.author.username}'")
            user = strategy.author
            paper_account = user.paper_account
            if not paper_account:
                print(f"  - User '{user.username}' has no paper account. Skipping.")
                continue

            # Generate signals for the strategy
            try:
                config = json.loads(strategy.config_json)
                config['start_date'] = (datetime.now() - timedelta(days=100)).strftime('%Y-%m-%d')
                config['end_date'] = datetime.now().strftime('%Y-%m-%d')

                custom_strategy = CustomStrategy(config=config)
                signals_data = custom_strategy.generate_signals()
            except Exception as e:
                print(f"  - Error generating signals for strategy '{strategy.name}': {e}")
                continue

            # Check for trade signals
            for ticker, df in signals_data.items():
                if df.empty: continue

                last_signal = int(df.iloc[-1]['Signal'])
                current_price = latest_prices.get(ticker)
                if not current_price:
                    print(f"  - No price data for {ticker}. Skipping.")
                    continue

                position = PaperPosition.query.filter_by(account_id=paper_account.id, symbol=ticker).first()

                if last_signal == 1 and not position: # Buy signal and no position
                    # Simple position sizing: 10% of buying power
                    trade_size = paper_account.buying_power * 0.10
                    quantity = trade_size / current_price
                    execute_buy(paper_account, ticker, quantity, current_price)

                elif last_signal == -1 and position: # Sell signal and have a position
                    execute_sell(paper_account, ticker, position.quantity, current_price)

        # 5. Commit all database changes for this cycle
        db.session.commit()
        print("\n--- Paper Trading Cycle Finished ---")


if __name__ == '__main__':
    run_paper_trading_cycle()
