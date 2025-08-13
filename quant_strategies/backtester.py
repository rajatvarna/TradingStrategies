import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import yfinance as yf

from quant_strategies.strategy_base import Strategy
from quant_strategies.order import Order


class Backtester:
    """
    A portfolio-level backtester for quantitative trading strategies.

    This class can run two types of strategies:
    1. Signal-based: The strategy generates a DataFrame with a 'Signal' column.
    2. Order-based: The strategy generates a list of Order objects for each day.

    The backtester handles portfolio management, position sizing (for signal-based),
    and performance calculation.
    """

    def __init__(self, strategy: Strategy, initial_capital=100000, max_positions=15,
                 trade_risk_pct=0.01, margin=1.5):
        """
        Initializes the Backtester.
        """
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        self.trade_risk_pct = trade_risk_pct
        self.margin = margin
        self.equity_curve = None
        self.trade_log = None
        self.is_order_based = self.strategy.strategy_type == 'order'

    def run(self):
        """
        Runs the backtest by dispatching to the appropriate handler.
        """
        print(f"Running backtest in {'Order-based' if self.is_order_based else 'Signal-based'} mode.")
        if self.is_order_based:
            return self._run_order_based_backtest()
        else:
            return self._run_signal_based_backtest()

    def _run_signal_based_backtest(self):
        """
        Runs the simulation for signal-based strategies.
        This is the original backtesting logic.
        """
        # 1. Get signals and data from the strategy
        signals = self.strategy.generate_signals()

        # 2. Consolidate data
        all_data_frames = [df.add_prefix(f'{ticker}_') for ticker, df in signals.items() if not df.empty]
        if not all_data_frames:
            print("No signal data available. Exiting backtest.")
            return None, None

        portfolio_data = pd.concat(all_data_frames, axis=1)
        portfolio_data.sort_index(inplace=True)
        portfolio_data.ffill(inplace=True)
        portfolio_data.fillna(0, inplace=True)

        # 3. Initialize portfolio
        portfolio_value = [self.initial_capital]
        portfolio_dates = [portfolio_data.index[0]]
        open_positions = {}
        self.trade_log = []

        # 4. Main simulation loop
        for i in range(1, len(portfolio_data)):
            current_date = portfolio_data.index[i]
            prev_date = portfolio_data.index[i-1]
            current_day_data = portfolio_data.iloc[i]

            # Update portfolio value
            current_capital = portfolio_value[-1]
            pnl = sum((current_day_data[f'{t}_Close'] - portfolio_data.loc[prev_date, f'{t}_Close']) * p['quantity']
                      for t, p in open_positions.items() if portfolio_data.loc[prev_date, f'{t}_Close'] > 0)
            current_capital += pnl

            # Process exits
            exited_tickers = []
            for ticker, pos in list(open_positions.items()):
                if current_day_data[f'{ticker}_Signal'] == -1:
                    exit_price = current_day_data[f'{ticker}_Close']
                    self._log_trade(ticker, pos['entry_date'], pos['entry_price'], current_date, exit_price)
                    exited_tickers.append(ticker)
            for ticker in exited_tickers:
                del open_positions[ticker]

            # Process entries
            num_slots = self.max_positions - len(open_positions)
            if num_slots > 0:
                candidates = [t for t in self.strategy.tickers if current_day_data[f'{t}_Signal'] == 1 and t not in open_positions]
                for ticker in candidates[:num_slots]:
                    entry_price = current_day_data[f'{ticker}_Close']
                    if entry_price > 0:
                        amount_to_invest = current_capital * self.trade_risk_pct * self.margin
                        quantity = amount_to_invest / entry_price
                        open_positions[ticker] = {'entry_date': current_date, 'entry_price': entry_price, 'quantity': quantity}

            portfolio_value.append(current_capital)
            portfolio_dates.append(current_date)

        # 5. Finalize
        self._close_open_positions(portfolio_data.index[-1], portfolio_data.iloc[-1], open_positions)
        self.equity_curve = pd.Series(portfolio_value, index=portfolio_dates)
        self.trade_log = pd.DataFrame(self.trade_log)
        print("Backtest finished.")
        return self.equity_curve, self.trade_log

    def _run_order_based_backtest(self):
        """
        Runs the simulation for order-based strategies.
        This is a more event-driven style of backtesting.
        """
        # 1. Backtester downloads its own daily data for accounting
        print("Backtester downloading daily data for portfolio accounting...")
        daily_data = yf.download(self.strategy.tickers, start=self.strategy.start_date, end=self.strategy.end_date)

        if daily_data.empty:
            print("Could not download daily data for accounting. Exiting.")
            return None, None

        # 2. Initialize portfolio
        portfolio_value = [self.initial_capital]
        portfolio_dates = [daily_data.index[0]]
        open_positions = {} # {ticker: {'entry_date':, 'entry_price':, 'quantity':}}
        self.trade_log = []

        # 3. Main simulation loop
        for i in range(1, len(daily_data)):
            current_date = daily_data.index[i]
            prev_date = daily_data.index[i-1]

            # Update portfolio value
            current_capital = portfolio_value[-1]
            pnl = sum((daily_data.loc[current_date, ('Close', t)] - daily_data.loc[prev_date, ('Close', t)]) * p['quantity']
                      for t, p in open_positions.items() if daily_data.loc[prev_date, ('Close', t)] > 0)
            current_capital += pnl

            # Get and execute orders from the strategy
            portfolio_state = {'date': current_date, 'capital': current_capital, 'positions': open_positions, 'daily_data': daily_data}
            orders = self.strategy.generate_orders(portfolio_state)

            for order in orders:
                price = daily_data.loc[current_date, ('Close', order.symbol)]
                if pd.isna(price) or price <= 0: continue

                if order.action in ['BUY', 'SELL_SHORT'] and len(open_positions) < self.max_positions:
                    open_positions[order.symbol] = {'entry_date': current_date, 'entry_price': price, 'quantity': order.quantity}
                elif order.action in ['SELL', 'COVER'] and order.symbol in open_positions:
                    pos = open_positions.pop(order.symbol)
                    self._log_trade(order.symbol, pos['entry_date'], pos['entry_price'], current_date, price)

            portfolio_value.append(current_capital)
            portfolio_dates.append(current_date)

        # 4. Finalize
        last_date = daily_data.index[-1]
        last_prices = daily_data.loc[last_date]
        for ticker, pos in open_positions.items():
            last_price = last_prices[('Close', ticker)]
            self._log_trade(ticker, pos['entry_date'], pos['entry_price'], last_date, last_price, status='Open')

        self.equity_curve = pd.Series(portfolio_value, index=portfolio_dates)
        self.trade_log = pd.DataFrame(self.trade_log)
        print("Backtest finished.")
        return self.equity_curve, self.trade_log

    def _log_trade(self, ticker, buy_date, buy_price, sell_date, sell_price, status='Closed'):
        trade_return = (sell_price - buy_price) / buy_price if buy_price > 0 else 0
        self.trade_log.append({
            'Ticker': ticker, 'Buy Date': buy_date, 'Buy Price': buy_price,
            'Sell Date': sell_date, 'Sell Price': sell_price, 'Return': trade_return, 'Status': status
        })

    def _close_open_positions(self, last_date, last_day_data, open_positions):
        for ticker, pos_info in open_positions.items():
            last_price = last_day_data[f'{ticker}_Close']
            self._log_trade(ticker, pos_info['entry_date'], pos_info['entry_price'], last_date, last_price, status='Open')

    def calculate_statistics(self, benchmark_ticker='^GSPC'):
        if self.equity_curve is None or self.equity_curve.empty:
            print("Cannot calculate statistics: Equity curve is empty.")
            return {metric: 0 for metric in ['CAGR (%)', 'Volatility (%)', 'Sharpe Ratio', 'Max Drawdown (%)', 'Alpha', 'Beta']}

        print("Calculating performance statistics...")
        metrics = {}
        returns = self.equity_curve.pct_change().dropna()

        years = (self.equity_curve.index[-1] - self.equity_curve.index[0]).days / 365.25
        metrics['CAGR (%)'] = ((self.equity_curve.iloc[-1] / self.initial_capital) ** (1/years) - 1) * 100 if years > 0 else 0
        metrics['Volatility (%)'] = returns.std() * np.sqrt(252) * 100
        metrics['Sharpe Ratio'] = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 1e-9 else 0

        roll_max = self.equity_curve.cummax()
        drawdown = (self.equity_curve - roll_max) / roll_max
        metrics['Max Drawdown (%)'] = drawdown.min() * 100

        try:
            bench_data = yf.download(benchmark_ticker, start=self.equity_curve.index[0], end=self.equity_curve.index[-1])
            bench_returns = bench_data['Close'].pct_change().dropna()
            aligned_returns, aligned_bench = returns.align(bench_returns, join='inner')
            if len(aligned_returns) > 1:
                X = sm.add_constant(aligned_bench)
                model = sm.OLS(aligned_returns, X).fit()
                metrics['Alpha'] = model.params.iloc[0] * 252
                metrics['Beta'] = model.params.iloc[1]
            else:
                metrics['Alpha'], metrics['Beta'] = np.nan, np.nan
        except Exception as e:
            print(f"Could not calculate Alpha/Beta: {e}")
            metrics['Alpha'], metrics['Beta'] = np.nan, np.nan
        return metrics

    def plot_equity_curve(self):
        if self.equity_curve is None:
            print("No equity curve to plot. Run backtest first.")
            return
        plt.figure(figsize=(12, 6))
        self.equity_curve.plot()
        plt.title('Portfolio Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value')
        plt.grid(True)
        plt.savefig('equity_curve.png')
        print("Equity curve plot saved to 'equity_curve.png'")
