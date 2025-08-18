import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import yfinance as yf
from scipy import stats

from quant_strategies.strategy_base import Strategy
from quant_strategies.order import Order
from quant_strategies.data_manager import DataManager


class Backtester:
    """
    A portfolio-level backtester for quantitative trading strategies.
    This class can run two types of strategies:
    1. Signal-based: The strategy generates a DataFrame with a 'Signal' column.
       The backtester handles portfolio construction, position sizing, and execution.
    2. Order-based: The strategy generates a list of Order objects for each day.
       This mode gives the strategy full control over trade execution.
    """

    def __init__(self, strategy: Strategy, initial_capital=100000, max_positions=15,
                 max_weight_per_trade=0.10, margin=1.5, data=None):
        """
        Initializes the Backtester.
        Args:
            strategy (Strategy): The strategy instance to backtest.
            initial_capital (float): The starting capital for the portfolio.
            max_positions (int): The maximum number of concurrent open positions.
            max_weight_per_trade (float): The maximum percentage of portfolio to allocate to a single trade.
            margin (float): The margin multiplier to use for position sizing.
            data (pd.DataFrame, optional): Pre-loaded market data to avoid re-downloading.
        """
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        self.max_weight_per_trade = max_weight_per_trade
        self.margin = margin
        self.data = data
        self.equity_curve = None
        self.trade_log = None
        self.is_order_based = self.strategy.strategy_type == 'order'

    def run(self):
        """
        Runs the backtest by dispatching to the appropriate handler based on strategy type.
        """
        print(f"Running backtest in {'Order-based' if self.is_order_based else 'Signal-based'} mode.")
        if self.is_order_based:
            return self._run_order_based_backtest()
        else:
            return self._run_signal_based_backtest()

    def _run_signal_based_backtest(self):
        """
        Runs the simulation for signal-based strategies with advanced portfolio management.
        This mode uses signals but applies constraints like max positions and position sizing.
        It also ranks entry candidates based on a 'PositionScore' provided by the strategy.
        """
        print("Generating signals...")
        signals = self.strategy.generate_signals(data=self.data)

        # Consolidate all ticker data into a single DataFrame for easier processing.
        all_data_frames = []
        for ticker, df in signals.items():
            if not df.empty:
                # Ensure required columns are present
                required_cols = ['Close', 'Signal', 'PositionScore']
                if all(col in df.columns for col in required_cols):
                    temp_df = df[required_cols].copy()
                    temp_df.columns = [f'{ticker}_{col}' for col in temp_df.columns]
                    all_data_frames.append(temp_df)

        if not all_data_frames:
            print("No signal data with required columns (Close, Signal, PositionScore) available.")
            return None, None

        portfolio_data = pd.concat(all_data_frames, axis=1)
        portfolio_data.sort_index(inplace=True)
        # Forward fill Close and PositionScore, but not Signal
        close_cols = [col for col in portfolio_data.columns if col.endswith('_Close')]
        score_cols = [col for col in portfolio_data.columns if col.endswith('_PositionScore')]
        portfolio_data[close_cols + score_cols] = portfolio_data[close_cols + score_cols].ffill()
        portfolio_data.fillna(0, inplace=True)

        # Initialize portfolio state
        portfolio_value = [self.initial_capital]
        portfolio_dates = [portfolio_data.index[0]]
        open_positions = {}  # {ticker: {'entry_date':, 'entry_price':, 'quantity':}}
        self.trade_log = []

        # Main simulation loop
        for i in range(1, len(portfolio_data)):
            current_date = portfolio_data.index[i]
            prev_date = portfolio_data.index[i-1]
            current_day_data = portfolio_data.iloc[i]

            # Mark-to-market portfolio value
            current_capital = portfolio_value[-1]
            pnl = sum(
                (current_day_data[f'{t}_Close'] - portfolio_data.loc[prev_date, f'{t}_Close']) * p['quantity']
                for t, p in open_positions.items()
                if f'{t}_Close' in current_day_data and
                   portfolio_data.loc[prev_date, f'{t}_Close'] > 0
            )
            current_capital += pnl

            # Process Exits
            exited_tickers = []
            for ticker, pos in list(open_positions.items()):
                # 1. Check for Stop-Loss
                stop_loss_price = pos.get('stop_loss')
                if stop_loss_price is not None and current_day_data.get(f'{ticker}_Low', np.inf) <= stop_loss_price:
                    # Exit at stop-loss price
                    self._log_trade(ticker, pos['entry_date'], pos['entry_price'], current_date, stop_loss_price, status='Stop-Loss')
                    exited_tickers.append(ticker)
                    continue # Move to next position

                # 2. Check for Signal-based Exit
                if current_day_data.get(f'{ticker}_Signal') == -1:
                    exit_price = current_day_data[f'{ticker}_Close']
                    self._log_trade(ticker, pos['entry_date'], pos['entry_price'], current_date, exit_price)
                    exited_tickers.append(ticker)

            for ticker in exited_tickers:
                del open_positions[ticker]

            # Process Entries
            num_slots = self.max_positions - len(open_positions)
            if num_slots > 0:
                # Find and rank candidates
                candidates = []
                for t in self.strategy.tickers:
                    if current_day_data.get(f'{t}_Signal') == 1 and t not in open_positions:
                        candidates.append({
                            'ticker': t,
                            'score': current_day_data.get(f'{t}_PositionScore', 0),
                            'price': current_day_data.get(f'{t}_Close', 0)
                        })

                # Sort candidates by score (descending)
                candidates.sort(key=lambda x: x['score'], reverse=True)

                for entry in candidates[:num_slots]:
                    ticker = entry['ticker']
                    entry_price = entry['price']
                    if entry_price > 0:
                        # Position sizing based on max weight
                        amount_to_invest = current_capital * self.max_weight_per_trade * self.margin
                        quantity = amount_to_invest / entry_price

                        # Store stop-loss if available
                        stop_loss = current_day_data.get(f'{ticker}_stop_loss')

                        open_positions[ticker] = {
                            'entry_date': current_date,
                            'entry_price': entry_price,
                            'quantity': quantity,
                            'stop_loss': stop_loss
                        }

            portfolio_value.append(current_capital)
            portfolio_dates.append(current_date)

        # Finalization
        self._close_open_positions(portfolio_data.index[-1], portfolio_data.iloc[-1], open_positions)
        self.equity_curve = pd.Series(portfolio_value, index=portfolio_dates)
        self.trade_log = pd.DataFrame(self.trade_log) if self.trade_log else pd.DataFrame()
        print("Backtest finished.")
        return self.equity_curve, self.trade_log

    def _run_order_based_backtest(self):
        """
        Runs the simulation for order-based strategies. The strategy is given the portfolio
        state and is responsible for generating specific buy/sell orders with quantities.
        """
        print("Backtester downloading daily data for portfolio accounting...")
        dm = DataManager()
        daily_data = dm.download_data(
            self.strategy.tickers,
            start_date=self.strategy.start_date,
            end_date=self.strategy.end_date,
            save=False
        )

        if daily_data.empty:
            print("Could not download daily data for accounting. Exiting.")
            return None, None

        # Initialize portfolio state variables
        portfolio_value = [self.initial_capital]
        portfolio_dates = [daily_data.index[0]]
        open_positions = {}  # {ticker: {'entry_date':, 'entry_price':, 'quantity':}}
        self.trade_log = []

        # Main simulation loop
        for i in range(1, len(daily_data)):
            current_date = daily_data.index[i]
            prev_date = daily_data.index[i-1]

            # Mark-to-market
            current_capital = portfolio_value[-1]
            pnl = sum(
                (daily_data.loc[current_date, ('Close', t)] - daily_data.loc[prev_date, ('Close', t)]) * p['quantity']
                for t, p in open_positions.items() if daily_data.loc[prev_date, ('Close', t)] > 0
            )
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

        # Finalization
        last_date = daily_data.index[-1]
        last_prices = daily_data.loc[last_date]
        for ticker, pos in open_positions.items():
            last_price = last_prices[('Close', ticker)]
            self._log_trade(ticker, pos['entry_date'], pos['entry_price'], last_date, last_price, status='Open')

        self.equity_curve = pd.Series(portfolio_value, index=portfolio_dates)
        self.trade_log = pd.DataFrame(self.trade_log) if self.trade_log else pd.DataFrame()
        print("Backtest finished.")
        return self.equity_curve, self.trade_log

    def _log_trade(self, ticker, buy_date, buy_price, sell_date, sell_price, status='Closed'):
        """Helper function to record a single trade's details."""
        trade_return = (sell_price - buy_price) / buy_price if buy_price > 0 else 0
        self.trade_log.append({
            'Ticker': ticker, 'Buy Date': buy_date, 'Buy Price': buy_price,
            'Sell Date': sell_date, 'Sell Price': sell_price, 'Return': trade_return, 'Status': status
        })

    def _close_open_positions(self, last_date, last_day_data, open_positions):
        """Closes all open positions at the end of the backtest period."""
        for ticker, pos_info in open_positions.items():
            close_price_col = f'{ticker}_Close'
            if close_price_col in last_day_data:
                last_price = last_day_data[close_price_col]
                self._log_trade(ticker, pos_info['entry_date'], pos_info['entry_price'], last_date, last_price, status='Open')

    def calculate_statistics(self, benchmark_ticker='SPY'):
        """
        Calculates key performance indicators (KPIs) for the backtest results.
        """
        if self.equity_curve is None or self.equity_curve.empty or len(self.equity_curve) < 2:
            print("Cannot calculate statistics: Equity curve is empty or too short.")
            return {}

        print("Calculating performance statistics...")
        metrics = {}
        returns = self.equity_curve.pct_change().dropna()

        # --- Performance Metrics ---
        trading_days = 252
        years = (self.equity_curve.index[-1] - self.equity_curve.index[0]).days / 365.25
        metrics['CAGR (%)'] = ((self.equity_curve.iloc[-1] / self.initial_capital) ** (1/years) - 1) * 100 if years > 0 else 0
        metrics['Annual Volatility (%)'] = returns.std() * np.sqrt(trading_days) * 100

        # Sharpe and Sortino Ratios
        risk_free_rate = 0.0
        sharpe_ratio = (returns.mean() * trading_days - risk_free_rate) / (returns.std() * np.sqrt(trading_days)) if returns.std() > 0 else 0
        metrics['Sharpe Ratio'] = sharpe_ratio

        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(trading_days)
        sortino_ratio = (returns.mean() * trading_days - risk_free_rate) / downside_std if downside_std > 0 else 0
        metrics['Sortino Ratio'] = sortino_ratio

        # --- Risk Metrics ---
        roll_max = self.equity_curve.cummax()
        drawdown = (self.equity_curve - roll_max) / roll_max
        metrics['Max Drawdown (%)'] = drawdown.min() * 100
        metrics['Calmar Ratio'] = metrics['CAGR (%)'] / abs(metrics['Max Drawdown (%)']) if metrics['Max Drawdown (%)'] != 0 else np.nan

        # --- Alpha and Beta vs. Benchmark ---
        try:
            bench_data = yf.download(benchmark_ticker, start=self.equity_curve.index[0], end=self.equity_curve.index[-1])
            bench_returns = bench_data['Close'].pct_change().dropna()
            aligned_returns, aligned_bench = returns.align(bench_returns, join='inner')

            if len(aligned_returns) > 1:
                X = sm.add_constant(aligned_bench)
                model = sm.OLS(aligned_returns, X).fit()
                metrics['Alpha'] = model.params.iloc[0] * trading_days
                metrics['Beta'] = model.params.iloc[1]
            else:
                metrics['Alpha'], metrics['Beta'] = np.nan, np.nan
        except Exception as e:
            print(f"Could not calculate Alpha/Beta vs. {benchmark_ticker}: {e}")
            metrics['Alpha'], metrics['Beta'] = np.nan, np.nan

        return metrics

    def plot_equity_curve(self, filename='equity_curve.png'):
        """
        Generates and saves a plot of the portfolio's equity curve over time.
        """
        if self.equity_curve is None:
            print("No equity curve to plot. Run backtest first.")
            return
        plt.figure(figsize=(12, 6))
        self.equity_curve.plot(title='Portfolio Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value')
        plt.grid(True)
        plt.savefig(filename)
        print(f"Equity curve plot saved to '{filename}'")
