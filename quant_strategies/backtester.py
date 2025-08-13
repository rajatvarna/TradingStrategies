import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

class Backtester:
    """
    A portfolio-level backtester for quantitative trading strategies.

    This class takes a strategy object, runs a simulation based on its signals,
    and calculates a comprehensive set of performance metrics.
    """

    def __init__(self, strategy, initial_capital=1000000, max_positions=15,
                 trade_risk_pct=0.01, margin=1.5):
        """
        Initializes the Backtester.

        Args:
            strategy: An instantiated strategy object with a `run_signal_generation` method.
            initial_capital (float): The starting capital for the backtest.
            max_positions (int): The maximum number of concurrent open positions.
            trade_risk_pct (float): The percentage of the portfolio to risk per trade.
            margin (float): The margin multiplier (e.g., 1.5 for 1.5x margin).
        """
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        self.trade_risk_pct = trade_risk_pct
        self.margin = margin
        self.equity_curve = None
        self.trade_log = None

    def run(self):
        """
        Runs the day-by-day portfolio simulation.
        """
        print("Starting backtest...")

        # 1. Get signals and data from the strategy
        signals = self.strategy.run_signal_generation()

        # 2. Consolidate data into a single DataFrame for easier processing
        all_data_frames = []
        for ticker, df in signals.items():
            if not df.empty:
                # Add ticker prefix to columns to avoid clashes
                renamed_df = df.add_prefix(f'{ticker}_')
                all_data_frames.append(renamed_df)

        if not all_data_frames:
            print("No signal data available for any tickers. Exiting backtest.")
            return None, None

        portfolio_data = pd.concat(all_data_frames, axis=1)
        portfolio_data.sort_index(inplace=True)
        # Forward-fill data to handle non-trading days or missing data points
        portfolio_data.fillna(method='ffill', inplace=True)
        portfolio_data.fillna(0, inplace=True) # Fill any remaining NaNs at the start

        # 3. Initialize portfolio tracking
        portfolio_value = [self.initial_capital]
        portfolio_dates = [portfolio_data.index[0]]
        open_positions = {}  # {ticker: {'entry_date':, 'entry_price':, 'quantity':}}
        self.trade_log = []

        # 4. Main simulation loop
        for i in range(1, len(portfolio_data)):
            current_date = portfolio_data.index[i]
            prev_date = portfolio_data.index[i-1]
            current_day_data = portfolio_data.iloc[i]

            # --- a. Update portfolio value based on previous day's holdings ---
            current_capital = portfolio_value[-1]
            pnl = 0
            for ticker, pos in open_positions.items():
                prev_price = portfolio_data.loc[prev_date, f'{ticker}_Close']
                current_price = current_day_data[f'{ticker}_Close']
                if prev_price > 0: # Avoid division by zero
                    pnl += (current_price - prev_price) * pos['quantity']

            current_capital += pnl
            portfolio_value.append(current_capital)
            portfolio_dates.append(current_date)

            # --- b. Process exits ---
            exited_tickers = []
            for ticker in list(open_positions.keys()):
                if current_day_data[f'{ticker}_Signal'] == -1:
                    exit_price = current_day_data[f'{ticker}_Close']
                    pos_info = open_positions[ticker]
                    trade_return = (exit_price - pos_info['entry_price']) / pos_info['entry_price'] if pos_info['entry_price'] > 0 else 0

                    self.trade_log.append({
                        'Ticker': ticker, 'Buy Date': pos_info['entry_date'], 'Buy Price': pos_info['entry_price'],
                        'Sell Date': current_date, 'Sell Price': exit_price, 'Return': trade_return, 'Status': 'Closed'
                    })
                    exited_tickers.append(ticker)

            for ticker in exited_tickers:
                del open_positions[ticker]

            # --- c. Process entries ---
            num_slots_available = self.max_positions - len(open_positions)
            if num_slots_available > 0:
                # Find all entry candidates for the current day
                entry_candidates = []
                for ticker in self.strategy.tickers:
                    if ticker != 'BTC-USD' and current_day_data[f'{ticker}_Signal'] == 1 and ticker not in open_positions:
                        entry_candidates.append({
                            'ticker': ticker,
                            'score': current_day_data.get(f'{ticker}_PositionScore', 0) # Use PositionScore if available
                        })

                # Rank candidates (e.g., by PositionScore)
                entry_candidates.sort(key=lambda x: x['score'], reverse=True)

                # Take the top candidates for available slots
                for candidate in entry_candidates[:num_slots_available]:
                    ticker = candidate['ticker']
                    entry_price = current_day_data[f'{ticker}_Close']
                    if entry_price > 0:
                        amount_to_invest = current_capital * self.trade_risk_pct * self.margin
                        quantity = amount_to_invest / entry_price

                        open_positions[ticker] = {
                            'entry_date': current_date,
                            'entry_price': entry_price,
                            'quantity': quantity
                        }

        # 5. Handle positions still open at the end of the backtest
        last_date = portfolio_data.index[-1]
        for ticker, pos_info in open_positions.items():
            last_price = portfolio_data.loc[last_date, f'{ticker}_Close']
            trade_return = (last_price - pos_info['entry_price']) / pos_info['entry_price'] if pos_info['entry_price'] > 0 else 0
            self.trade_log.append({
                'Ticker': ticker, 'Buy Date': pos_info['entry_date'], 'Buy Price': pos_info['entry_price'],
                'Sell Date': last_date, 'Sell Price': last_price, 'Return': trade_return, 'Status': 'Open'
            })

        self.equity_curve = pd.Series(portfolio_value, index=portfolio_dates)
        self.trade_log = pd.DataFrame(self.trade_log)

        print("Backtest finished.")
        return self.equity_curve, self.trade_log

    def calculate_statistics(self, benchmark_ticker='^GSPC'):
        """
        Calculates and returns a dictionary of performance metrics.

        Args:
            benchmark_ticker (str): The ticker for the benchmark (e.g., S&P 500).
        """
        if self.equity_curve is None:
            raise ValueError("Run the backtest before calculating statistics.")

        print("Calculating performance statistics...")

        metrics = {}
        returns = self.equity_curve.pct_change().dropna()

        # CAGR
        years = (self.equity_curve.index[-1] - self.equity_curve.index[0]).days / 365.25
        metrics['CAGR (%)'] = ((self.equity_curve.iloc[-1] / self.initial_capital) ** (1/years) - 1) * 100 if years > 0 else 0

        # Annualized Volatility
        metrics['Volatility (%)'] = returns.std() * np.sqrt(252) * 100

        # Sharpe Ratio
        metrics['Sharpe Ratio'] = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else 0

        # Max Drawdown
        roll_max = self.equity_curve.cummax()
        drawdown = (self.equity_curve - roll_max) / roll_max
        metrics['Max Drawdown (%)'] = drawdown.min() * 100

        # Alpha and Beta
        # (Requires benchmark data, which would be downloaded here)
        try:
            bench_data = yf.download(benchmark_ticker, start=self.equity_curve.index[0], end=self.equity_curve.index[-1])
            bench_returns = bench_data['Adj Close'].pct_change().dropna()

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
        """
        Plots the portfolio equity curve.
        """
        if self.equity_curve is None:
            print("No equity curve to plot. Run backtest first.")
            return

        plt.figure(figsize=(12, 6))
        self.equity_curve.plot()
        plt.title('Portfolio Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value')
        plt.grid(True)
        plt.show()
