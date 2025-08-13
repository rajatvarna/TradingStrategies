# Understanding the Crypto Momentum Breakout Strategy

This document explains the logic and philosophy behind the Crypto Momentum Breakout strategy. The goal is to make its mechanics clear for both users and the AI assistant.

### Core Philosophy

The strategy is based on two well-known market principles:

1.  **Momentum:** The idea that assets that have been performing well recently will continue to perform well in the short term.
2.  **Breakout:** The concept of entering a trade when an asset's price moves decisively above a recent high, suggesting the beginning of a strong new trend.

By combining these, the strategy aims to identify cryptocurrencies that are already showing signs of strength and enter a position at a point where that strength is likely to accelerate.

### How It Works: The Rules

The strategy scans a universe of cryptocurrencies daily and applies a set of strict rules to decide whether to enter or exit a position.

#### Entry Signals (When to Buy)

A "buy" signal is generated for a cryptocurrency only if **all** of the following five conditions are met:

1.  **Price Breakout (`Close > HH20`):** The current closing price must be higher than the highest closing price over the last 20 days. This is the core breakout signal, indicating strong upward momentum.

2.  **Minimum Price (`Close > $0.50`):** The coin must have a price of at least $0.50. This is a liquidity and quality filter to avoid extremely low-priced, volatile "penny stocks" of the crypto world.

3.  **Minimum Volume (`Volume > 100,000`):** The coin must have traded at least 100,000 units in the last 24 hours. This ensures there is enough trading activity for us to enter and exit positions without significantly impacting the price.

4.  **Short-Term Trend (`EMA5 > EMA5_p`):** The 5-day Exponential Moving Average (EMA) must be higher than it was on the previous day. The EMA is a type of moving average that gives more weight to recent prices. This condition confirms that the immediate price trend is upwards.

5.  **Market Regime Filter (`RiskOn`):** The strategy only takes new trades if the broader crypto market is considered "risk-on". We use Bitcoin as a market barometer. A "risk-on" environment is defined as Bitcoin's current price being above its 20-day EMA. This helps avoid entering new long positions when the overall market is showing signs of weakness.

#### Exit Signal (When to Sell)

The strategy uses a simple but effective trend-based exit rule:

1.  **Trend Reversal (`EMA5 < EMA5_p`):** A "sell" signal is generated if the 5-day EMA crosses below its value from the previous day. This indicates that the short-term upward momentum has faded and a potential downtrend is beginning. The goal is to exit the position quickly to protect profits or cut losses.

### Risk Management

The backtester applies two main risk management rules on top of the strategy's signals:

*   **Maximum Positions:** The portfolio will not hold more than 15 open positions at any given time.
*   **Position Sizing:** Each new trade is sized to risk a small, predefined percentage of the total portfolio value, ensuring that no single trade can cause a catastrophic loss.
