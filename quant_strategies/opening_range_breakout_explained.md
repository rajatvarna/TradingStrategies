# Understanding the Equity Opening Range Breakout (ORB) Strategy

This document explains the logic and philosophy behind the Opening Range Breakout strategy for US equities. The goal is to make its mechanics clear for both users and the AI assistant.

### Core Philosophy

This is a **day trading momentum strategy**. The core idea is that the first few minutes of the stock market's open (the "opening range") can often predict the trading direction for the rest of the day.

The strategy aims to:
1.  Identify stocks that are "in play" for the day, meaning they have unusually high trading interest right at the open.
2.  Wait for the initial volatility to settle and a clear high/low range to be established.
3.  Enter a trade if the stock price "breaks out" of this range, betting that the momentum will continue in that direction for the remainder of the day.

All positions are closed by the end of the day to avoid overnight risk.

### How It Works: The Rules

The strategy is designed to run on a portfolio of stocks and follows a specific sequence of steps each trading day.

#### 1. Universe Selection

Not all stocks are suitable for this strategy. At the beginning of each month, the strategy selects a broad universe of up to 1,000 stocks to monitor. This selection is based on:
*   **Price:** The stock price must be above $5 to filter out penny stocks.
*   **Liquidity:** The universe consists of the stocks with the highest average dollar volume, ensuring we are trading liquid names.

#### 2. Identifying Candidates (The "Scan")

Five minutes after the market opens, the strategy scans all stocks in its universe to find the best trading candidates for the day. This is the most critical step and involves two powerful filters:

*   **Filter 1: High Relative Volume:** The primary filter looks for stocks experiencing unusually high volume. It calculates the "Relative Volume," which is the volume traded in the first 5 minutes compared to the average daily volume over the last 14 days. A stock must have a high relative volume (e.g., > 1) to be considered, as this indicates significant interest from traders.

*   **Filter 2: Top Movers:** From the list of high-volume stocks, the strategy selects only the top 20 with the highest relative volume. This focuses our capital on the most active and interesting stocks for the day.

#### 3. Entry and Exit Logic

For these top 20 stocks, the strategy then generates trade signals:

*   **Opening Range:** The high and low of the first 5-minute trading bar is defined as the "opening range."

*   **Entry Signal (When to Buy or Short):**
    *   If the first 5-minute bar was positive (close > open), a **buy stop order** is placed just above the high of that bar.
    *   If the first 5-minute bar was negative (close < open), a **sell stop order** (to initiate a short position) is placed just below the low of that bar.
    *   This means we only enter a trade if the price breaks out of the established range with momentum.

*   **Exit Signal (When to Sell):**
    *   **Stop Loss:** Every entry order has a pre-calculated stop-loss order attached. The stop-loss level is determined using the Average True Range (ATR), a measure of the stock's recent volatility. The stop is placed at a distance of 0.5 * ATR from the entry price.
    *   **End of Day:** If the stop loss is not hit, the position is automatically closed just before the market closes. This ensures no positions are held overnight.

### Risk Management

This strategy has risk management built into its core:

*   **Position Sizing:** The size of each trade (the number of shares) is calculated so that if the trade hits its stop-loss, the portfolio will only lose a small, fixed percentage (e.g., 1%) of its total value. This is a crucial professional risk management technique.
*   **Max Positions:** By only focusing on the top 20 candidates, the strategy limits its exposure and avoids being spread too thin across too many trades.
*   **Day Trading:** By closing all positions at the end of the day, the strategy avoids the risks associated with overnight market moves, news events, and earnings announcements.
