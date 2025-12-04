import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt


def calculate_sma(prices, period):
    """Calculate Simple Moving Average (returns 1-D numpy array of floats)."""
    prices = np.asarray(prices).ravel().astype(float)
    sma = np.zeros(len(prices), dtype=float)
    if period <= 0:
        raise ValueError("period must be > 0")
    for i in range(period - 1, len(prices)):
        sma[i] = float(np.mean(prices[i - period + 1:i + 1]))
    return sma


def backtest_ma_crossover(stock, short_period, long_period, initial_capital=10000,
                          start_date="2020-01-01", end_date="2025-06-30"):
    """
    Backtest Moving Average crossover strategy.
    Buy when short MA crosses above long MA (Golden Cross).
    Sell when short MA crosses below long MA (Death Cross).
    (No strategy logic changed; only type-safety and robustness fixes applied.)
    """
    if short_period >= long_period:
        raise ValueError("short_period must be less than long_period")

    # Explicit about adjusted prices to avoid yfinance FutureWarning
    data = yf.download(stock, start=start_date, end=end_date, auto_adjust=True)['Close']

    if data.empty:
        raise ValueError("Error: Could not fetch data.")

    # Ensure 1-D array of native floats
    prices_all = data.to_numpy().ravel().astype(float)
    dates_all = data.index

    # Compute MAs (full length)
    short_ma_full = calculate_sma(prices_all, short_period)
    long_ma_full = calculate_sma(prices_all, long_period)

    position = 0
    shares = 0.0
    cash = float(initial_capital)

    portfolio_value = []
    buy_and_hold_value = []
    trades = []

    start_idx = long_period - 1  # first index where both MAs are defined
    if start_idx >= len(prices_all):
        raise ValueError("Not enough data for the chosen long_period.")

    first_price = float(prices_all[start_idx])

    # iterate from start_idx to end (inclusive)
    for i in range(start_idx, len(prices_all)):
        current_price = float(prices_all[i])

        # Use the full-length MA arrays for crossover signals
        # BUY: short crosses above long
        if i > start_idx and short_ma_full[i] > long_ma_full[i] and short_ma_full[i - 1] <= long_ma_full[i - 1] and position == 0:
            shares = cash / current_price
            cash = 0.0
            position = 1
            trades.append(("BUY", dates_all[i], float(current_price)))

        # SELL: short crosses below long
        elif i > start_idx and short_ma_full[i] < long_ma_full[i] and short_ma_full[i - 1] >= long_ma_full[i - 1] and position == 1:
            cash = shares * current_price
            shares = 0.0
            position = 0
            trades.append(("SELL", dates_all[i], float(current_price)))

        # Record portfolio and buy&hold values (store as Python floats)
        portfolio_value.append(float(cash + shares * current_price))
        buy_and_hold_value.append(float(initial_capital * (current_price / first_price)))

    # Liquidate final position if any
    if position == 1:
        last_price = float(prices_all[-1])
        cash = shares * last_price
        shares = 0.0

    final_value = float(cash)
    buy_and_hold_final = float(initial_capital * (float(prices_all[-1]) / first_price))

    # Calculate winning/losing trades (pair BUY then SELL)
    winning_trades = 0
    losing_trades = 0
    # Only consider complete buy-sell pairs
    for k in range(0, len(trades) - 1, 2):
        buy_price = float(trades[k][2])
        sell_price = float(trades[k + 1][2])
        if sell_price > buy_price:
            winning_trades += 1
        else:
            losing_trades += 1

    # daily returns (safe: avoid division by zero)
    returns = []
    for j in range(1, len(portfolio_value)):
        prev = portfolio_value[j - 1]
        curr = portfolio_value[j]
        if prev != 0.0:
            returns.append((curr - prev) / prev)
        else:
            returns.append(0.0)

    # Return slices aligned to start_idx
    return {
        'dates': dates_all[start_idx:],
        'prices': prices_all[start_idx:],
        'short_ma': short_ma_full[start_idx:],
        'long_ma': long_ma_full[start_idx:],
        'portfolio_value': portfolio_value,
        'buy_and_hold_value': buy_and_hold_value,
        'trades': trades,
        'final_value': final_value,
        'buy_and_hold_final': buy_and_hold_final,
        'initial_capital': float(initial_capital),
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'returns': returns
    }


def plot_results(stock, short_period, long_period, results):
    """Plot the backtest results"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    ax1.plot(results['dates'], results['prices'], label='Price', linewidth=2, alpha=0.7, color='black')
    ax1.plot(results['dates'], results['short_ma'], label=f'{short_period}-day MA', linewidth=1.5)
    ax1.plot(results['dates'], results['long_ma'], label=f'{long_period}-day MA', linewidth=1.5)

    buy_signals = [t for t in results['trades'] if t[0] == 'BUY']
    sell_signals = [t for t in results['trades'] if t[0] == 'SELL']

    if buy_signals:
        ax1.scatter([t[1] for t in buy_signals], [t[2] for t in buy_signals],
                    marker='^', s=120, edgecolors='black', label='Buy')
    if sell_signals:
        ax1.scatter([t[1] for t in sell_signals], [t[2] for t in sell_signals],
                    marker='v', s=120, edgecolors='black', label='Sell')

    ax1.set_ylabel('Price ($)')
    ax1.set_title(f'{stock} - MA Crossover Strategy ({short_period}/{long_period})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Ensure lengths match for plotting
    ax2.plot(results['dates'], results['portfolio_value'], label='MA Strategy', linewidth=2.5)
    ax2.plot(results['dates'], results['buy_and_hold_value'], label='Buy & Hold', linewidth=2.5, linestyle='--')
    ax2.set_ylabel('Portfolio Value ($)')
    ax2.set_xlabel('Date')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def print_results(stock, short_period, long_period, results):
    """Print backtest statistics"""
    strategy_return = ((results['final_value'] - results['initial_capital']) /
                       results['initial_capital']) * 100
    buy_hold_return = ((results['buy_and_hold_final'] - results['initial_capital']) /
                       results['initial_capital']) * 100

    wins = results.get('winning_trades', 0)
    losses = results.get('losing_trades', 0)
    total_pairs = max(1, wins + losses)
    win_rate = (wins / total_pairs) * 100

    print(f"Stock: {stock}")
    print(f"Strategy: MA Crossover ({short_period}-day / {long_period}-day)")
    print(f"Initial Capital: ${results['initial_capital']:,.2f}")
    print(f"Total Number of Trades: {len(results['trades'])}")
    print(f"Final Portfolio Value: ${results['final_value']:,.2f}")
    print(f"Total Return: {strategy_return:+.2f}%")
    print(f"Winning Trades: {wins}")
    print(f"Losing Trades: {losses}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Final Value (Buy & Hold): ${results['buy_and_hold_final']:,.2f}")
    print(f"Total Return (Buy & Hold): {buy_hold_return:+.2f}%")

    for t in results['trades']:
        print(f"{t[0]} on {t[1].strftime('%Y-%m-%d')} at ${float(t[2]):.2f}")

stock = input("Enter stock symbol (e.g., AAPL, TSLA): ").upper()
short_window = int(input("Enter short SMA period (e.g., 20, 50): "))
long_window = int(input("Enter long SMA period (e.g., 50, 200): "))
initial_capital = float(input("Enter initial capital (default 10000): ") or 10000)
start_date = input("Enter start date (YYYY-MM-DD) or press Enter for 2020-01-01: ") or "2020-01-01"
end_date = input("Enter end date (YYYY-MM-DD) or press Enter for 2025-06-30: ") or "2025-06-30"

results = backtest_ma_crossover(
    stock,
    short_window,
    long_window,
    initial_capital,
    start_date,
    end_date
)

print_results(stock, short_window, long_window, results)
