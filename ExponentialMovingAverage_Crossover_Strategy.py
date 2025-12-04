import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt


def calculate_ema(prices, period):
    alpha = 2 / (period + 1)
    ema = np.zeros(len(prices), dtype=float)
    ema[0] = float(prices[0])

    for i in range(1, len(prices)):
        price_i = float(prices[i])
        ema[i] = alpha * price_i + (1 - alpha) * ema[i - 1]

    return ema


def backtest_ema_crossover(stock, short_period, long_period, start_date, end_date, initial_capital=10000):
    if short_period >= long_period:
        raise ValueError("Short EMA period must be less than long EMA period.")

    # Fetch data (explicit about auto_adjust)
    data = yf.download(stock, start=start_date, end=end_date, auto_adjust=True)['Close']

    if data.empty:
        raise ValueError("Error: Could not fetch price data.")

    # Ensure prices is a 1-D numpy array of native floats (no 0-D arrays)
    prices = data.to_numpy().ravel().astype(float)
    dates = data.index

    # EMAs
    short_ema = calculate_ema(prices, short_period)
    long_ema = calculate_ema(prices, long_period)

    position = 0
    shares = 0.0
    cash = float(initial_capital)

    portfolio_value = []
    buy_and_hold_value = []
    trades = []

    first_price = float(prices[0])

    for i in range(1, len(prices)):
        current_price = float(prices[i])

        # BUY
        if short_ema[i] > long_ema[i] and short_ema[i - 1] <= long_ema[i - 1] and position == 0:
            shares = cash / current_price
            cash = 0.0
            position = 1
            trades.append(("BUY", dates[i], float(current_price)))

        # SELL
        elif short_ema[i] < long_ema[i] and short_ema[i - 1] >= long_ema[i - 1] and position == 1:
            cash = shares * current_price
            shares = 0.0
            position = 0
            trades.append(("SELL", dates[i], float(current_price)))

        # Portfolio value (store as Python float)
        portfolio_value.append(float(cash + shares * current_price))
        buy_and_hold_value.append(float(initial_capital * (current_price / first_price)))

    # Close final position if any
    if position == 1:
        last_price = float(prices[-1])
        cash = shares * last_price
        shares = 0.0

    last_price = float(prices[-1])

    return {
        'dates': dates,
        'prices': prices,
        'short_ema': short_ema,
        'long_ema': long_ema,
        'portfolio_value': portfolio_value,
        'buy_and_hold_value': buy_and_hold_value,
        'trades': trades,
        'final_value': float(cash),
        'buy_and_hold_final': float(initial_capital * (last_price / first_price)),
        'initial_capital': initial_capital
    }


def plot_results(stock, short_period, long_period, results):

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Price + EMA
    ax1.plot(results['dates'], results['prices'], label='Price', linewidth=2, alpha=0.7)
    ax1.plot(results['dates'], results['short_ema'], label=f'EMA {short_period}', linewidth=1.5)
    ax1.plot(results['dates'], results['long_ema'], label=f'EMA {long_period}', linewidth=1.5)

    # Trades
    for trade_type, date, price in results['trades']:
        color = 'green' if trade_type == "BUY" else 'red'
        marker = '^' if trade_type == "BUY" else 'v'
        ax1.scatter(date, price, color=color, marker=marker, s=100)

    ax1.set_ylabel('Price ($)')
    ax1.set_title(f'{stock} - EMA Crossover Strategy ({short_period}/{long_period})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Portfolio value
    ax2.plot(results['dates'][1:], results['portfolio_value'], label='EMA Strategy', linewidth=2)
    ax2.plot(results['dates'][1:], results['buy_and_hold_value'], label='Buy & Hold', linewidth=2, linestyle='--')

    ax2.set_ylabel('Portfolio Value ($)')
    ax2.set_xlabel('Date')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def print_results(stock, short_period, long_period, results):
    strategy_return = ((results['final_value'] - results['initial_capital']) /
                       results['initial_capital']) * 100
    buy_hold_return = ((results['buy_and_hold_final'] - results['initial_capital']) /
                       results['initial_capital']) * 100

    print(f"\nStock: {stock}")
    print(f"Strategy: EMA Crossover ({short_period}/{long_period})")
    print(f"Initial Capital: ${results['initial_capital']:,.2f}")
    print(f"Number of Trades: {len(results['trades'])}")
    print(f"Final Value (Strategy): ${results['final_value']:,.2f}")
    print(f"Total Return (Strategy): {strategy_return:.2f}%")
    print(f"Final Value (Buy & Hold): ${results['buy_and_hold_final']:,.2f}")
    print(f"Total Return (Buy & Hold): {buy_hold_return:.2f}%")

    diff = results['final_value'] - results['buy_and_hold_final']
    if diff > 0:
        print(f"Strategy outperformed by ${diff:,.2f} ({strategy_return - buy_hold_return:.2f}%)")
    else:
        print(f"Strategy underperformed by ${abs(diff):,.2f} ({abs(strategy_return - buy_hold_return):.2f}%)")

    if results['trades']:
        print("\nRECENT TRADES (Last 5):")
        for t in results['trades'][-5:]:
            print(f"  {t[0]:4s} - {t[1].strftime('%Y-%m-%d')} at ${t[2]:.2f}")

print("Strategy: Buy when short EMA crosses above long EMA")
print("          Sell when short EMA crosses below long EMA")

stock = input("Enter stock symbol (e.g., AAPL, TSLA): ").strip().upper()
short_period = int(input("Enter short EMA period (e.g., 12, 20, 50): "))
long_period = int(input("Enter long EMA period (e.g., 26, 50, 200): "))
initial_capital = float(input("Enter initial capital (default 10000): ") or "10000")

start_date = input("Enter start date (YYYY-MM-DD) or press Enter for 2020-01-01: ").strip() or "2020-01-01"
end_date = input("Enter end date (YYYY-MM-DD) or press Enter for 2025-06-30: ").strip() or "2025-06-30"

results = backtest_ema_crossover(stock, short_period, long_period, start_date, end_date, initial_capital)

print_results(stock, short_period, long_period, results)
plot_results(stock, short_period, long_period, results)
