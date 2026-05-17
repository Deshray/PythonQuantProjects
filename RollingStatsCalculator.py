import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

def calculate_rolling_stats(stock, window_size, stat_type):
    """
    Calculate rolling statistics for a given stock
    stat_type: 'volatility', 'returns', 'sharpe'
    """
    # Fetch stock data
    data = yf.download(stock, start="2020-01-01", end="2025-06-30", auto_adjust=True)['Close']
    
    # Calculate daily returns
    daily_returns = data.pct_change().dropna()
    
    rolling_values = []
    dates = []
    
    # Calculate rolling statistic
    for i in range(window_size, len(daily_returns)):
        window_data = daily_returns.iloc[i-window_size:i]
        
        if stat_type == 'volatility':
            # Annualized volatility
            stat = window_data.std() * np.sqrt(252)
        elif stat_type == 'returns':
            # Annualized average return
            stat = window_data.mean() * 252
        elif stat_type == 'sharpe':
            # Sharpe ratio (assuming 0% risk-free rate for simplicity)
            stat = (window_data.mean() / window_data.std()) * np.sqrt(252)
        
        rolling_values.append(stat)
        dates.append(data.index[i])
    
    return dates, rolling_values

def plot_rolling_stats(stock, window_size, stat_type):
    """Plot the rolling statistics"""
    dates, values = calculate_rolling_stats(stock, window_size, stat_type)
    
    plt.figure(figsize=(12, 6))
    plt.plot(dates, values, linewidth=2)
    
    # Set title based on stat type
    titles = {
        'volatility': f'{window_size}-Day Rolling Volatility',
        'returns': f'{window_size}-Day Rolling Returns',
        'sharpe': f'{window_size}-Day Rolling Sharpe Ratio'
    }
    
    ylabels = {
        'volatility': 'Annualized Volatility',
        'returns': 'Annualized Return',
        'sharpe': 'Sharpe Ratio'
    }
    
    plt.title(f'{titles[stat_type]} for {stock}')
    plt.xlabel('Date')
    plt.ylabel(ylabels[stat_type])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

print("Rolling Statistics Visualizer")
print("-" * 40)

# User inputs
stock = input("Enter stock symbol (e.g., AAPL): ").strip().upper()
window_size = int(input("Enter window size in days (e.g., 30, 60, 90): "))

print("\nAvailable statistics:")
print("1. Volatility")
print("2. Returns")
print("3. Sharpe Ratio")

choice = input("\nChoose statistic (1/2/3): ").strip()

stat_map = {'1': 'volatility', '2': 'returns', '3': 'sharpe'}

if choice in stat_map:
    stat_type = stat_map[choice]
    plot_rolling_stats(stock, window_size, stat_type)
else:
    print("Invalid choice!")