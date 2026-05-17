import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

def calculate_max_drawdown(prices):
    """
    Calculate maximum drawdown and related metrics
    Returns: max_drawdown, peak_date, trough_date, recovery_date
    """
    peak = prices[0]
    peak_index = 0
    max_drawdown = 0
    max_dd_peak_index = 0
    max_dd_trough_index = 0
    
    drawdowns = []
    peaks = []
    
    for i in range(len(prices)):
        # Update peak if new high
        if prices[i] > peak:
            peak = prices[i]
            peak_index = i
        
        # Calculate current drawdown
        drawdown = (peak - prices[i]) / peak
        drawdowns.append(drawdown)
        peaks.append(peak)
        
        # Update maximum drawdown
        if drawdown > max_drawdown:
            max_drawdown = drawdown
            max_dd_peak_index = peak_index
            max_dd_trough_index = i
    
    # Find recovery date (when price exceeds the peak after the trough)
    recovery_index = None
    peak_price = prices[max_dd_peak_index]
    for i in range(max_dd_trough_index, len(prices)):
        if prices[i] >= peak_price:
            recovery_index = i
            break
    
    return max_drawdown, max_dd_peak_index, max_dd_trough_index, recovery_index, drawdowns, peaks

def plot_drawdown(stock, prices, drawdowns, peaks, peak_idx, trough_idx, recovery_idx):
    """Plot the stock price and drawdown over time"""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot 1: Stock Price
    dates = prices.index
    ax1.plot(dates, prices.values, label='Stock Price', linewidth=2)
    ax1.plot(dates, peaks, label='Running Peak', linestyle='--', alpha=0.7)
    
    # Mark the maximum drawdown period
    ax1.axvline(dates[peak_idx], color='green', linestyle='--', alpha=0.5, label='Peak')
    ax1.axvline(dates[trough_idx], color='red', linestyle='--', alpha=0.5, label='Trough')
    if recovery_idx:
        ax1.axvline(dates[recovery_idx], color='blue', linestyle='--', alpha=0.5, label='Recovery')
    
    ax1.set_ylabel('Price ($)')
    ax1.set_title(f'{stock} - Maximum Drawdown Analysis')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Drawdown percentage
    ax2.fill_between(dates, 0, [-d*100 for d in drawdowns], alpha=0.3, color='red')
    ax2.plot(dates, [-d*100 for d in drawdowns], color='red', linewidth=2)
    ax2.set_ylabel('Drawdown (%)')
    ax2.set_xlabel('Date')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(0, color='black', linewidth=0.5)
    
    plt.tight_layout()
    plt.show()

print("Maximum Drawdown Calculator")
print("="*50)

# User inputs
stock = input("Enter stock symbol (e.g., AAPL, TSLA): ").strip().upper()

start_date = input("Enter start date (YYYY-MM-DD) or press Enter for default (2020-01-01): ").strip()
end_date = input("Enter end date (YYYY-MM-DD) or press Enter for default (2025-06-30): ").strip()

if not start_date:
    start_date = "2020-01-01"
if not end_date:
    end_date = "2025-06-30"

print(f"\nFetching data for {stock}...")

# Fetch stock data
data = yf.download(stock, start=start_date, end=end_date, auto_adjust=True)['Close']

if data.empty:
    print("Error: Could not fetch data for the specified stock.")
else:
    # Calculate maximum drawdown
    max_dd, peak_idx, trough_idx, recovery_idx, drawdowns, peaks = calculate_max_drawdown(data.values)
    
    # Display results
    print("\n" + "="*50)
    print("MAXIMUM DRAWDOWN ANALYSIS")
    print("="*50)
    print(f"Maximum Drawdown: {max_dd*100:.2f}%")
    print(f"\nPeak Date: {data.index[peak_idx].strftime('%Y-%m-%d')}")
    print(f"Peak Price: ${data.values[peak_idx]:.2f}")
    print(f"\nTrough Date: {data.index[trough_idx].strftime('%Y-%m-%d')}")
    print(f"Trough Price: ${data.values[trough_idx]:.2f}")
    print(f"Price Drop: ${data.values[peak_idx] - data.values[trough_idx]:.2f}")
    
    if recovery_idx:
        print(f"\nRecovery Date: {data.index[recovery_idx].strftime('%Y-%m-%d')}")
        days_to_recover = (data.index[recovery_idx] - data.index[trough_idx]).days
        print(f"Days to Recover: {days_to_recover}")
    else:
        print("\nStatus: Has not yet recovered to peak price")
    
    print("="*50)
    
    # Plot the results
    plot_drawdown(stock, data, drawdowns, peaks, peak_idx, trough_idx, recovery_idx)