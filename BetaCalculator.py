import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

def calculate_beta(stock, market_index, start_date, end_date):
    """
    Calculate beta of a stock relative to a market index
    Beta = Covariance(stock, market) / Variance(market)
    """
    
    # Fetch stock and market data
    stock_data = yf.download(stock, start=start_date, end=end_date, auto_adjust=True)['Close']
    market_data = yf.download(market_index, start=start_date, end=end_date, auto_adjust=True)['Close']
    
    if stock_data.empty or market_data.empty:
        print("Error: Could not fetch data.")
        return None, None, None
    
    # Calculate daily returns
    stock_returns = stock_data.pct_change().dropna()
    market_returns = market_data.pct_change().dropna()
    
    # Align the data (in case of different dates)
    aligned_data = np.column_stack([stock_returns, market_returns])
    aligned_data = aligned_data[~np.isnan(aligned_data).any(axis=1)]
    
    stock_returns_aligned = aligned_data[:, 0]
    market_returns_aligned = aligned_data[:, 1]
    
    # Calculate beta
    covariance = np.cov(stock_returns_aligned, market_returns_aligned)[0, 1]
    market_variance = np.var(market_returns_aligned)
    beta = covariance / market_variance
    
    # Calculate correlation for additional insight
    correlation = np.corrcoef(stock_returns_aligned, market_returns_aligned)[0, 1]
    
    return beta, correlation, (stock_returns_aligned, market_returns_aligned)

def plot_beta(stock, market_index, stock_returns, market_returns, beta):
    """Plot scatter plot of stock returns vs market returns"""
    
    plt.figure(figsize=(10, 6))
    plt.scatter(market_returns * 100, stock_returns * 100, alpha=0.5, s=20)
    
    # Plot the beta line
    x_line = np.array([market_returns.min(), market_returns.max()])
    y_line = beta * x_line
    plt.plot(x_line * 100, y_line * 100, 'r-', linewidth=2, 
             label=f'Beta = {beta:.2f}')
    
    plt.xlabel(f'{market_index} Daily Returns (%)')
    plt.ylabel(f'{stock} Daily Returns (%)')
    plt.title(f'Beta Analysis: {stock} vs {market_index}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.tight_layout()
    plt.show()

def interpret_beta(beta):
    """Provide interpretation of beta value"""
    
    if beta > 1:
        print(f"Beta = {beta:.2f} (> 1)")
        print("→ Stock is MORE volatile than the market")
        print("→ Tends to amplify market movements")
        print(f"→ If market moves 1%, stock tends to move {beta:.2f}%")
    elif beta < 1 and beta > 0:
        print(f"Beta = {beta:.2f} (0 < Beta < 1)")
        print("→ Stock is LESS volatile than the market")
        print("→ Tends to dampen market movements")
        print(f"→ If market moves 1%, stock tends to move {beta:.2f}%")
    elif beta < 0:
        print(f"Beta = {beta:.2f} (< 0)")
        print("→ Stock moves OPPOSITE to the market")
        print("→ Rare but possible (e.g., inverse ETFs, gold)")
        print(f"→ If market moves 1%, stock tends to move {beta:.2f}%")
    else:  # beta ≈ 1
        print(f"Beta = {beta:.2f} (≈ 1)")
        print("→ Stock moves WITH the market")
        print("→ Similar volatility to the market")

# User inputs
stock = input("Enter stock symbol (e.g., AAPL, TSLA): ").strip().upper()

market_input = input("Enter market index (or press Enter for S&P 500): ").strip().upper()
market_index = market_input if market_input else "^GSPC"

# User input for date range
start_date = input("Enter start date (YYYY-MM-DD) or press Enter for default (2020-01-01): ").strip()
end_date = input("Enter end date (YYYY-MM-DD) or press Enter for default (2025-06-30): ").strip()

if not start_date:
    start_date = "2020-01-01"
if not end_date:
    end_date = "2025-06-30"

# Calculate beta
result = calculate_beta(stock, market_index, start_date, end_date)

if result[0] is not None:
    beta, correlation, (stock_returns, market_returns) = result
    
    # Display results
    print(f"Stock: {stock}")
    print(f"Market Index: {market_index}")
    print(f"\nBeta: {beta:.3f}")
    print(f"Correlation: {correlation:.3f}")
    
    interpret_beta(beta)
    
    #Note: Beta is calculated using historical data and may change over time. Past volatility doesn't guarantee future volatility.
    
    # Plot the relationship
    plot_beta(stock, market_index, stock_returns, market_returns, beta)