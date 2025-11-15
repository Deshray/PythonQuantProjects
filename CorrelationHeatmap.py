import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Values close to +1 = stocks move together
#Values close to -1 = stocks move opposite
#Values close to 0 = no clear relationship

# User input for stock symbols
stocks_input = input("Enter stock symbols (comma-separated, e.g., 'AAPL, MSFT, GOOGL, TSLA'): ")
stocks = [stock.strip().upper() for stock in stocks_input.split(',')]

if len(stocks) < 2:
    print("Error: Please enter at least 2 stocks to calculate correlations.")
else:
    # User input for date range
    start_date = input("Enter start date (YYYY-MM-DD) or press Enter for default (2020-01-01): ").strip()
    end_date = input("Enter end date (YYYY-MM-DD) or press Enter for default (2025-06-30): ").strip()
    
    if not start_date:
        start_date = "2020-01-01"
    if not end_date:
        end_date = "2025-06-30"
    
    # Fetch stock data
    data = yf.download(stocks, start=start_date, end=end_date, auto_adjust=True)['Close']
    
    # Handle case where only one stock is successfully downloaded
    if data.empty:
        print("Error: Could not fetch data for the specified stocks.")
    else:
        # Calculate daily returns
        returns = data.pct_change().dropna()
        
        # Calculate correlation matrix
        correlation_matrix = returns.corr()
        
        # Create the heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, 
                    annot=True,  # Show correlation values
                    cmap='RdYlGn',  # Red for negative, yellow for neutral, green for positive
                    center=0,  # Center the colormap at 0
                    vmin=-1, 
                    vmax=1,
                    square=True,
                    linewidths=1,
                    cbar_kws={"shrink": 0.8})
        
        plt.title('Stock Correlation Matrix\n(Based on Daily Returns)', fontsize=14, pad=20)
        plt.tight_layout()
        plt.show()
        
        # Find highest and lowest correlations (excluding diagonal)
        for i in range(len(stocks)):
            for j in range(i+1, len(stocks)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:
                    relationship = "strongly positive" if corr_value > 0 else "strongly negative"
                    print(f"{stocks[i]} and {stocks[j]}: {corr_value:.3f} ({relationship})")