import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf

def time_series_analysis(data, frequency):
    # Create a pandas Series for the time series data
    time_series = pd.Series(data)

    # Plot the original time series
    plt.figure(figsize=(12, 6))
    plt.plot(time_series, marker='o')
    plt.title('Original Time Series')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.show()

    # Decompose the time series into trend, seasonality, and residuals
    decomposition = seasonal_decompose(time_series, model='additive', period=frequency)

    # Plot the decomposed components
    decomposition.plot()
    plt.show()

    # Plot the autocorrelation function (ACF)
    plot_acf(time_series, lags=20)
    plt.title('Autocorrelation Function (ACF)')
    plt.show()

def main():
    # User input for dataset
    sample_size = int(input("Enter the number of time points: "))
    data = list(map(float, input(f"Enter {sample_size} data values (comma-separated): ").split(',')))
    
    if len(data) != sample_size:
        print("Error: The number of data values does not match the specified sample size.")
        return

    frequency = int(input("Enter the frequency of the time series (e.g., 12 for monthly data over a year): "))

    # Perform time series analysis
    time_series_analysis(data, frequency)

if __name__ == "__main__":
    main()
