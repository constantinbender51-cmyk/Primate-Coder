import yfinance as yf
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import os
def fetch_financial_data():
    """
    Fetch 60 days of hourly data for Bitcoin, HSI, S&P 500, and Gold
    """
    print("Starting data fetch from Yahoo Finance...")
    
    # Define tickers
    tickers = {
        'BTC-USD': 'Bitcoin',
        '^HSI': 'Hang Seng Index',
        '^GSPC': 'S&P 500',
        'GC=F': 'Gold Futures'
    }
    
    # Calculate date range (last 60 days)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=60)
    
    print(f"Fetching data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print("=" * 60)
    
    all_data = {}
    
    for ticker, name in tickers.items():
        print(f"Fetching {name} ({ticker})...")
        
        try:
            # Fetch data with 1-hour interval
            data = yf.download(
                ticker, 
                start=start_date, 
                end=end_date, 
                interval='1h',
                progress=False
            )
            
            if data.empty:
                print(f"  ⚠️  No data returned for {ticker}")
                continue
            
            # Add ticker name and symbol as columns
            data['Ticker'] = ticker
            data['Asset'] = name
            
            # Reset index to get datetime as a column
            data = data.reset_index()
            
            all_data[ticker] = data
            print(f"  ✅ Successfully fetched {len(data)} hourly records")
            
            # Rate limiting - wait 1 second between requests
            time.sleep(1)
            
        except Exception as e:
            print(f"  ❌ Error fetching {ticker}: {e}")
            continue
    
    return all_data

def calculate_technical_indicators(all_data):
    """
    Calculate 4-hour SMA and EMA for each asset
    """
    print("\n" + "=" * 60)
    print("CALCULATING TECHNICAL INDICATORS")
    print("=" * 60)
    
    for ticker, data in all_data.items():
        print(f"Calculating indicators for {ticker}...")
        
        # Sort by datetime to ensure proper calculation
        data = data.sort_values('Datetime')
        
        # Calculate 4-hour Simple Moving Average (SMA)
        data['SMA_4H'] = data['Close'].rolling(window=4, min_periods=1).mean()
        
        # Calculate 4-hour Exponential Moving Average (EMA)
        data['EMA_4H'] = data['Close'].ewm(span=4, adjust=False).mean()
        
        # Calculate SMA/EMA crossover signals
        data['SMA_EMA_Crossover'] = np.where(data['SMA_4H'] > data['EMA_4H'], 1, -1)
        
        # Update the data in the dictionary
        all_data[ticker] = data
        
        print(f"  ✅ Added 4-hour SMA and EMA for {ticker}")
    
    return all_data

def save_data_to_csv(all_data):
    """
    Save each dataset to individual CSV files
    """
    print("\n" + "=" * 60)
    print("Saving data to CSV files...")
    
    # Create data directory if it doesn't exist
    if not os.path.exists('financial_data'):
        os.makedirs('financial_data')
    
    for ticker, data in all_data.items():
        filename = f"financial_data/{ticker.replace('^', '').replace('=', '').replace('-', '_')}_hourly.csv"
        data.to_csv(filename, index=False)
        print(f"  ✅ Saved {ticker} data to {filename}")
    
    # Also create a combined dataset
    if all_data:
        combined_data = pd.concat(all_data.values(), ignore_index=True)
        combined_filename = "financial_data/combined_hourly_data.csv"
        combined_data.to_csv(combined_filename, index=False)
        print(f"  ✅ Saved combined data to {combined_filename}")

def display_summary_statistics(all_data):
    """
    Display summary statistics for each asset
    """
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    
    for ticker, data in all_data.items():
        print(f"\n📊 {ticker} - {data['Asset'].iloc[0]}")
        print(f"   Period: {data['Datetime'].min()} to {data['Datetime'].max()}")
        print(f"   Total records: {len(data):,}")
        
        # Extract min and max as scalar values
        # Extract min and max as scalar values
        close_min = float(data['Close'].min())
        close_max = float(data['Close'].max())
        latest_close = float(data['Close'].iloc[-1])
        
        print(f"   Price range: ${close_min:.2f} - ${close_max:.2f}")
        print(f"   Latest close: ${latest_close:.2f}")
        
        if len(data) > 1:
            first_close = float(data['Close'].iloc[0])
            price_change = ((latest_close - first_close) / first_close) * 100
            print(f"   Total return: {price_change:+.2f}%")
        
        avg_volume = float(data['Volume'].mean())
        print(f"   Average volume: {avg_volume:,.0f}")
        
        # Display technical indicators
        # Display technical indicators
        latest_sma = float(data['SMA_4H'].iloc[-1])
        latest_ema = float(data['EMA_4H'].iloc[-1])
        crossover_signal = int(data['SMA_EMA_Crossover'].iloc[-1])
        
        print(f"   Latest 4H SMA: ${latest_sma:.2f}")
        print(f"   Latest 4H EMA: ${latest_ema:.2f}")
        
        if crossover_signal == 1:
            print(f"   SMA/EMA Signal: 📈 SMA above EMA (Bullish)")
        elif crossover_signal == -1:
            print(f"   SMA/EMA Signal: 📉 EMA above SMA (Bearish)")
        else:
            print(f"   SMA/EMA Signal: ➖ Neutral")
            print(f"   SMA/EMA Signal: ➖ Neutral")
    """
    Main function to orchestrate the data fetching process
    """
    print("🚀 FINANCIAL DATA FETCHER")
    print("Fetching 60 days of hourly data for:")
    print("  • Bitcoin (BTC-USD)")
    print("  • Hang Seng Index (^HSI)")
    print("  • S&P 500 (^GSPC)")
    print("  • Gold Futures (GC=F)")
    print("=" * 60)
    
    try:
        # Fetch data
        all_data = fetch_financial_data()
        
        if not all_data:
            print("❌ No data was successfully fetched. Please check your internet connection and try again.")
            return
        
        # Calculate technical indicators
        all_data = calculate_technical_indicators(all_data)
        
        # Save data
        save_data_to_csv(all_data)
        
        # Display summary statistics
        display_summary_statistics(all_data)
        print("✅ DATA FETCH COMPLETED SUCCESSFULLY!")
        print("All data has been saved to the 'financial_data' directory")
        
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()