import requests
import pandas as pd
from datetime import datetime, timedelta
import time

def fetch_ohlcv_data(symbol='BTCUSDT', interval='1d', limit=1000):
    """
    Fetch OHLCV data from Binance API
    """
    base_url = 'https://api.binance.com/api/v3/klines'
    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': limit
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Convert to DataFrame
        df = pd.DataFrame(data, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Convert timestamp to datetime
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        
        # Convert price and volume columns to numeric
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume']
        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric)
        
        return df
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None

def fetch_2000_ohlcv_points(symbol='BTCUSDT', interval='1d'):
    """
    Fetch 2000 OHLCV data points by making multiple API calls
    """
    all_data = []
    limit = 1000  # Maximum limit per request
    remaining_points = 2000
    
    # Start from current time and go backwards
    end_time = None
    
    while remaining_points > 0:
        current_limit = min(limit, remaining_points)
        
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': current_limit
        }
        
        # Add endTime parameter for pagination
        if end_time:
            params['endTime'] = int(end_time.timestamp() * 1000)
        
        df = fetch_ohlcv_data(symbol, interval, current_limit)
        
        if df is None or df.empty:
            break
            
        all_data.append(df)
        remaining_points -= len(df)
        
        # Update end_time for next request (go backwards in time)
        end_time = df['open_time'].min() - timedelta(days=1)
        
        # Rate limiting
        time.sleep(0.1)
    
    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        final_df = final_df.drop_duplicates(subset=['open_time']).sort_values('open_time').reset_index(drop=True)
        return final_df.head(2000)
    else:
        return None

def display_ohlcv_data(df):
    """
    Display OHLCV data in a formatted way
    """
    if df is None or df.empty:
        print("No data to display")
        return
    
    print("=" * 100)
    print(f"OHLCV DATA FOR {df['symbol'].iloc[0] if 'symbol' in df.columns else 'BTCUSDT'} - DAILY TIMEFRAME")
    print(f"Total data points: {len(df)}")
    print("=" * 100)
    
    # Display summary statistics
    print("\nSUMMARY STATISTICS:")
    print(f"Date range: {df['open_time'].min().strftime('%Y-%m-%d')} to {df['open_time'].max().strftime('%Y-%m-%d')}")
    print(f"Total days: {(df['open_time'].max() - df['open_time'].min()).days}")
    
    # Display price statistics
    print("\nPRICE STATISTICS:")
    print(f"Highest price: ${df['high'].max():.2f}")
    print(f"Lowest price: ${df['low'].min():.2f}")
    print(f"Current price: ${df['close'].iloc[-1]:.2f}")
    
    # Display recent data
    print("\nRECENT 10 DATA POINTS:")
    recent_data = df.tail(10)[['open_time', 'open', 'high', 'low', 'close', 'volume']].copy()
    recent_data['open_time'] = recent_data['open_time'].dt.strftime('%Y-%m-%d')
    print(recent_data.to_string(index=False, float_format='%.2f'))
    
    # Display oldest data
    print("\nOLDEST 10 DATA POINTS:")
    oldest_data = df.head(10)[['open_time', 'open', 'high', 'low', 'close', 'volume']].copy()
    oldest_data['open_time'] = oldest_data['open_time'].dt.strftime('%Y-%m-%d')
    print(oldest_data.to_string(index=False, float_format='%.2f'))

def main():
    """
    Main function to fetch and display OHLCV data
    """
    print("Fetching 2000 OHLCV data points from Binance...")
    
    # Fetch the data
    df = fetch_2000_ohlcv_points()
    
    if df is not None:
        print(f"Successfully fetched {len(df)} data points")
        
        # Add symbol column for display
        df['symbol'] = 'BTCUSDT'
        
        # Display the data
        display_ohlcv_data(df)
        
        # Save to CSV for reference
        csv_filename = f'binance_ohlcv_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        df.to_csv(csv_filename, index=False)
        print(f"\nData saved to: {csv_filename}")
        
    else:
        print("Failed to fetch data from Binance API")

if __name__ == "__main__":
    main()