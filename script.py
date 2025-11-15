import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import matplotlib.pyplot as plt

def calculate_technical_indicators(df):
    """
    Calculate various technical indicators from OHLCV data
    """
    # Moving Averages
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['ema_12'] = df['close'].ewm(span=12).mean()
    df['ema_26'] = df['close'].ewm(span=26).mean()
    
    # RSI (Relative Strength Index)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    
    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    
    # ATR (Average True Range)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    true_range = np.maximum(np.maximum(high_low, high_close), low_close)
    df['atr'] = true_range.rolling(window=14).mean()
    
    # Volume indicators
    df['volume_sma'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']
    
    # Price changes and returns
    df['daily_return'] = df['close'].pct_change() * 100
    df['price_change'] = df['close'] - df['open']
    df['price_change_pct'] = (df['price_change'] / df['open']) * 100
    
    return df

def calculate_statistical_metrics(df):
    """
    Calculate statistical metrics from the data
    """
    metrics = {}
    
    # Basic price statistics
    metrics['avg_daily_return'] = df['daily_return'].mean()
    metrics['volatility'] = df['daily_return'].std()
    metrics['sharpe_ratio'] = metrics['avg_daily_return'] / metrics['volatility'] if metrics['volatility'] != 0 else 0
    
    # Risk metrics
    metrics['max_drawdown'] = (df['close'] / df['close'].cummax() - 1).min() * 100
    metrics['var_95'] = df['daily_return'].quantile(0.05)
    
    # Volume statistics
    metrics['avg_volume'] = df['volume'].mean()
    metrics['volume_volatility'] = df['volume'].std()
    
    # Trend metrics
    metrics['uptrend_days'] = len(df[df['price_change'] > 0])
    metrics['downtrend_days'] = len(df[df['price_change'] < 0])
    metrics['trend_ratio'] = metrics['uptrend_days'] / len(df)
    
    return metrics

def generate_trading_signals(df):
    """
    Generate basic trading signals based on technical indicators
    """
    signals = []
    
    for i in range(1, len(df)):
        signal = {
            'date': df.iloc[i]['open_time'],
            'price': df.iloc[i]['close'],
            'signal': 'HOLD',
            'strength': 0
        }
        
        # RSI based signals
        if df.iloc[i]['rsi'] < 30 and df.iloc[i-1]['rsi'] >= 30:
            signal['signal'] = 'BUY'
            signal['strength'] = 1
        elif df.iloc[i]['rsi'] > 70 and df.iloc[i-1]['rsi'] <= 70:
            signal['signal'] = 'SELL'
            signal['strength'] = 1
        
        # Moving average crossover
        if (df.iloc[i]['sma_20'] > df.iloc[i]['sma_50'] and 
            df.iloc[i-1]['sma_20'] <= df.iloc[i-1]['sma_50']):
            if signal['signal'] == 'HOLD':
                signal['signal'] = 'BUY'
                signal['strength'] = max(signal['strength'], 0.5)
        elif (df.iloc[i]['sma_20'] < df.iloc[i]['sma_50'] and 
              df.iloc[i-1]['sma_20'] >= df.iloc[i-1]['sma_50']):
            if signal['signal'] == 'HOLD':
                signal['signal'] = 'SELL'
                signal['strength'] = max(signal['strength'], 0.5)
        
        signals.append(signal)
    
    return pd.DataFrame(signals)

def display_enhanced_analysis(df):
    """
    Display enhanced analysis with technical indicators and statistics
    """
    if df is None or df.empty:
        print("No data to display")
        return
    
    print("=" * 100)
    print(f"ENHANCED OHLCV ANALYSIS FOR BTCUSDT")
    print(f"Data points: {len(df)} | Date range: {df['open_time'].min().strftime('%Y-%m-%d')} to {df['open_time'].max().strftime('%Y-%m-%d')}")
    print("=" * 100)
    
    # Calculate technical indicators
    df = calculate_technical_indicators(df)
    
    # Calculate statistical metrics
    metrics = calculate_statistical_metrics(df)
    
    # Display statistical summary
    print("\n📊 STATISTICAL SUMMARY:")
    print(f"Average Daily Return: {metrics['avg_daily_return']:.4f}%")
    print(f"Daily Volatility: {metrics['volatility']:.4f}%")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
    print(f"Maximum Drawdown: {metrics['max_drawdown']:.2f}%")
    print(f"Value at Risk (95%): {metrics['var_95']:.4f}%")
    print(f"Uptrend Days: {metrics['uptrend_days']} ({metrics['trend_ratio']:.1%})")
    
    # Display technical indicators for recent period
    recent = df.tail(5)
    print("\n🔧 RECENT TECHNICAL INDICATORS:")
    tech_data = recent[['open_time', 'close', 'sma_20', 'rsi', 'macd']].copy()
    tech_data['open_time'] = tech_data['open_time'].dt.strftime('%Y-%m-%d')
    print(tech_data.to_string(index=False, float_format='%.2f'))
    
    # Generate and display trading signals
    signals_df = generate_trading_signals(df)
    recent_signals = signals_df.tail(10)
    
    print("\n🎯 RECENT TRADING SIGNALS:")
    if not recent_signals.empty:
        signal_data = recent_signals[['date', 'price', 'signal', 'strength']].copy()
        signal_data['date'] = signal_data['date'].dt.strftime('%Y-%m-%d')
        print(signal_data.to_string(index=False, float_format='%.2f'))
    else:
        print("No significant signals in recent period")
    
    # Market regime analysis
    volatility_regime = 'HIGH' if metrics['volatility'] > 3 else 'LOW'
    trend_strength = 'STRONG BULL' if metrics['trend_ratio'] > 0.6 else 'STRONG BEAR' if metrics['trend_ratio'] < 0.4 else 'SIDEWAYS'
    
    print(f"\n📈 MARKET REGIME ANALYSIS:")
    print(f"Volatility Regime: {volatility_regime}")
    print(f"Trend Strength: {trend_strength}")
    print(f"Current RSI: {df['rsi'].iloc[-1]:.1f} ({'OVERSOLD' if df['rsi'].iloc[-1] < 30 else 'OVERBOUGHT' if df['rsi'].iloc[-1] > 70 else 'NEUTRAL'})")

def fetch_ohlcv_data(symbol='BTCUSDT', interval='1d', limit=1000, end_time=None):
    """
    Fetch OHLCV data from Binance API
    """
    base_url = 'https://api.binance.com/api/v3/klines'
    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': limit
    }
    
    # Add endTime parameter if provided
    if end_time:
        params['endTime'] = int(end_time.timestamp() * 1000)
    
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
    end_time = datetime.now()
    
    while remaining_points > 0:
        current_limit = min(limit, remaining_points)
        
        print(f"Fetching {current_limit} data points...")
        
        df = fetch_ohlcv_data(symbol, interval, current_limit, end_time)
        
        if df is None or df.empty:
            print("No more data available")
            break
            
        all_data.append(df)
        remaining_points -= len(df)
        
        # Update end_time for next request (go backwards in time)
        # Use the earliest open_time minus 1 day to avoid overlap
        end_time = df['open_time'].min() - timedelta(days=1)
        
        print(f"Fetched {len(df)} points. Remaining: {remaining_points}")
        
        # Rate limiting
        time.sleep(0.1)
    
    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        final_df = final_df.drop_duplicates(subset=['open_time']).sort_values('open_time').reset_index(drop=True)
        return final_df.head(2000) if len(final_df) > 2000 else final_df
    else:
        return None

def main():
    """
    Main function to fetch and display enhanced OHLCV analysis
    """
    print("Fetching 2000 OHLCV data points from Binance for enhanced analysis...")
    
    # Fetch the data
    df = fetch_2000_ohlcv_points()
    
    if df is not None:
        print(f"\n✅ Successfully fetched {len(df)} data points")
        
        # Add symbol column
        df['symbol'] = 'BTCUSDT'
        
        # Display enhanced analysis
        display_enhanced_analysis(df)
        
        # Save enhanced data to CSV
        csv_filename = f'enhanced_binance_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        
        # Calculate technical indicators before saving
        df = calculate_technical_indicators(df)
        df.to_csv(csv_filename, index=False)
        print(f"\n💾 Enhanced data saved to: {csv_filename}")
        
        print("\n🚀 NEXT STEPS:")
        print("1. Use the data for backtesting trading strategies")
        print("2. Build machine learning models for price prediction")
        print("3. Create interactive dashboards for real-time monitoring")
        print("4. Develop risk management systems")
        
    else:
        print("❌ Failed to fetch data from Binance API")

if __name__ == "__main__":
    main()