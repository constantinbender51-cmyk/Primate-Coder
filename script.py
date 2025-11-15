import pandas as pd
import numpy as np
from binance.spot import Spot
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
import time
warnings.filterwarnings('ignore')

def fetch_crypto_data_chunked(symbol, days_to_fetch=10000):
    """Fetch OHLCV data for any cryptocurrency from Binance using chunking"""
    client = Spot()
    
    all_data = []
    chunk_size = 1000  # Binance API limit per request
    
    # Calculate end time (current time)
    end_time = None
    
    for i in range(0, days_to_fetch, chunk_size):
        limit = min(chunk_size, days_to_fetch - i)
        
        try:
            klines = client.klines(
                symbol=symbol,
                interval='1d',
                limit=limit,
                endTime=end_time
            )
            
            if not klines:
                break
                
            all_data.extend(klines)
            
            # Set end_time for next chunk (oldest data)
            end_time = int(klines[0][0]) - 1  # Subtract 1ms from first candle
            
            print(f"Fetched {len(klines)} days for {symbol}, total: {len(all_data)} days")
            
            # Small delay to avoid rate limiting
            time.sleep(0.1)
            
        except Exception as e:
            print(f"Error fetching {symbol} chunk: {e}")
            break
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])
    
    # Convert to numeric types
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col])
    
    # Convert timestamp to datetime
    df['date'] = pd.to_datetime(df['open_time'], unit='ms')
    
    # Sort by date (oldest first)
    df = df.sort_values('date').reset_index(drop=True)
    
    return df[['date', 'open', 'high', 'low', 'close', 'volume']]

def create_features_with_altcoins(btc_df, eth_df, xrp_df, ada_df):
    """Create technical indicators and features including altcoin data"""
    df = btc_df.copy()
    
    # Price-based features for Bitcoin
    df['price_change'] = df['close'].pct_change()
    df['high_low_ratio'] = df['high'] / df['low']
    df['close_open_ratio'] = df['close'] / df['open']
    
    # Moving averages (10-day max lookback)
    df['sma_5'] = df['close'].rolling(5).mean()
    df['sma_10'] = df['close'].rolling(10).mean()
    
    # Price relative to moving averages
    df['price_vs_sma5'] = df['close'] / df['sma_5']
    df['price_vs_sma10'] = df['close'] / df['sma_10']
    
    # Volatility (10-day max lookback)
    df['volatility_5'] = df['price_change'].rolling(5).std()
    df['volatility_10'] = df['price_change'].rolling(10).std()
    
    # Volume features
    df['volume_sma_5'] = df['volume'].rolling(5).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma_5']
    
    # Lagged features (3 lags for 10-day window)
    for lag in [1, 2, 3]:
        df[f'close_lag_{lag}'] = df['close'].shift(lag)
        df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
        df[f'price_change_lag_{lag}'] = df['price_change'].shift(lag)
    
    # Add altcoin features
    # Ethereum features
    df['eth_close'] = eth_df['close']
    df['eth_price_change'] = eth_df['close'].pct_change()
    df['eth_volume_ratio'] = eth_df['volume'] / eth_df['volume'].rolling(5).mean()
    df['btc_eth_ratio'] = df['close'] / eth_df['close']  # BTC dominance vs ETH
    
    # Ripple features
    df['xrp_close'] = xrp_df['close']
    df['xrp_price_change'] = xrp_df['close'].pct_change()
    df['xrp_volume_ratio'] = xrp_df['volume'] / xrp_df['volume'].rolling(5).mean()
    df['btc_xrp_ratio'] = df['close'] / xrp_df['close']  # BTC dominance vs XRP
    
    # Cardano features
    df['ada_close'] = ada_df['close']
    df['ada_price_change'] = ada_df['close'].pct_change()
    df['ada_volume_ratio'] = ada_df['volume'] / ada_df['volume'].rolling(5).mean()
    df['btc_ada_ratio'] = df['close'] / ada_df['close']  # BTC dominance vs ADA
    
    # Altcoin momentum indicators
    df['altcoin_momentum'] = (df['eth_price_change'] + df['xrp_price_change'] + df['ada_price_change']) / 3
    df['altcoin_volume_strength'] = (df['eth_volume_ratio'] + df['xrp_volume_ratio'] + df['ada_volume_ratio']) / 3
    
    # Target: Next day price direction (1 = up, 0 = down)
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    
    # Drop NaN values
    df = df.dropna()
    
    return df

def main():
    # Fetch Bitcoin data
    print("Fetching 10,000 days of Bitcoin data...")
    btc_df = fetch_crypto_data_chunked('BTCUSDT', 10000)
    
    # Fetch altcoin data
    print("\nFetching Ethereum data...")
    eth_df = fetch_crypto_data_chunked('ETHUSDT', 10000)
    
    print("\nFetching Ripple data...")
    xrp_df = fetch_crypto_data_chunked('XRPUSDT', 10000)
    
    print("\nFetching Cardano data...")
    ada_df = fetch_crypto_data_chunked('ADAUSDT', 10000)
    
    # Create features with altcoin data
    df = create_features_with_altcoins(btc_df, eth_df, xrp_df, ada_df)
    
    # Dataset information
    print(f"\nDataset Info:")
    print(f"  Total days: {len(df)}")
    print(f"  Date range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
    print(f"  BTC price range: ${df['close'].min():.0f} - ${df['close'].max():.0f}")
    print(f"  ETH price range: ${df['eth_close'].min():.2f} - ${df['eth_close'].max():.2f}")
    print(f"  XRP price range: ${df['xrp_close'].min():.4f} - ${df['xrp_close'].max():.4f}")
    print(f"  ADA price range: ${df['ada_close'].min():.4f} - ${df['ada_close'].max():.4f}")
    print(f"  Features: {len([col for col in df.columns if col not in ['date', 'target', 'close']])}")
    print(f"  Lookback window: 10 days (max)")
    
    # Prepare features and target
    feature_cols = [col for col in df.columns if col not in ['date', 'target', 'close']]
    X = df[feature_cols]
    y = df['target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )
    
    print(f"  Train samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize classification models with parameters
    models = {
        'Logistic Regression': {
            'model': LogisticRegression(random_state=42, max_iter=5000),
            'params': 'C=1.0, max_iter=5000'
        },
        'Random Forest': {
            'model': RandomForestClassifier(n_estimators=100, random_state=42),
            'params': 'n_estimators=100, max_depth=None'
        },
        'Gradient Boosting': {
            'model': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'params': 'n_estimators=100, learning_rate=0.1'
        }
    }
    
    # Print model hyperparameters
    print(f"\nModel Hyperparameters:")
    for name, model_info in models.items():
        print(f"  {name}: {model_info['params']}")
    
    # Train and evaluate models
    results = {}
    
    print("\nModel Performance:")
    for name, model_info in models.items():
        model = model_info['model']
        
        # Train model
        if name == 'Logistic Regression':
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        results[name] = {'accuracy': accuracy, 'f1_score': f1}
        
        print(f"{name}:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1-Score: {f1:.4f}")
    
    # Find best model by F1 score
    best_model = max(results, key=lambda x: results[x]['f1_score'])
    print(f"\nBest Model: {best_model}")

if __name__ == "__main__":
    main()