import pandas as pd
import numpy as np
from binance.spot import Spot
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings
import time
warnings.filterwarnings('ignore')
import yfinance as yf

def fetch_crypto_data_chunked(symbol, hours_to_fetch=1000, start_date=None):
    """Fetch OHLCV data for any cryptocurrency from Binance using chunking - 30-MINUTE DATA"""
    client = Spot()
    
    all_data = []
    chunk_size = 1000  # Binance API limit per request
    
    # Calculate end time (current time or specified start date)
    if start_date:
        # Convert start_date to timestamp in milliseconds
        import datetime
        if isinstance(start_date, str):
            start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        end_time = int(start_date.timestamp() * 1000)
    else:
        end_time = None  # Current time
    
    for i in range(0, hours_to_fetch, chunk_size):
        limit = min(chunk_size, hours_to_fetch - i)
        
        try:
            klines = client.klines(
                symbol=symbol,
                interval='30m',  # 30-minute data
                limit=limit,
                endTime=end_time
            )
            
            if not klines:
                break
                
            all_data.extend(klines)
            
            # Set end_time for next chunk (oldest data)
            end_time = int(klines[0][0]) - 1  # Subtract 1ms from first candle
            
            # Small delay to avoid rate limiting
            time.sleep(0.1)
            
        except Exception as e:
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
    
    return df[['date', 'open', 'high', 'low', 'close', 'volume']]
def fetch_yahoo_data(symbol, periods=1000, start_date='2022-01-01'):
    """Fetch data from Yahoo Finance"""
    try:
        # Download data
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, interval='30m')
        
        # If we don't have enough 30m data, try daily and resample
        if len(df) < periods:
            df = ticker.history(start=start_date, interval='1d')
            # Resample daily to 30m by forward filling
            df = df.resample('30T').ffill()
        
        # Ensure we have the required columns
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        df.columns = ['open', 'high', 'low', 'close', 'volume']
        df = df.reset_index()
        df = df.rename(columns={'Date': 'date'})
        
        # Take the last 'periods' rows
        df = df.tail(periods).reset_index(drop=True)
        
        return df
        
    except Exception as e:
        # Return empty DataFrame with required columns
        return pd.DataFrame(columns=['date', 'open', 'high', 'low', 'close', 'volume'])

def calculate_macd(df, fast_period=12, slow_period=26, signal_period=9):
    """Calculate MACD indicator"""
    df_copy = df.copy()
    
    # Calculate EMAs
    ema_fast = df_copy['close'].ewm(span=fast_period, adjust=False).mean()
    ema_slow = df_copy['close'].ewm(span=slow_period, adjust=False).mean()
    
    # Calculate MACD line
    macd_line = ema_fast - ema_slow
    
    # Calculate Signal line
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    
    # Calculate MACD Histogram (MACD - Signal)
    macd_histogram = macd_line - signal_line
    
    return macd_line, signal_line, macd_histogram

def create_features_with_altcoins(btc_df, eth_df, xrp_df, ada_df):
    """Create technical indicators and features including altcoin data - ONLY DERIVATIVES - 30-MINUTE ADJUSTED"""
    df = btc_df.copy()
    
    # Price-based derivative features for Bitcoin (NO RAW PRICES)
    df['price_change'] = df['close'].pct_change()
    df['high_low_ratio'] = df['high'] / df['low']
    df['close_open_ratio'] = df['close'] / df['open']
    
    # Moving averages (adjusted for 30-minute data - 24h and 48h windows)
    df['sma_24'] = df['close'].rolling(48).mean()  # 24 hours (48 * 30min)
    df['sma_48'] = df['close'].rolling(96).mean()  # 48 hours (96 * 30min)
    
    # Price relative to moving averages (derivatives)
    df['price_vs_sma24'] = df['close'] / df['sma_24']
    df['price_vs_sma48'] = df['close'] / df['sma_48']
    
    # Volatility (adjusted for 30-minute data)
    df['volatility_24'] = df['price_change'].rolling(48).std()  # 24 hours
    df['volatility_48'] = df['price_change'].rolling(96).std()  # 48 hours
    
    # Volume features (derivatives only)
    df['volume_sma_24'] = df['volume'].rolling(48).mean()  # 24 hours
    df['volume_ratio'] = df['volume'] / df['volume_sma_24']
    
    # MACD features for Bitcoin
    macd_line, signal_line, macd_histogram = calculate_macd(df)
    df['macd_line'] = macd_line
    df['macd_signal'] = signal_line
    df['macd_histogram'] = macd_histogram
    
    # Add altcoin features (derivatives only)
    # Ethereum features
    df['eth_price_change'] = eth_df['close'].pct_change()
    df['eth_volume_ratio'] = eth_df['volume'] / eth_df['volume'].rolling(48).mean()
    df['btc_eth_ratio'] = df['close'] / eth_df['close']  # BTC dominance vs ETH
    
    # Ripple features
    df['xrp_price_change'] = xrp_df['close'].pct_change()
    df['xrp_volume_ratio'] = xrp_df['volume'] / xrp_df['volume'].rolling(48).mean()
    df['btc_xrp_ratio'] = df['close'] / xrp_df['close']  # BTC dominance vs XRP
    
    # Cardano features
    df['ada_price_change'] = ada_df['close'].pct_change()
    df['ada_volume_ratio'] = ada_df['volume'] / ada_df['volume'].rolling(48).mean()
    df['btc_ada_ratio'] = df['close'] / ada_df['close']  # BTC dominance vs ADA
    
    # Altcoin momentum indicators
    df['altcoin_momentum'] = (df['eth_price_change'] + df['xrp_price_change'] + df['ada_price_change']) / 3
    df['altcoin_volume_strength'] = (df['eth_volume_ratio'] + df['xrp_volume_ratio'] + df['ada_volume_ratio']) / 3
    
    # Target: Next period price direction (1 = up, 0 = down)
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    
    # Drop NaN values
    df = df.dropna()
    
    return df

def add_polynomial_features(X):
    """Add squared and cubic versions of key features to capture non-linear relationships"""
    X_poly = X.copy()
    
    # Select key features for polynomial expansion
    features_for_poly = [
        'price_change', 'volatility_24', 'volatility_48', 
        'eth_price_change', 'xrp_price_change', 'ada_price_change',
        'altcoin_momentum', 'macd_histogram'
    ]
    # Add squared features
    for feature in features_for_poly:
        if feature in X.columns:
            squared_feature_name = f"{feature}_squared"
            X_poly[squared_feature_name] = X[feature] ** 2
    
    # Add cubic features
    for feature in features_for_poly:
        if feature in X.columns:
            cubic_feature_name = f"{feature}_cubed"
            X_poly[cubic_feature_name] = X[feature] ** 3
    
    return X_poly

def add_lagged_features_selected(X):
    """Add lagged versions of SELECTED features with specific periods: [1, 3, 12] periods"""
    X_lagged = X.copy()
    
    # Selected features for lagging - added price_change and price_vs_sma24
    selected_features = [
        'price_change', 'price_vs_sma24', 'volume_ratio', 'volatility_24', 'volatility_48',
        'eth_price_change', 'xrp_price_change', 'ada_price_change',
        'altcoin_momentum', 'macd_histogram'
    ]
    # Lag periods - removed 24
    lag_periods = [1, 3, 12]  # 30-minute periods
    
    # Add lagged features for selected features
    lagged_features_added = []
    for feature in selected_features:
        if feature in X.columns:
            for lag in lag_periods:
                lagged_feature_name = f"{feature}_lag_{lag}"
                X_lagged[lagged_feature_name] = X[feature].shift(lag)
                lagged_features_added.append(lagged_feature_name)
    
    # Drop rows with NaN values from lagging
    X_lagged = X_lagged.iloc[12:]
    
    return X_lagged, lagged_features_added

def normalize_features(X):
    """Normalize features based on their value ranges"""
    X_normalized = X.copy()
    
    # Features that can be negative (normalize to [-1, 1])
    negative_features = [
        'price_change', 'eth_price_change', 'xrp_price_change', 'ada_price_change', 
        'altcoin_momentum', 'macd_line', 'macd_signal', 'macd_histogram'
    ]
    # Features that are always positive (normalize to [0, 1])
    positive_features = [
        'high_low_ratio', 'close_open_ratio',
        'sma_24', 'sma_48', 'price_vs_sma24', 'price_vs_sma48',
        'volatility_24', 'volatility_48',
        'volume_sma_24', 'volume_ratio',
        'eth_volume_ratio', 'btc_eth_ratio',
        'xrp_volume_ratio', 'btc_xrp_ratio',
        'ada_volume_ratio', 'btc_ada_ratio',
        'altcoin_volume_strength'
    ]
    # Add polynomial features to appropriate categories
    poly_features = [col for col in X.columns if col.endswith('_squared') or col.endswith('_cubed')]
    for feature in poly_features:
        base_feature = feature.replace('_squared', '').replace('_cubed', '')
        if base_feature in negative_features:
            # Squared/cubed negative features become positive
            positive_features.append(feature)
        elif base_feature in positive_features:
            positive_features.append(feature)
    
    # Add lagged features to appropriate categories
    lagged_features = [col for col in X.columns if '_lag_' in col]
    for feature in lagged_features:
        base_feature = feature.split('_lag_')[0]
        if base_feature in negative_features:
            negative_features.append(feature)
        elif base_feature in positive_features:
            positive_features.append(feature)
    
    # Normalize negative features to [-1, 1]
    for feature in negative_features:
        if feature in X.columns:
            scaler = MinMaxScaler(feature_range=(-1, 1))
            X_normalized[feature] = scaler.fit_transform(X[[feature]])
    
    # Normalize positive features to [0, 1]
    for feature in positive_features:
        if feature in X.columns:
            scaler = MinMaxScaler(feature_range=(0, 1))
            X_normalized[feature] = scaler.fit_transform(X[[feature]])
    
    return X_normalized

def main():
    # Set start date to None to fetch data up to current time
    start_date = None
    # Fetch Bitcoin data
    # Fetch Bitcoin data
    btc_df = fetch_crypto_data_chunked('BTCUSDT', 1000, start_date)
    
    # Fetch altcoin data
    eth_df = fetch_crypto_data_chunked('ETHUSDT', 1000, start_date)
    xrp_df = fetch_crypto_data_chunked('XRPUSDT', 1000, start_date)
    ada_df = fetch_crypto_data_chunked('ADAUSDT', 1000, start_date)
    
    # Create features with altcoin data
    df = create_features_with_altcoins(btc_df, eth_df, xrp_df, ada_df)
    # Dataset information - simplified
    print(f"\nDataset Info:")
    print(f"  Total periods: {len(df)}")
    print(f"  Date range: {df['date'].min().strftime('%Y-%m-%d %H:%M')} to {df['date'].max().strftime('%Y-%m-%d %H:%M')}")
    print(f"  BTC price range: ${df['close'].min():.0f} - ${df['close'].max():.0f}")
    
    # Prepare features and target (EXCLUDE RAW PRICE DATA)
    exclude_cols = ['date', 'target', 'close', 'open', 'high', 'low', 'volume']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    X = df[feature_cols]
    y = df['target']
    
    # Add polynomial features (squared and cubed)
    X_with_poly = add_polynomial_features(X)
    
    # Add lagged features for SELECTED features with specific periods
    X_with_lags, lagged_features_added = add_lagged_features_selected(X_with_poly)
    
    # Update target to match lagged features (drop first 12 rows - maximum lag)
    y_lagged = y.iloc[12:]
    
    # Normalize features
    X_normalized = normalize_features(X_with_lags)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_normalized, y_lagged, test_size=0.2, random_state=42, shuffle=False
    )
    
    # Scale features (additional standardization for models that need it)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Scale features (additional standardization for models that need it)
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
            'model': RandomForestClassifier(n_estimators=700, random_state=42),  # Changed to 700
            'params': 'n_estimators=700, max_depth=None'
        },
        'Gradient Boosting': {
            'model': GradientBoostingClassifier(n_estimators=700, learning_rate=0.05, random_state=42),  # Changed to 700
            'params': 'n_estimators=700, learning_rate=0.05'
        }
    }
    print(f"\nModel Hyperparameters:")
    for name, model_info in models.items():
        print(f"  {name}: {model_info['params']}")
    
    # Train and evaluate models
    results = {}
    
    results = {}
    # Train and evaluate models
    results = {}
    
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
        results[name] = {'accuracy': accuracy, 'f1_score': f1, 'model': model, 'predictions': y_pred}
    # Find best model by F1 score
    best_model = max(results, key=lambda x: results[x]['f1_score'])
    print(f"\nBest Model: {best_model}")
    
    # Compare all models
    print("\n=== SUMMARY ===")
    print(f"{'Model':<20} {'Accuracy':<10} {'F1':<10}")
    print("-" * 40)
    
    for name in ['Logistic Regression', 'Random Forest', 'Gradient Boosting']:
        print(f"{name:<20} {results[name]['accuracy']:.4f}    {results[name]['f1_score']:.4f}")
    
    print("\n=== COMPLETE ===")

if __name__ == "__main__":
    main()