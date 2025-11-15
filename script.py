import pandas as pd
import numpy as np
from binance.spot import Spot
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def fetch_bitcoin_data():
    """Fetch 10,000 days of Bitcoin OHLCV data from Binance"""
    client = Spot()
    
    # Get daily candles for BTCUSDT - 10,000 days
    klines = client.klines(
        symbol='BTCUSDT',
        interval='1d',
        limit=10000
    )
    
    # Convert to DataFrame
    df = pd.DataFrame(klines, columns=[
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
    
    return df[['date', 'open', 'high', 'low', 'close', 'volume']]

def create_features(df):
    """Create technical indicators and features for classification"""
    df = df.copy()
    
    # Price-based features
    df['price_change'] = df['close'].pct_change()
    df['high_low_ratio'] = df['high'] / df['low']
    df['close_open_ratio'] = df['close'] / df['open']
    
    # Moving averages
    df['sma_5'] = df['close'].rolling(5).mean()
    df['sma_10'] = df['close'].rolling(10).mean()
    df['sma_20'] = df['close'].rolling(20).mean()
    
    # Price relative to moving averages
    df['price_vs_sma5'] = df['close'] / df['sma_5']
    df['price_vs_sma10'] = df['close'] / df['sma_10']
    df['price_vs_sma20'] = df['close'] / df['sma_20']
    
    # Volatility
    df['volatility_5'] = df['price_change'].rolling(5).std()
    df['volatility_10'] = df['price_change'].rolling(10).std()
    
    # Volume features
    df['volume_sma_5'] = df['volume'].rolling(5).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma_5']
    
    # Lagged features
    for lag in [1, 2, 3]:
        df[f'close_lag_{lag}'] = df['close'].shift(lag)
        df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
        df[f'price_change_lag_{lag}'] = df['price_change'].shift(lag)
    
    # Target: Next day price direction (1 = up, 0 = down)
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    
    # Drop NaN values
    df = df.dropna()
    
    return df

def main():
    # Fetch and prepare data
    print("Fetching Bitcoin data...")
    df = fetch_bitcoin_data()
    df = create_features(df)
    
    # Dataset information
    print(f"\nDataset Info:")
    print(f"  Total days: {len(df)}")
    print(f"  Date range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
    print(f"  Price range: ${df['close'].min():.0f} - ${df['close'].max():.0f}")
    print(f"  Features: {len([col for col in df.columns if col not in ['date', 'target', 'close']])}")
    print(f"  Lookback window: 20 days (max)")
    
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
            'model': LogisticRegression(random_state=42),
            'params': 'C=1.0, max_iter=1000'
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
    
    # Train and evaluate models
    results = {}
    
    print("\nModel Parameters:")
    for name, model_info in models.items():
        model = model_info['model']
        params = model_info['params']
        print(f"  {name}: {params}")
    
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