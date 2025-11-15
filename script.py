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
    """Fetch 1000 days of Bitcoin OHLCV data from Binance"""
    client = Spot()
    
    # Get daily candles for BTCUSDT
    klines = client.klines(
        symbol='BTCUSDT',
        interval='1d',
        limit=1000
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
    
    # Prepare features and target
    feature_cols = [col for col in df.columns if col not in ['date', 'target', 'close']]
    X = df[feature_cols]
    y = df['target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize classification models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    
    # Train and evaluate models
    results = {}
    
    for name, model in models.items():
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
    
    # Minimalistic output
    print("\nModel Performance:")
    for model_name, metrics in results.items():
        print(f"{model_name}:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1-Score: {metrics['f1_score']:.4f}")
    
    # Find best model by F1 score
    best_model = max(results, key=lambda x: results[x]['f1_score'])
    print(f"\nBest Model: {best_model}")

if __name__ == "__main__":
    main()