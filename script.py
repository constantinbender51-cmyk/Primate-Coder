import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

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
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
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
    
    # Additional features for ML
    df['high_low_ratio'] = df['high'] / df['low']
    df['open_close_ratio'] = df['open'] / df['close']
    df['volatility'] = df['daily_return'].rolling(window=5).std()
    
    return df

def prepare_ml_data(df, prediction_horizon=1):
    """
    Prepare data for machine learning - predict if price will go up in next N days
    """
    # Create target variable: 1 if price increases in next N days, 0 otherwise
    df['future_return'] = df['close'].shift(-prediction_horizon) / df['close'] - 1
    df['target'] = (df['future_return'] > 0).astype(int)
    
    # Feature selection
    feature_columns = [
        'open', 'high', 'low', 'close', 'volume',
        'sma_20', 'sma_50', 'ema_12', 'ema_26',
        'rsi', 'macd', 'macd_signal', 'macd_histogram',
        'bb_position', 'atr', 'volume_ratio',
        'daily_return', 'volatility', 'high_low_ratio', 'open_close_ratio'
    ]
    
    # Remove rows with NaN values
    df_clean = df.dropna(subset=feature_columns + ['target'])
    
    X = df_clean[feature_columns]
    y = df_clean['target']
    
    return X, y, df_clean

def train_ml_models(X, y):
    """
    Train multiple machine learning models and compare performance
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', random_state=42)
    }
    
    results = {}
    
    print("🤖 TRAINING MACHINE LEARNING MODELS")
    print("=" * 60)
    
    for name, model in models.items():
        print(f"\n📊 Training {name}...")
        
        # Use scaled data for linear models, original for tree-based
        if name in ['Logistic Regression', 'SVM']:
            X_tr = X_train_scaled
            X_te = X_test_scaled
        else:
            X_tr = X_train
            X_te = X_test
        
        # Train model
        model.fit(X_tr, y_train)
        
        # Predictions
        y_pred = model.predict(X_te)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        cv_scores = cross_val_score(model, X_tr, y_train, cv=5, scoring='accuracy')
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'predictions': y_pred,
            'scaler': scaler if name in ['Logistic Regression', 'SVM'] else None
        }
        
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   Cross-validation: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return results, X_test, y_test, scaler

def analyze_feature_importance(results, feature_names):
    """
    Analyze feature importance from the best performing model
    """
    # Get the best model (highest accuracy)
    best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
    best_model = results[best_model_name]['model']
    
    print(f"\n🔍 FEATURE IMPORTANCE ANALYSIS (Best Model: {best_model_name})")
    print("=" * 60)
    
    if hasattr(best_model, 'feature_importances_'):
        # Tree-based models
        importances = best_model.feature_importances_
        feature_imp_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(feature_imp_df.head(10).to_string(index=False))
        
        return feature_imp_df
    elif hasattr(best_model, 'coef_'):
        # Linear models
        coefficients = best_model.coef_[0]
        feature_imp_df = pd.DataFrame({
            'feature': feature_names,
            'coefficient': coefficients
        }).sort_values('coefficient', key=abs, ascending=False)
        
        print("\nTop 10 Most Influential Features (by absolute coefficient):")
        print(feature_imp_df.head(10).to_string(index=False))
        
        return feature_imp_df
    else:
        print("Feature importance not available for this model type")
        return None

def make_predictions(results, df, feature_columns, prediction_horizon=1):
    """
    Make predictions for the most recent data
    """
    print(f"\n🎯 PREDICTIONS FOR NEXT {prediction_horizon} DAY(S)")
    print("=" * 60)
    
    # Get the best model
    best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
    best_result = results[best_model_name]
    
    # Prepare recent data for prediction
    recent_data = df.tail(10).copy()
    
    predictions = []
    
    for i, (idx, row) in enumerate(recent_data.iterrows()):
        # Prepare features
        features = row[feature_columns].values.reshape(1, -1)
        
        # Scale if needed
        if best_result['scaler'] is not None:
            features = best_result['scaler'].transform(features)
        
        # Make prediction
        prediction = best_result['model'].predict(features)[0]
        probability = best_result['model'].predict_proba(features)[0]
        
        predictions.append({
            'date': row['open_time'].strftime('%Y-%m-%d'),
            'price': row['close'],
            'prediction': 'UP' if prediction == 1 else 'DOWN',
            'confidence': max(probability),
            'up_probability': probability[1],
            'down_probability': probability[0]
        })
    
    # Display predictions
    pred_df = pd.DataFrame(predictions)
    print(f"\nUsing {best_model_name} (Accuracy: {best_result['accuracy']:.4f})")
    print("\nRecent Predictions:")
    display_cols = ['date', 'price', 'prediction', 'confidence', 'up_probability']
    print(pred_df[display_cols].to_string(index=False, float_format='%.4f'))
    
    return pred_df

def display_model_comparison(results):
    """
    Display comparison of all trained models
    """
    print("\n🏆 MODEL PERFORMANCE COMPARISON")
    print("=" * 60)
    
    comparison_data = []
    for name, result in results.items():
        comparison_data.append({
            'Model': name,
            'Accuracy': result['accuracy'],
            'CV Score': result['cv_mean'],
            'CV Std': result['cv_std']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False, float_format='%.4f'))
    
    # Highlight best model
    best_model = comparison_df.loc[comparison_df['Accuracy'].idxmax()]
    print(f"\n🏅 BEST MODEL: {best_model['Model']} (Accuracy: {best_model['Accuracy']:.4f})")

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
    
    if end_time:
        params['endTime'] = int(end_time.timestamp() * 1000)
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        df = pd.DataFrame(data, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        
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
    limit = 1000
    remaining_points = 2000
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
        end_time = df['open_time'].min() - timedelta(days=1)
        
        print(f"Fetched {len(df)} points. Remaining: {remaining_points}")
        time.sleep(0.1)
    
    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        final_df = final_df.drop_duplicates(subset=['open_time']).sort_values('open_time').reset_index(drop=True)
        return final_df.head(2000) if len(final_df) > 2000 else final_df
    else:
        return None

def main():
    """
    Main function to fetch data and train ML models for price prediction
    """
    print("🤖 BITCOIN PRICE PREDICTION WITH MACHINE LEARNING")
    print("=" * 60)
    
    # Fetch data
    print("\n📊 Fetching OHLCV data from Binance...")
    df = fetch_2000_ohlcv_points()
    
    if df is None:
        print("❌ Failed to fetch data from Binance API")
        return
    
    print(f"✅ Successfully fetched {len(df)} data points")
    
    # Calculate technical indicators
    print("\n🔧 Calculating technical indicators...")
    df = calculate_technical_indicators(df)
    
    # Prepare data for ML
    print("\n📈 Preparing data for machine learning...")
    prediction_horizon = 1  # Predict next day's direction
    X, y, df_clean = prepare_ml_data(df, prediction_horizon)
    
    print(f"📋 Dataset Info:")
    print(f"   Total samples: {len(df_clean)}")
    print(f"   Features: {X.shape[1]}")
    print(f"   Target distribution: {y.value_counts().to_dict()}")
    
    # Train ML models
    results, X_test, y_test, scaler = train_ml_models(X, y)
    
    # Display model comparison
    display_model_comparison(results)
    
    # Analyze feature importance
    feature_imp_df = analyze_feature_importance(results, X.columns.tolist())
    
    # Make predictions
    predictions_df = make_predictions(results, df_clean, X.columns.tolist(), prediction_horizon)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save predictions
    predictions_df.to_csv(f'price_predictions_{timestamp}.csv', index=False)
    
    # Save feature importance
    if feature_imp_df is not None:
        feature_imp_df.to_csv(f'feature_importance_{timestamp}.csv', index=False)
    
    print(f"\n💾 Results saved to:")
    print(f"   - price_predictions_{timestamp}.csv")
    print(f"   - feature_importance_{timestamp}.csv")
    
    print("\n🚀 NEXT STEPS:")
    print("1. Experiment with different prediction horizons (3-day, 7-day)")
    print("2. Try more advanced models (LSTM, XGBoost)")
    print("3. Add sentiment analysis features")
    print("4. Implement ensemble methods")
    print("5. Backtest trading strategies based on predictions")

if __name__ == "__main__":
    main()
