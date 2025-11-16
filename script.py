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

def fetch_crypto_data_chunked(symbol, hours_to_fetch=2500):
    """Fetch OHLCV data for any cryptocurrency from Binance using chunking - HOURLY DATA"""
    client = Spot()
    
    all_data = []
    chunk_size = 1000  # Binance API limit per request
    
    # Calculate end time (current time)
    end_time = None
    
    for i in range(0, hours_to_fetch, chunk_size):
        limit = min(chunk_size, hours_to_fetch - i)
        
        try:
            klines = client.klines(
                symbol=symbol,
                interval='1h',  # Changed from '1d' to '1h' for hourly data
                limit=limit,
                endTime=end_time
            )
            
            if not klines:
                break
                
            all_data.extend(klines)
            
            # Set end_time for next chunk (oldest data)
            end_time = int(klines[0][0]) - 1  # Subtract 1ms from first candle
            
            print(f"Fetched {len(klines)} hours for {symbol}, total: {len(all_data)} hours")
            
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
    """Create technical indicators and features including altcoin data - ONLY DERIVATIVES - HOURLY ADJUSTED"""
    df = btc_df.copy()
    
    # Price-based derivative features for Bitcoin (NO RAW PRICES)
    df['price_change'] = df['close'].pct_change()
    df['high_low_ratio'] = df['high'] / df['low']
    df['close_open_ratio'] = df['close'] / df['open']
    
    # Moving averages (adjusted for hourly data - 24h and 48h windows)
    df['sma_24'] = df['close'].rolling(24).mean()  # 24 hours = 1 day
    df['sma_48'] = df['close'].rolling(48).mean()  # 48 hours = 2 days
    
    # Price relative to moving averages (derivatives)
    df['price_vs_sma24'] = df['close'] / df['sma_24']
    df['price_vs_sma48'] = df['close'] / df['sma_48']
    
    # Volatility (adjusted for hourly data)
    df['volatility_24'] = df['price_change'].rolling(24).std()
    df['volatility_48'] = df['price_change'].rolling(48).std()
    
    # Volume features (derivatives only)
    df['volume_sma_24'] = df['volume'].rolling(24).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma_24']
    
    # MACD features for Bitcoin
    macd_line, signal_line, macd_histogram = calculate_macd(df)
    df['macd_line'] = macd_line
    df['macd_signal'] = signal_line
    df['macd_histogram'] = macd_histogram
    
    # Add altcoin features (derivatives only)
    # Ethereum features
    df['eth_price_change'] = eth_df['close'].pct_change()
    df['eth_volume_ratio'] = eth_df['volume'] / eth_df['volume'].rolling(24).mean()
    df['btc_eth_ratio'] = df['close'] / eth_df['close']  # BTC dominance vs ETH
    
    # Ripple features
    df['xrp_price_change'] = xrp_df['close'].pct_change()
    df['xrp_volume_ratio'] = xrp_df['volume'] / xrp_df['volume'].rolling(24).mean()
    df['btc_xrp_ratio'] = df['close'] / xrp_df['close']  # BTC dominance vs XRP
    
    # Cardano features
    df['ada_price_change'] = ada_df['close'].pct_change()
    df['ada_volume_ratio'] = ada_df['volume'] / ada_df['volume'].rolling(24).mean()
    df['btc_ada_ratio'] = df['close'] / ada_df['close']  # BTC dominance vs ADA
    
    # Altcoin momentum indicators
    df['altcoin_momentum'] = (df['eth_price_change'] + df['xrp_price_change'] + df['ada_price_change']) / 3
    df['altcoin_volume_strength'] = (df['eth_volume_ratio'] + df['xrp_volume_ratio'] + df['ada_volume_ratio']) / 3
    
    # Target: Next N-hour price direction (1 = up, 0 = down)
    df['target'] = (df['close'].shift(-holding_period) > df['close']).astype(int)
    
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
    """Add lagged versions of SELECTED features with specific periods: [1, 3, 12] hours"""
    X_lagged = X.copy()
    
    # Selected features for lagging - added price_change and price_vs_sma24
    selected_features = [
        'price_change', 'price_vs_sma24', 'volume_ratio', 'volatility_24', 'volatility_48',
        'eth_price_change', 'xrp_price_change', 'ada_price_change',
        'altcoin_momentum', 'macd_histogram'
    ]
    
    # Lag periods - removed 24
    lag_periods = [1, 3, 12]
    
    # Add lagged features for selected features
    lagged_features_added = []
    for feature in selected_features:
        if feature in X.columns:
            for lag in lag_periods:
                lagged_feature_name = f"{feature}_lag_{lag}"
                X_lagged[lagged_feature_name] = X[feature].shift(lag)
                lagged_features_added.append(lagged_feature_name)
    
    # Drop rows with NaN values created by lagging (drop first 12 rows - maximum lag)
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
    # Fetch Bitcoin data
    print("Fetching 2,500 hours of Bitcoin data...")
    btc_df = fetch_crypto_data_chunked('BTCUSDT', 2500)  # Changed to 2,500 hours
    
    # Fetch altcoin data
    print("\nFetching Ethereum data...")
    eth_df = fetch_crypto_data_chunked('ETHUSDT', 2500)  # Changed to 2,500 hours
    
    print("\nFetching Ripple data...")
    xrp_df = fetch_crypto_data_chunked('XRPUSDT', 2500)  # Changed to 2,500 hours
    
    print("\nFetching Cardano data...")
    ada_df = fetch_crypto_data_chunked('ADAUSDT', 2500)  # Changed to 2,500 hours
    
    # Create features with altcoin data
    df = create_features_with_altcoins(btc_df, eth_df, xrp_df, ada_df)
    
    # Dataset information
    print(f"\nDataset Info:")
    print(f"  Total hours: {len(df)}")
    print(f"  Date range: {df['date'].min().strftime('%Y-%m-%d %H:%M')} to {df['date'].max().strftime('%Y-%m-%d %H:%M')}")
    print(f"  BTC price range: ${df['close'].min():.0f} - ${df['close'].max():.0f}")
    print(f"  ETH price range: ${eth_df['close'].min():.2f} - ${eth_df['close'].max():.2f}")
    print(f"  XRP price range: ${xrp_df['close'].min():.4f} - ${xrp_df['close'].max():.4f}")
    print(f"  ADA price range: ${ada_df['close'].min():.4f} - ${ada_df['close'].max():.4f}")
    
    # Prepare features and target (EXCLUDE RAW PRICE DATA)
    exclude_cols = ['date', 'target', 'close', 'open', 'high', 'low', 'volume']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    X = df[feature_cols]
    y = df['target']
    
    print(f"\nOriginal feature columns (DERIVATIVES ONLY): {feature_cols}")
    
    # Add polynomial features (squared and cubed)
    X_with_poly = add_polynomial_features(X)
    squared_feature_cols = [col for col in X_with_poly.columns if col.endswith('_squared')]
    cubed_feature_cols = [col for col in X_with_poly.columns if col.endswith('_cubed')]
    
    print(f"\nSquared features added: {squared_feature_cols}")
    print(f"Cubed features added: {cubed_feature_cols}")
    
    # Add lagged features for SELECTED features with specific periods
    X_with_lags, lagged_features_added = add_lagged_features_selected(X_with_poly)
    
    # Update target to match lagged features (drop first 12 rows - maximum lag)
    y_lagged = y.iloc[12:]
    
    # Define selected features for reporting (same as in add_lagged_features_selected)
    selected_features_for_report = [
        'price_change', 'price_vs_sma24', 'volume_ratio', 'volatility_24', 'volatility_48',
        'eth_price_change', 'xrp_price_change', 'ada_price_change',
        'altcoin_momentum', 'macd_histogram'
    ]
    
    # Calculate feature counts
    original_features = len(X.columns)
    polynomial_features = len(squared_feature_cols) + len(cubed_feature_cols)
    lagged_features = len(lagged_features_added)
    total_features = original_features + polynomial_features + lagged_features
    
    print(f"\nLagged features added: {lagged_features} features")
    print(f"  Selected features: {len(selected_features_for_report)} key features")
    print(f"  Lag periods: [1, 3, 12] hours")
    
    print(f"\nTotal features: {total_features}")
    print(f"  Original: {original_features}")
    print(f"  Polynomial: {polynomial_features}")
    print(f"  Lagged: {lagged_features}")
    
    # Print first 2 hours of non-lagged features for clarity
    print(f"\nFirst 2 hours of features (BEFORE NORMALIZATION - showing first 10 non-lagged features):")
    for i in range(2):
        print(f"\nHour {i+1} ({df.iloc[i+12]['date'].strftime('%Y-%m-%d %H:%M')}):")
        for j, feature in enumerate(feature_cols[:10]):  # Show only first 10 features
            if feature in X_with_lags.columns:
                value = X_with_lags.iloc[i][feature]
                print(f"  {feature}: {value:.6f}")
    
    # Normalize features
    X_normalized = normalize_features(X_with_lags)
    
    # Print first 2 hours of normalized features
    print(f"\nFirst 2 hours of features (AFTER NORMALIZATION - showing first 10 non-lagged features):")
    for i in range(2):
        print(f"\nHour {i+1} ({df.iloc[i+12]['date'].strftime('%Y-%m-%d %H:%M')}):")
        for j, feature in enumerate(feature_cols[:10]):  # Show only first 10 features
            if feature in X_normalized.columns:
                value = X_normalized.iloc[i][feature]
                print(f"  {feature}: {value:.6f}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_normalized, y_lagged, test_size=0.2, random_state=42, shuffle=False
    )
    
    print(f"\n  Train samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    
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
        results[name] = {'accuracy': accuracy, 'f1_score': f1, 'model': model, 'predictions': y_pred}
        
        print(f"{name}:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1-Score: {f1:.4f}")
    
    # Find best model by F1 score
    best_model = max(results, key=lambda x: results[x]['f1_score'])
    print(f"\nBest Model: {best_model}")
    
    # Backtesting simulation for ALL models
    print("\n=== BACKTESTING SIMULATION FOR ALL MODELS ===")
    
    # Get test data for all models
    test_start_idx = len(y_lagged) - len(y_test) + 12  # Correct alignment
    test_dates = df.iloc[test_start_idx:test_start_idx + len(y_test)]['date'].values
    test_prices = df.iloc[test_start_idx:test_start_idx + len(y_test)]['close'].values
    
    def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
        """Calculate Sharpe ratio for a series of returns"""
        if len(returns) == 0:
            return 0.0
        excess_returns = np.array(returns) - risk_free_rate
        if np.std(excess_returns) == 0:
            return 0.0
        return np.mean(excess_returns) / np.std(excess_returns)
    
    def run_backtest(predictions, model_name, test_dates, test_prices, initial_balance=10000.0):
        """Run backtesting simulation with SIMPLIFIED balance calculation"""
        balance = initial_balance
        portfolio_values = []
        returns = []
        
        # Trading simulation
        for i in range(len(predictions)):
            current_price = test_prices[i]
            prediction = predictions[i]
            
            # FIXED LOGIC: Use PREVIOUS period's prediction for CURRENT period's balance
            if i > 0:  # Need previous price and previous prediction
                previous_price = test_prices[i-1]
                previous_prediction = predictions[i-1]  # Use the prediction from previous hour
                
                # Calculate new balance based on previous prediction and price ratio
                if previous_prediction == 1:  # Previous hour predicted UP
                    balance = balance * (1 + (current_price / previous_price - 1))
                else:  # Previous hour predicted DOWN
                    balance = balance * (1 - (current_price / previous_price - 1))
            
            # Track portfolio value at each step
            portfolio_values.append(balance)
            
            # Calculate daily returns
            if i > 0:
                daily_return = (balance - portfolio_values[i-1]) / portfolio_values[i-1]
                returns.append(daily_return)
            
            # Print detailed information for first 20 hours
            if i < 20:
                print(f"\nHour {i+1} ({test_dates[i]}):")
                print(f"  Price: ${current_price:,.2f}")
                if i > 0:
                    print(f"  Using prediction from hour {i}: {previous_prediction} ({'UP' if previous_prediction == 1 else 'DOWN'})")
                    print(f"  Previous price (hour {i}): ${previous_price:,.2f}")
                    print(f"  Price ratio (prev/current): {previous_price/current_price:.6f}")
                print(f"  Balance: ${balance:,.2f}")
                
                # Show action taken
                if i > 0:
                    if previous_prediction == 1:
                        print(f"  Action: LONG - Balance = (1 + {current_price/previous_price - 1:.6f}) * previous balance")
                    else:
                        print(f"  Action: SHORT - Balance = (1 - {current_price/previous_price - 1:.6f}) * previous balance")
        # Calculate final portfolio value
        final_balance = balance
        
        # Calculate performance metrics
        total_return = (final_balance - initial_balance) / initial_balance * 100
        buy_hold_return = (test_prices[-1] - test_prices[0]) / test_prices[0] * 100
        
        # Calculate Sharpe ratio
        sharpe_ratio = calculate_sharpe_ratio(returns) if returns else 0.0
        
        return {
            'model_name': model_name,
            'final_balance': final_balance,
            'total_return': total_return,
            'buy_hold_return': buy_hold_return,
            'sharpe_ratio': sharpe_ratio,
            'portfolio_values': portfolio_values,
            'returns': returns
        }
    # Run backtest for all models
    backtest_results = {}
    
    for name in ['Logistic Regression', 'Random Forest', 'Gradient Boosting']:
        print(f"\n=== {name.upper()} BACKTESTING SIMULATION ===")
        
        # Get model predictions
        if name == 'Logistic Regression':
            predictions = results[name]['predictions']
        else:
            predictions = results[name]['predictions']
        
        # Run backtest
        print(f"\nDetailed Hourly Breakdown for {name}:")
        result = run_backtest(predictions, name, test_dates, test_prices)
        backtest_results[name] = result
        
        # Print results
        print(f"\nTrading Simulation Results:")
        print(f"  Initial Balance: ${10000:,.2f}")
        print(f"  Final Balance: ${result['final_balance']:,.2f}")
        print(f"  Total Return: {result['total_return']:+.2f}%")
        print(f"  Buy & Hold Return: {result['buy_hold_return']:+.2f}%")
        print(f"  Sharpe Ratio: {result['sharpe_ratio']:.4f}")
        
        # Compare strategy vs buy & hold
        if result['total_return'] > result['buy_hold_return']:
            outperformance = result['total_return'] - result['buy_hold_return']
            print(f"  Strategy Outperformance: +{outperformance:.2f}% vs Buy & Hold")
        else:
            underperformance = result['buy_hold_return'] - result['total_return']
            print(f"  Strategy Underperformance: -{underperformance:.2f}% vs Buy & Hold")
    
    # Compare all models
    print("\n=== MODEL COMPARISON SUMMARY ===")
    print("\nPerformance Comparison:")
    print(f"{'Model':<20} {'Total Return':<12} {'Sharpe Ratio':<12}")
    print("-" * 45)
    
    for name in ['Logistic Regression', 'Random Forest', 'Gradient Boosting']:
        result = backtest_results[name]
        print(f"{name:<20} {result['total_return']:>+10.2f}% {result['sharpe_ratio']:>11.4f}")
    
    # Find best model by Sharpe ratio
    best_sharpe_model = max(backtest_results.keys(), 
                           key=lambda x: backtest_results[x]['sharpe_ratio'])
    best_return_model = max(backtest_results.keys(), 
                           key=lambda x: backtest_results[x]['total_return'])
    
    print(f"\nBest Model by Sharpe Ratio: {best_sharpe_model} ({backtest_results[best_sharpe_model]['sharpe_ratio']:.4f})")
    print(f"Best Model by Total Return: {best_return_model} ({backtest_results[best_return_model]['total_return']:+.2f}%)")
    
    print("\n=== BACKTESTING COMPLETE ===")

if __name__ == "__main__":
    main()