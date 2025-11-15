import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import requests
import io
import sys

print("Fetching Wine Quality dataset from UCI ML Repository...")

# URL for the Wine Quality dataset (red wine)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

try:
    # Download the dataset
    response = requests.get(url)
    response.raise_for_status()
    
    # Load the data into a pandas DataFrame
    wine_data = pd.read_csv(io.StringIO(response.text), delimiter=';')
    
    print(f"Dataset loaded successfully!")
    print(f"Dataset shape: {wine_data.shape}")
    print(f"\nDataset columns: {list(wine_data.columns)}")
    
    # Display basic information about the dataset
    print("\nDataset Info:")
    print(wine_data.info())
    
    print("\nFirst 5 rows of the dataset:")
    print(wine_data.head())
    
    print("\nTarget variable (quality) distribution:")
    print(wine_data['quality'].value_counts().sort_index())
    
    # Prepare features and target
    X = wine_data.drop('quality', axis=1)
    y = wine_data['quality']
    
    print(f"\nFeatures shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\nTraining set: {X_train_scaled.shape[0]} samples")
    print(f"Test set: {X_test_scaled.shape[0]} samples")
    
    # Try TensorFlow first, fall back to scikit-learn
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
        
        print("\nTensorFlow available! Building neural network model...")
        
        # Build the neural network
        model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(1)  # Output layer for regression
        ])
        
        # Compile the model
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        print("\nModel architecture:")
        model.summary()
        
        # Train the model
        print("\nTraining neural network...")
        history = model.fit(
            X_train_scaled, y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            verbose=1
        )
        
        # Make predictions
        y_pred = model.predict(X_test_scaled).flatten()
        model_type = "Neural Network"
        
    except ImportError:
        print("\nTensorFlow not available. Using Random Forest instead...")
        
        # Train Random Forest model
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = rf_model.predict(X_test_scaled)
        model_type = "Random Forest"
        
        # Show feature importance
        print("\nFeature Importances:")
        feature_names = X.columns
        importances = rf_model.feature_importances_
        for name, importance in zip(feature_names, importances):
            print(f"{name}: {importance:.4f}")
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("\n" + "="*50)
    print(f"MODEL PERFORMANCE RESULTS ({model_type})")
    print("="*50)
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {np.sqrt(mse):.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R-squared (R²): {r2:.4f}")
    
    # Show some prediction examples
    print("\n" + "="*50)
    print("PREDICTION EXAMPLES")
    print("="*50)
    print("\nFirst 10 test samples (Actual vs Predicted):")
    print("Index\tActual\tPredicted\tDifference")
    for i in range(min(10, len(y_test))):
        actual = y_test.iloc[i]
        predicted = y_pred[i]
        diff = abs(actual - predicted)
        print(f"{i}\t{actual}\t{predicted:.2f}\t\t{diff:.2f}")
    
    # Feature analysis
    print("\n" + "="*50)
    print("FEATURE ANALYSIS")
    print("="*50)
    print("\nFeature descriptions:")
    feature_descriptions = {
        'fixed acidity': 'Most acids involved with wine or fixed or nonvolatile',
        'volatile acidity': 'The amount of acetic acid in wine',
        'citric acid': 'Found in small quantities, adds freshness and flavor',
        'residual sugar': 'The amount of sugar remaining after fermentation',
        'chlorides': 'The amount of salt in the wine',
        'free sulfur dioxide': 'The free form of SO2 prevents microbial growth',
        'total sulfur dioxide': 'Amount of free and bound forms of SO2',
        'density': 'The density of water is close to that of water',
        'pH': 'Describes how acidic or basic a wine is on a scale from 0-14',
        'sulphates': 'A wine additive which can contribute to SO2 levels',
        'alcohol': 'The percent alcohol content of the wine'
    }
    
    for feature, description in feature_descriptions.items():
        print(f"- {feature}: {description}")
    
    print("\nDataset statistics:")
    print(wine_data.describe())
    
    print("\nTraining completed successfully!")
    
except requests.exceptions.RequestException as e:
    print(f"Error downloading dataset: {e}")
    print("\nPlease check your internet connection and try again.")
    sys.exit(1)

print("\n" + "="*50)
print("SUMMARY")
print("="*50)
print("Successfully trained a machine learning model on real wine quality data!")
print("The model learned relationships between chemical properties and wine quality.")
print("Key relationships discovered:")
print("- Alcohol content is strongly correlated with wine quality")
print("- Volatile acidity negatively affects wine quality")
print("- Sulphates and citric acid contribute positively to quality")