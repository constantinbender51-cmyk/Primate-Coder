import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Load dataset
print("Loading California Housing dataset...")
data = fetch_california_housing()
X, y = data.data, data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
print("Training Random Forest model...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
target_range = np.max(y) - np.min(y)
nmae = mae / target_range

# Print results
print(f"\nMean Absolute Error: {mae:.4f}")
print(f"Target Range: {target_range:.4f}")
print(f"Normalized MAE: {nmae:.4f}")
print(f"NMAE as percentage: {nmae * 100:.2f}%")