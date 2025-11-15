import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Load dataset
print("Loading California Housing dataset...")
data = fetch_california_housing()
X, y = data.data, data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

target_range = np.max(y) - np.min(y)
print(f"Target Range: {target_range:.4f}\n")

# Define models to test
models = {
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Linear Regression': LinearRegression(),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'Support Vector Regression': SVR()
}

# Train and evaluate each model
results = []
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    nmae = mae / target_range
    
    results.append((name, mae, nmae))
    print(f"  MAE: {mae:.4f}")
    print(f"  NMAE: {nmae:.4f} ({nmae * 100:.2f}%)\n")

# Show comparison
print("\n=== Model Comparison ===")
print("Model\t\t\tNMAE (%)")
print("-" * 30)
for name, mae, nmae in sorted(results, key=lambda x: x[2]):
    print(f"{name:<20} {nmae * 100:.2f}%")

# Best model
best_model = min(results, key=lambda x: x[2])
print(f"\nBest model: {best_model[0]} with NMAE = {best_model[2] * 100:.2f}%")