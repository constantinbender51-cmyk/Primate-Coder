import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Load dataset
data = fetch_california_housing()
X, y = data.data, data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

target_range = np.max(y) - np.min(y)

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
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    nmae = mae / target_range
    results.append((name, nmae))

# Show comparison
print("\n=== Model NMAE Comparison ===")
print("Model\t\t\tNMAE (%)")
print("-" * 30)
for name, nmae in sorted(results, key=lambda x: x[1]):
    print(f"{name:<20} {nmae * 100:.2f}%")

# Best model
best_model = min(results, key=lambda x: x[1])
print(f"\nBest: {best_model[0]} ({best_model[1] * 100:.2f}% NMAE)")