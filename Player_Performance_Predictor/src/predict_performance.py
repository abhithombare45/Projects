import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# Load metrics
df = pd.read_csv('../data/processed/metrics.csv')

# Simulate a performance score (e.g., expected goals, manually labeled for demo)
df['performance_score'] = df['avg_speed'] * 0.5 + df['total_distance'] * 0.1  # Dummy formula

# Features and target
X = df[['avg_speed', 'total_distance']]
y = df['performance_score']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = XGBRegressor()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Save predictions
df['predicted_performance'] = model.predict(X)
df.to_csv('../data/processed/metrics_with_predictions.csv', index=False)
print("Performance prediction complete")
