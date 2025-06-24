import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

# ===============================
# PART 1: UNDERSTANDING THE MATH
# ===============================

print("=== PART 1: Understanding Linear Regression Mathematics ===\n")

# Let's start with a simple example to understand the core concept
# We want to predict house prices based on size

# Sample data: house sizes (sq ft) and their prices ($1000s)
house_sizes = np.array([1000, 1200, 1500, 1800, 2000, 2200, 2500])
house_prices = np.array([150, 180, 220, 270, 300, 330, 380])

print("Sample Data:")
print("House Sizes (sq ft):", house_sizes)
print("House Prices ($1000s):", house_prices)
print()

# The linear regression equation is: y = mx + b
# Where:
# y = predicted price
# x = house size
# m = slope (how much price increases per sq ft)
# b = y-intercept (base price when size = 0)

# Let's calculate these manually using the normal equation
""" For simple linear regression: 
    m = (n*Σ(xy) - Σ(x)*Σ(y)) / (n*Σ(x²) - (Σ(x))²)
    b = (Σ(y) - m*Σ(x)) / n """

n = len(house_sizes)
sum_x = np.sum(house_sizes)
sum_y = np.sum(house_prices)
sum_xy = np.sum(house_sizes * house_prices)
sum_x_squared = np.sum(house_sizes**2)  # Calculate the sum of squares of house_sizes

# Calculate slope (m) and intercept (b)
m = (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x**2)
b = (sum_y - m * sum_x) / n

print(f"Calculated manually:")
print(f"Slope (m): {m:.4f} - This means price increases by ${m * 1000:.0f} per sq ft")
print(f"Intercept (b): {b:.4f} - This is the base price when size = 0")
print(f"Linear equation: y = {m:.4f}x + {b:.4f}")
print()

# ===============================
# PART 2: IMPLEMENTING FROM SCRATCH
# ===============================

print("=== PART 2: Linear Regression from Scratch ===\n")


class SimpleLinearRegression:
    """
    A simple linear regression implementation to understand the core concepts
    """

    def __init__(self):
        self.slope = None
        self.intercept = None
        self.cost_history = []

    def fit(self, X, y, learning_rate=0.01, epochs=1000):
        """
        Train the model using gradient descent

        X: input features (1D array)
        y: target values (1D array)
        learning_rate: step size for gradient descent
        epochs: number of training iterations
        """
        # Initialize parameters randomly
        self.slope = np.random.randn()
        self.intercept = np.random.randn()

        n = len(X)

        # Gradient descent optimization
        for epoch in range(epochs):
            # Make predictions with current parameters
            y_pred = self.slope * X + self.intercept

            # Calculate cost (Mean Squared Error)
            cost = np.mean((y - y_pred) ** 2)
            self.cost_history.append(cost)

            # Calculate gradients (partial derivatives)
            d_slope = (-2 / n) * np.sum(X * (y - y_pred))
            d_intercept = (-2 / n) * np.sum(y - y_pred)

            # Update parameters
            self.slope -= learning_rate * d_slope
            self.intercept -= learning_rate * d_intercept

            # Print progress every 100 epochs
            if epoch % 100 == 0:
                print(
                    f"Epoch {epoch}: Cost = {cost:.4f}, Slope = {self.slope:.4f}, Intercept = {self.intercept:.4f}"
                )

    def predict(self, X):
        """Make predictions using the trained model"""
        return self.slope * X + self.intercept

    def plot_cost(self):
        """Plot the cost function during training"""
        plt.figure(figsize=(10, 4))
        plt.plot(self.cost_history)
        plt.title("Cost Function During Training")
        plt.xlabel("Epoch")
        plt.ylabel("Mean Squared Error")
        plt.grid(True)
        plt.show()


# Train our custom model
print("Training our custom linear regression model...")
custom_model = SimpleLinearRegression()
custom_model.fit(house_sizes, house_prices, learning_rate=0.0000001, epochs=1000)
print()

print(f"Final parameters after training:")
print(f"Slope: {custom_model.slope:.4f}")
print(f"Intercept: {custom_model.intercept:.4f}")
print()

# ===============================
# PART 3: VISUALIZATION
# ===============================

print("=== PART 3: Visualizing the Results ===\n")

# Create predictions for plotting
x_plot = np.linspace(800, 2700, 100)
y_manual = m * x_plot + b  # Manual calculation
y_custom = custom_model.predict(x_plot)  # Our custom model

# Plot the results
plt.figure(figsize=(12, 8))

# Original data points
# Manual
plt.subplot(2, 2, 1)
plt.scatter(
    house_sizes, house_prices, color="red", s=100, alpha=0.7, label="Actual Data"
)
plt.plot(
    x_plot, y_manual, color="blue", linewidth=2, label=f"Manual: y = {m:.4f}x + {b:.1f}"
)
plt.xlabel("House Size (sq ft)")
plt.ylabel("Price ($1000s)")
plt.title("Manual Calculation Results")
plt.legend()
plt.grid(True, alpha=0.3)

# Custom
plt.subplot(2, 2, 2)
plt.scatter(
    house_sizes, house_prices, color="red", s=100, alpha=0.7, label="Actual Data"
)
plt.plot(
    x_plot,
    y_custom,
    color="green",
    linewidth=2,
    label=f"Custom Model: y = {custom_model.slope:.4f}x + {custom_model.intercept:.1f}",
)
plt.xlabel("House Size (sq ft)")
plt.ylabel("Price ($1000s)")
plt.title("Custom Model Results")
plt.legend()
plt.grid(True, alpha=0.3)

# Cost function
plt.subplot(2, 2, 3)
plt.plot(custom_model.cost_history, color="orange", linewidth=2)
plt.title("Training Cost Over Time")
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error")
plt.grid(True, alpha=0.3)

# Residuals (errors)
predictions = custom_model.predict(house_sizes)
residuals = house_prices - predictions

plt.subplot(2, 2, 4)
plt.scatter(predictions, residuals, color="purple", s=100, alpha=0.7)
plt.axhline(y=0, color="black", linestyle="--", alpha=0.5)
plt.xlabel("Predicted Values")
plt.ylabel("Residuals (Actual - Predicted)")
plt.title("Residual Plot")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ===============================
# PART 4: MULTIPLE LINEAR REGRESSION
# ===============================

print("\n=== PART 4: Multiple Linear Regression ===\n")

# Generate synthetic dataset with multiple features
X_multi, y_multi = make_regression(
    n_samples=100, n_features=3, noise=10, random_state=42
)

# Add feature names for better understanding
feature_names = ["Size (sq ft)", "Age (years)", "Location Score"]
X_multi_df = pd.DataFrame(X_multi, columns=feature_names)

print("Multiple Linear Regression Example:")
print("Features:", feature_names)
print("Dataset shape:", X_multi.shape)
print("\nFirst 5 rows of features:")
print(X_multi_df.head())
print()


class MultipleLinearRegression:
    """
    Multiple linear regression implementation using matrix operations
    """

    def __init__(self):
        self.weights = None
        self.intercept = None

    def fit(self, X, y):
        """
        Train using the normal equation: θ = (X^T X)^(-1) X^T y
        This is the analytical solution for linear regression
        """
        # Add bias column (column of ones) to X
        X_with_bias = np.column_stack([np.ones(X.shape[0]), X])

        # Normal equation: θ = (X^T X)^(-1) X^T y
        theta = np.linalg.inv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ y

        # Separate intercept and weights
        self.intercept = theta[0]
        self.weights = theta[1:]

    def predict(self, X):
        """Make predictions"""
        return X @ self.weights + self.intercept


# Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X_multi, y_multi, test_size=0.2, random_state=42
)

# Train our multiple regression model
multi_model = MultipleLinearRegression()
multi_model.fit(X_train, y_train)

# Make predictions
y_pred_train = multi_model.predict(X_train)
y_pred_test = multi_model.predict(X_test)

print("Multiple Linear Regression Results:")
print(
    "Model Equation: y = {:.2f} + {:.2f}*{} + {:.2f}*{} + {:.2f}*{}".format(
        multi_model.intercept,
        multi_model.weights[0],
        feature_names[0],
        multi_model.weights[1],
        feature_names[1],
        multi_model.weights[2],
        feature_names[2],
    )
)
print()

# Calculate metrics
train_mse = mean_squared_error(y_train, y_pred_train)
test_mse = mean_squared_error(y_test, y_pred_test)
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)

print(f"Training MSE: {train_mse:.2f}")
print(f"Testing MSE: {test_mse:.2f}")
print(f"Training R²: {train_r2:.3f}")
print(f"Testing R²: {test_r2:.3f}")
print()

# ===============================
# PART 5: COMPARISON WITH SKLEARN
# ===============================

print("=== PART 5: Comparison with Scikit-learn ===\n")

# Train sklearn model for comparison
sklearn_model = LinearRegression()
sklearn_model.fit(X_train, y_train)
sklearn_pred = sklearn_model.predict(X_test)

print("Comparison of our implementation vs Scikit-learn:")
print(f"Our model - Test MSE: {test_mse:.6f}")
print(f"Sklearn model - Test MSE: {mean_squared_error(y_test, sklearn_pred):.6f}")
print()

print("Weight comparison:")
print("Feature", " " * 15, "Our Model", " " * 5, "Sklearn")
print("-" * 50)
for i, feature in enumerate(feature_names):
    print(
        f"{feature:<20} {multi_model.weights[i]:>10.4f} {sklearn_model.coef_[i]:>10.4f}"
    )
print(
    f"{'Intercept':<20} {multi_model.intercept:>10.4f} {sklearn_model.intercept_:>10.4f}"
)
print()

# ===============================
# PART 6: MODEL EVALUATION
# ===============================

print("=== PART 6: Understanding Model Performance ===\n")

# Visualize predictions vs actual values
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_test, alpha=0.7, color="blue")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", linewidth=2)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title(f"Predictions vs Actual (R² = {test_r2:.3f})")
plt.grid(True, alpha=0.3)

# Residual plot
residuals_multi = y_test - y_pred_test
plt.subplot(1, 2, 2)
plt.scatter(y_pred_test, residuals_multi, alpha=0.7, color="green")
plt.axhline(y=0, color="red", linestyle="--", linewidth=2)
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("Key Insights:")
print(
    "- R² (coefficient of determination) measures how well the model explains the variance in the data"
)
print(
    "- R² = 1 means perfect predictions, R² = 0 means the model is no better than predicting the mean"
)
print(
    "- MSE (Mean Squared Error) penalizes larger errors more heavily than smaller ones"
)
print(
    "- Residual plots help identify if the model assumptions are met (residuals should be randomly distributed around 0)"
)
print()

# ===============================
# PART 7: PRACTICAL TIPS
# ===============================

print("=== PART 7: Practical Tips and Common Issues ===\n")

# Feature scaling example
from sklearn.preprocessing import StandardScaler

print("Feature Scaling Impact:")
print("Original feature ranges:")
for i, feature in enumerate(feature_names):
    print(f"{feature}: [{X_train[:, i].min():.2f}, {X_train[:, i].max():.2f}]")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train on scaled data
scaled_model = MultipleLinearRegression()
scaled_model.fit(X_train_scaled, y_train)
scaled_pred = scaled_model.predict(X_test_scaled)

print(f"\nPerformance comparison:")
print(f"Original features - Test MSE: {test_mse:.6f}")
print(f"Scaled features - Test MSE: {mean_squared_error(y_test, scaled_pred):.6f}")
print()

print("Important Considerations:")
print("1. Linear regression assumes a linear relationship between features and target")
print(
    "2. It's sensitive to outliers - consider robust regression for outlier-heavy data"
)
print("3. Feature scaling can help with numerical stability and interpretation")
print(
    "4. Check assumptions: linearity, independence, homoscedasticity, normality of residuals"
)
print("5. For high-dimensional data, consider regularization (Ridge, Lasso)")
print()

print("When to use Linear Regression:")
print("✓ When you need an interpretable model")
print("✓ As a baseline for more complex models")
print("✓ When the relationship appears roughly linear")
print("✓ When you have limited data")
print("✓ For understanding feature importance")
print()

print("Tutorial completed! You now understand:")
print("- The mathematical foundation of linear regression")
print("- How gradient descent optimizes the parameters")
print("- The difference between simple and multiple linear regression")
print("- How to implement it from scratch")
print("- How to evaluate model performance")
print("- Common practical considerations")
