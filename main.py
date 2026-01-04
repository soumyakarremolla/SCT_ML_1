"""
Main script for linear regression on the House Prices dataset.

Predicts house prices based on square footage (GrLivArea), number of bedrooms (BedroomAbvGr), and number of bathrooms (FullBath).
Now supports dynamic user input for predictions.
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def main():
    print("Loading training data...")
    data = pd.read_csv("house_price_regression/data/train.csv")

    # Select relevant features and target
    features = ["GrLivArea", "BedroomAbvGr", "FullBath"]
    target = "SalePrice"

    print(f"Using features: {features}")
    X = data[features]
    y = data[target]

    # Handle missing values (if any)
    X = X.fillna(X.median())

    # Split into train and test sets
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the linear regression model
    print("Training linear regression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\nModel Evaluation on Test Set:")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R^2 Score: {r2:.4f}")

    # Dynamic user input for prediction
    print("\n--- Predict House Price ---")
    try:
        grlivarea = float(input("Enter square footage (GrLivArea): "))
        bedrooms = int(input("Enter number of bedrooms (BedroomAbvGr): "))
        fullbath = int(input("Enter number of full bathrooms (FullBath): "))
        user_features = pd.DataFrame(
            [[grlivarea, bedrooms, fullbath]],
            columns=["GrLivArea", "BedroomAbvGr", "FullBath"]
        )
        predicted_price = model.predict(user_features)[0]
        print(f"\nPredicted House Price: ${predicted_price:,.2f}")
    except Exception as e:
        print(f"Error in input or prediction: {e}")

    print("\nScript complete. You can rerun to test with different values.")

if __name__ == "__main__":
    main()
