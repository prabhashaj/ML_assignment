import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
st.title('Nvidia Stock Price Prediction with Ridge Regression')

# Add Nvidia logo
st.image("nvidia.png", width=300)  # Ensure the correct path for the Nvidia logo

# Load Nvidia stock price dataset directly from the path
file_path = 'NvidiaStockPrice.csv'  # Replace with the correct path to your file
data = pd.read_csv(file_path)

# Show dataset preview
st.subheader("Dataset Preview")
st.dataframe(data.head())

# Convert 'Date' column to datetime if available
if 'Date' in data.columns:
    data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y')

# Drop non-numeric columns like 'Date'
if 'Date' in data.columns:
    data = data.drop('Date', axis=1, errors='ignore')

# Check for missing values
if data.isnull().sum().any():
    st.warning("Dataset contains missing values. Dropping missing values for simplicity.")
    data = data.dropna()

# Feature Engineering: Create additional features (e.g., moving average)
data['Moving_Avg_5'] = data['Close'].rolling(window=5).mean()
data['Moving_Avg_10'] = data['Close'].rolling(window=10).mean()
data = data.dropna()  # Drop rows with NaN values after creating moving averages

# Define features (X) and target (y)
X = data.drop('Close', axis=1)
y = data['Close']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define parameter grid for Ridge regression
param_grid = {'alpha': [0.1, 1.0, 10.0, 100.0]}

# Set up GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(Ridge(), param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train_scaled, y_train)

# Best model from grid search
best_ridge = grid_search.best_estimator_

# Make predictions using the best model
predictions = best_ridge.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

# Display evaluation metrics
st.subheader("Model Performance after Tuning")
st.write(f"Mean Squared Error: {mse:.4f}")
st.write(f"Mean Absolute Error: {mae:.4f}")
st.write(f"R-squared: {r2:.4f}")

# Display a line chart for predicted vs actual values
st.subheader("Prediction vs Actual")
comparison = pd.DataFrame({"Actual": y_test, "Predicted": predictions})
st.line_chart(comparison)

# Plot a scatter plot of predictions vs actual
fig, ax = plt.subplots()
ax.scatter(y_test, predictions, edgecolors=(0, 0, 0))
ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--', lw=2)
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
ax.set_title('Actual vs Predicted Values')
st.pyplot(fig)

# User input for current prediction
st.subheader("Make Your Own Predictions")
input_data = {}
for feature in X.columns:
    input_data[feature] = st.number_input(f"Input {feature}", value=0.0)

# Create DataFrame for prediction inputs
input_df = pd.DataFrame([input_data])
input_scaled = scaler.transform(input_df)

# Predict with the user inputs
prediction = best_ridge.predict(input_scaled)
st.write(f"Predicted Stock Price: {prediction[0]:.2f}")

# User input for future predictions
st.subheader("Future Stock Price Predictions")
start_year = st.number_input("Start Year:", min_value=2024, max_value=2030, value=2024)
end_year = st.number_input("End Year:", min_value=2024, max_value=2030, value=2026)

if st.button("Predict Future Prices"):
    future_dates = pd.date_range(start=f"{start_year}-01-01", end=f"{end_year}-12-31", freq='B')
    future_data = pd.DataFrame(index=future_dates)

    # Prepare input data for predictions
    last_known_values = X.iloc[-1].to_frame().T  # Get the last known values as a DataFrame
    future_data = np.tile(last_known_values.values.flatten(), (len(future_dates), 1))  # Repeat for each future date

    # Generate variations for future predictions
    random_variation = np.random.normal(loc=0, scale=0.05 * last_known_values.values.flatten(), size=future_data.shape)
    future_data += random_variation

    # Predict future prices
    future_predictions = best_ridge.predict(scaler.transform(future_data))

    # Display future predictions
    future_results = pd.DataFrame({'Date': future_dates, 'Predicted Price': future_predictions})
    st.line_chart(future_results.set_index('Date'))

    st.subheader("Future Predictions Summary")
    st.dataframe(future_results)
