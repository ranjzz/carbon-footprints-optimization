import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. Load your data (replace 'data.csv' with your file)
data = pd.read_excel(r"C:\Users\ranji\Downloads\carbon-footprint-project\data1.csv.xlsx")

# 2. Preprocess data (example)
X = data[['route_distance', 'fuel_used', 'weather_index', 'traffic_index', 'cargo_weight']]
y = data['carbon_emission']

# 3. Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Build model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# 5. Train model
model.fit(X_train, y_train, epochs=10, batch_size=32)

# 6. Predict and evaluate
y_pred = model.predict(X_test).flatten()

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
percent_error = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print(f"MAE: {mae}")
print(f"RMSE: {rmse}")
print(f"Percentage Error: {percent_error:.2f}%")
# 7. Route Suggestion Engine Example

# Simulated new route options (you can replace this with real-time data later)
new_routes = pd.DataFrame([
    {'route_distance': 120, 'fuel_used': 15, 'weather_index': 3, 'traffic_index': 2, 'cargo_weight': 800},
    {'route_distance': 100, 'fuel_used': 18, 'weather_index': 4, 'traffic_index': 3, 'cargo_weight': 750},
    {'route_distance': 110, 'fuel_used': 14, 'weather_index': 2, 'traffic_index': 1, 'cargo_weight': 770}
])

# Predict emissions for each route
emissions = model.predict(new_routes).flatten()

# Add predictions to the dataframe
new_routes['predicted_emission'] = emissions

# Suggest the route with the least predicted emission
best_route = new_routes.loc[new_routes['predicted_emission'].idxmin()]

print("\nRoute Suggestions with Predicted Emissions:")
print(new_routes)
print("\nBest Route Based on Lowest Predicted Emission:")
print(best_route)
import matplotlib.pyplot as plt

# Plot actual vs predicted
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Emissions")
plt.ylabel("Predicted Emissions")
plt.title("Actual vs Predicted Carbon Emissions")
plt.grid(True)
plt.show()
model.save("carbon_emission_model.h5")
