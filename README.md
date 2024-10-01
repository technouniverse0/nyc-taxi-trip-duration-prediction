# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load the dataset
file_path = r"C:\Users\bhavik\Downloads\nyc_taxi_trip_duration.csv"  # Modify the path if needed
nyc_data = pd.read_csv(file_path)

# Define Haversine formula to calculate distance between two geo-coordinates
def haversine(lon1, lat1, lon2, lat2):
    # Radius of the Earth in kilometers
    R = 6371.0
    # Convert degrees to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    # Differences between coordinates
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    # Haversine formula
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    # Distance in kilometers
    distance = R * c
    return distance

# Calculate the distance for each row in the dataset
nyc_data['distance_km'] = haversine(nyc_data['pickup_longitude'], nyc_data['pickup_latitude'],
                                    nyc_data['dropoff_longitude'], nyc_data['dropoff_latitude'])

# Convert 'pickup_datetime' to datetime and extract features
nyc_data['pickup_datetime'] = pd.to_datetime(nyc_data['pickup_datetime'])

# Extract hour, day of the week, and month from the pickup_datetime
nyc_data['pickup_hour'] = nyc_data['pickup_datetime'].dt.hour
nyc_data['pickup_day_of_week'] = nyc_data['pickup_datetime'].dt.dayofweek
nyc_data['pickup_month'] = nyc_data['pickup_datetime'].dt.month

# Select the features for modeling
features = ['passenger_count', 'distance_km', 'pickup_hour', 'pickup_day_of_week', 'pickup_month']
target = 'trip_duration'

# Split the dataset into features (X) and target (y)
X = nyc_data[features]
y = nyc_data[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_lr = lr_model.predict(X_test)

# Calculate the R^2 score for the Linear Regression model
r2_lr = r2_score(y_test, y_pred_lr)

# Print the R^2 score
print(f'R² Score for Linear Regression Model: {r2_lr}')
# Remove extreme outliers
nyc_data = nyc_data[(nyc_data['trip_duration'] > 60) & (nyc_data['trip_duration'] < 7200)]  # Keep trips between 1 minute and 2 hours
nyc_data['log_trip_duration'] = np.log1p(nyc_data['trip_duration'])  # Log-transform trip duration
# Extract more features from pickup_datetime
nyc_data['pickup_day'] = nyc_data['pickup_datetime'].dt.day

# Create a rush hour flag (1 = rush hour, 0 = non-rush hour)
def is_rush_hour(hour):
    return 1 if (7 <= hour <= 9) or (16 <= hour <= 19) else 0

nyc_data['rush_hour'] = nyc_data['pickup_hour'].apply(is_rush_hour)
from sklearn.ensemble import RandomForestRegressor

# Select features for the model
features = ['passenger_count', 'distance_km', 'pickup_hour', 'pickup_day_of_week', 'pickup_month', 'pickup_day', 'rush_hour']
X = nyc_data[features]
y = nyc_data['log_trip_duration']  # Use log-transformed target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions and evaluate
y_pred_rf = rf_model.predict(X_test)
r2_rf = r2_score(y_test, y_pred_rf)

print(f'R² Score for Random Forest Model: {r2_rf}')
from sklearn.ensemble import GradientBoostingRegressor

# Train Gradient Boosting Regressor
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)

# Make predictions and evaluate
y_pred_gb = gb_model.predict(X_test)
r2_gb = r2_score(y_test, y_pred_gb)

print(f'R² Score for Gradient Boosting Model: {r2_gb}')
