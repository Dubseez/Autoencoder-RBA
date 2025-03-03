import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from geopy.distance import geodesic
from datetime import datetime

# Load trained model and preprocessing objects
autoencoder = load_model("autoencoder_model.keras")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")
ip_frequencies = joblib.load("ip_frequencies.pkl")

# Load test dataset
df_test = pd.read_csv("test_login_data.csv")

# Preserve raw coordinates before processing
df_test["raw_latitude"] = df_test["latitude"]
df_test["raw_longitude"] = df_test["longitude"]

# Encode categorical features
df_test["ip_address"] = df_test["ip_address"].map(ip_frequencies).fillna(0.0001)  # Unseen IPs get low frequency

for col in ["timezone", "device_info"]:
    if col in label_encoders:
        # Replace unseen labels with 'Unknown'
        df_test[col] = df_test[col].apply(lambda x: x if x in label_encoders[col].classes_ else "Unknown")
        # Ensure 'Unknown' is in the classes
        if "Unknown" not in label_encoders[col].classes_:
            label_encoders[col].classes_ = np.append(label_encoders[col].classes_, "Unknown")
        df_test[col] = label_encoders[col].transform(df_test[col])

# Compute Geo-Velocity using raw coordinates
df_test = df_test.sort_values(by=["user_id", "login_time"])

# Compute previous raw coordinates and login time
df_test["prev_raw_latitude"] = df_test.groupby("user_id")["raw_latitude"].shift(1)
df_test["prev_raw_longitude"] = df_test.groupby("user_id")["raw_longitude"].shift(1)
df_test["prev_login_time"] = df_test.groupby("user_id")["login_time"].shift(1)

#  Do NOT drop NaN values to prevent data leakage

def calculate_speed(row):
    try:
        if pd.isna(row["prev_raw_latitude"]) or pd.isna(row["prev_raw_longitude"]) or pd.isna(row["prev_login_time"]):
            return 0.0  # No previous location → no velocity
        
        prev_location = (row["prev_raw_latitude"], row["prev_raw_longitude"])
        current_location = (row["raw_latitude"], row["raw_longitude"])
        
        prev_time = datetime.strptime(row["prev_login_time"], "%d-%m-%Y %H:%M")
        curr_time = datetime.strptime(row["login_time"], "%d-%m-%Y %H:%M")

        time_diff = (curr_time - prev_time).total_seconds() / 3600.0  # Convert to hours
        if time_diff <= 0:
            print(f"🚨 Warning: Negative or zero time_diff ({time_diff}), setting geo_velocity to 0")
            return 0.0  

        distance = geodesic(prev_location, current_location).km
        return distance / time_diff

    except Exception as e:
        print(f"Error in calculate_speed: {e}")
        return 0.0

df_test["geo_velocity"] = df_test.apply(calculate_speed, axis=1)

# Extract login hour
df_test["login_hour"] = pd.to_datetime(df_test["login_time"], format="%d-%m-%Y %H:%M").dt.hour

# Select numerical features for the autoencoder
df_test["ip_frequency"] = df_test["ip_address"]  # Already mapped from ip_frequencies.pkl
numerical_cols = ["latitude", "longitude", "typing_speed", "mouse_speed", "geo_velocity", "login_hour", "ip_frequency"]


# Normalize using the same scaler from training
df_test[numerical_cols] = scaler.transform(df_test[numerical_cols])

# Prepare test data for prediction
X_test = df_test[numerical_cols].values

# Compute reconstruction error (Mean Squared Error)
reconstructions = autoencoder.predict(X_test)
mse = np.mean(np.power(X_test - reconstructions, 2), axis=1)
df_test["risk_score"] = mse

# Define risk thresholds and assign risk decisions
def risk_category(score, threshold=0.57):  # Increased threshold to reduce false positives
    if score < threshold:
        return "Allow"
    elif score < threshold * 2:
        return "MFA"
    else:
        return "Block"

df_test["risk_decision"] = df_test["risk_score"].apply(risk_category)

# Save the results
df_test.drop(columns=["latitude", "longitude"], inplace=True)

df_test.to_csv("test_results.csv", index=False)
print(" Risk scoring complete. Results saved to test_results.csv")
