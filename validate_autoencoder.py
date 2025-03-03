import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# Load scaler, autoencoder model, and IP frequency mapping
scaler = joblib.load("scaler.pkl")
autoencoder = load_model("autoencoder_model.keras")

try:
    ip_frequencies = joblib.load("ip_frequencies.pkl")  # Load precomputed IP frequency mapping
except FileNotFoundError:
    ip_frequencies = {}  # If missing, initialize as empty dictionary

def detect_anomalies(data):
    """Detects anomalies using the trained autoencoder."""
    # Compute missing columns
    if 'login_hour' not in data:
        data['login_hour'] = pd.to_datetime(data['login_time']).dt.hour  # Extract login hour

    if 'ip_frequency' not in data:
        # Compute frequency (normalized) using training distribution
        data["ip_frequency"] = data["ip_address"].map(ip_frequencies).fillna(1 / len(ip_frequencies))

    # Ensure only the expected features are used
    required_features = ['latitude', 'longitude', 'typing_speed', 'mouse_speed', 'geo_velocity', 'login_hour', 'ip_frequency']
    data = data[required_features]  # Keep only necessary columns

    # Handle missing values (fill with median)
    data.fillna(data.median(), inplace=True)

    # Normalize using the previously fitted scaler
    data_scaled = scaler.transform(data)

    # Reconstruct using the autoencoder
    reconstructed = autoencoder.predict(data_scaled)

    # Compute reconstruction errors (Mean Squared Error per row)
    reconstruction_errors = np.mean(np.square(data_scaled - reconstructed), axis=1)

    return reconstruction_errors

def main():
    # Load real login data with geo_velocity
    df = pd.read_csv("augmented_login_data_v4_with_geo_velocity.csv")

    # Detect anomalies
    reconstruction_errors = detect_anomalies(df)

    # Define threshold for anomalies (e.g., top 5% highest errors)
    threshold = np.percentile(reconstruction_errors, 95)
    df["anomaly_score"] = reconstruction_errors
    df["is_anomalous"] = df["anomaly_score"] > threshold

    # Save results
    df.to_csv("validated_anomalies.csv", index=False)
    print("✅ Validation complete! Anomalies saved in validated_anomalies.csv.")

if __name__ == "__main__":
    main()
