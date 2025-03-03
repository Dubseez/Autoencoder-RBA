import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import joblib
from datetime import datetime
from geopy.distance import geodesic

# Load dataset
df = pd.read_csv("augmented_login_data_v4.csv")
print(f"✅ Dataset loaded. Shape: {df.shape}")

# Encode categorical features
label_encoders = {}
categorical_cols = ["ip_address", "timezone", "device_info"]
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
joblib.dump(label_encoders, "label_encoders.pkl")
print("✅ Label encoders saved.")

# Compute frequency of each IP address & add as feature
ip_frequencies = df["ip_address"].value_counts(normalize=True).to_dict()
df["ip_frequency"] = df["ip_address"].map(ip_frequencies)
joblib.dump(ip_frequencies, "ip_frequencies.pkl")

# Convert `login_time` to datetime
df["login_time"] = pd.to_datetime(df["login_time"], format="%d-%m-%Y %H:%M", dayfirst=True, errors="coerce")


# Sort by user_id and login_time for sequential processing
df = df.sort_values(by=["user_id", "login_time"])
df["prev_latitude"] = df.groupby("user_id")["latitude"].shift(1)
df["prev_longitude"] = df.groupby("user_id")["longitude"].shift(1)
df["prev_login_time"] = df.groupby("user_id")["login_time"].shift(1)

# Drop NaN values (ensures every entry has a valid previous login)
df.dropna(inplace=True)
if df.empty:
    raise ValueError(" Error: The dataset is empty after preprocessing! Check data loading.")

def calculate_speed(row):
    try:
        prev_location = (row["prev_latitude"], row["prev_longitude"])
        current_location = (row["latitude"], row["longitude"])
        
        # Ensure valid datetime values
        if pd.isnull(row["prev_login_time"]) or pd.isnull(row["login_time"]):
            return 0.0
        
        time_diff = (row["login_time"] - row["prev_login_time"]).total_seconds() / 3600.0
        distance = geodesic(prev_location, current_location).km
        
        return distance / time_diff if time_diff > 0 else 0.0
    except:
        return 0.0

df["geo_velocity"] = df.apply(calculate_speed, axis=1)
print("✅ Geo-velocity computed.")

# Extract login hour
df["login_hour"] = df["login_time"].dt.hour

# Normalize numerical features
numerical_cols = ["latitude", "longitude", "typing_speed", "mouse_speed", "geo_velocity", "login_hour", "ip_frequency"]
scaler = MinMaxScaler()

df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
joblib.dump(scaler, "scaler.pkl")
print("✅ Numerical features normalized and scaler saved.")

# Prepare training data
X = df[numerical_cols].values  #Converts the DataFrame into a NumPy array
if X.shape[0] == 0:
    raise ValueError(" Error: No training data available after preprocessing!")

# Train-validation split
X_train, X_val = train_test_split(X, test_size=0.1, random_state=42)

# Define a deeper Autoencoder model
input_dim = X_train.shape[1]   # Gets number of features 
input_layer = Input(shape=(input_dim,))  #  Defines the input layer with the same number of features
encoded = Dense(16, activation='relu')(input_layer)  #First hidden layer with 16 neurons, using 'relu' activation
encoded = Dense(8, activation='relu')(encoded)
encoded = Dense(4, activation='relu')(encoded)
decoded = Dense(8, activation='relu')(encoded)
decoded = Dense(16, activation='relu')(decoded)
decoded = Dense(input_dim, activation='sigmoid')(decoded)

autoencoder = Model(input_layer, decoded)  #Connects the input layer to the output layer.
autoencoder.compile(optimizer='adam', loss='mse')

# Train the model
autoencoder.fit(X_train, X_train, epochs=50, batch_size=32, shuffle=True, validation_data=(X_val, X_val))

# Save model
autoencoder.save("autoencoder_model.keras")
print(" Autoencoder training complete. Model saved successfully!")
