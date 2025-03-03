from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from math import radians, sin, cos, sqrt, atan2
import logging
import tensorflow as tf
import numpy as np
import joblib  

app = Flask(__name__)


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+mysqlconnector://root:root@localhost/risk_auth_db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Load Autoencoder Model, Scaler, and IP Frequency Data
with app.app_context():
    print("\U0001F680 Loading Autoencoder Model, Scaler, and IP Frequency Data...")
    autoencoder_model = tf.keras.models.load_model("autoencoder_model.keras")
    scaler = joblib.load("scaler.pkl")  # Load trained scaler
    ip_frequency_dict = joblib.load("ip_frequencies.pkl")  # Load IP frequency data
    print("✅ Model and data loaded successfully!")

# Define LoginAttempts Model
class LoginAttempts(db.Model):
    __tablename__ = 'login_attempts'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.String(255), nullable=False)
    ip_address = db.Column(db.String(50), nullable=False)
    latitude = db.Column(db.Float, nullable=False, default=0.0)
    longitude = db.Column(db.Float, nullable=False, default=0.0)
    timezone = db.Column(db.String(50), nullable=True)
    device_info = db.Column(db.String(255), nullable=True, default='Unknown')
    typing_speed = db.Column(db.Float, nullable=True, default=0.0)
    mouse_speed = db.Column(db.Float, nullable=True, default=0.0)
    geo_velocity = db.Column(db.Float, nullable=True, default=0.0)  
    login_time = db.Column(db.DateTime, default=datetime.utcnow)

# Haversine formula to calculate distance between two coordinates
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth's radius in km
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c  # Distance in km

# Function to detect anomalies using Autoencoder
def detect_anomalies(typing_speed, mouse_speed, latitude, longitude, ip_address, geo_velocity, login_hour):
    ip_freq = ip_frequency_dict.get(ip_address, 0)  # Get IP frequency, default to 0
    input_data = np.array([[latitude, longitude, typing_speed, mouse_speed, geo_velocity, login_hour, ip_freq]])
    input_data = scaler.transform(input_data)  # Normalize input

    reconstructed = autoencoder_model.predict(input_data)
    reconstruction_error = np.mean(np.abs(input_data - reconstructed))

    anomaly_threshold = 0.5  # Tuned threshold
    is_anomalous = reconstruction_error > anomaly_threshold

    return is_anomalous, reconstruction_error

@app.route('/')
def home():
    return jsonify({"message": "Risk-Based Authentication API is Running!"})

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    user_id = data.get("user_id", "Unknown")
    ip_address = data.get("ip_address", "0.0.0.0")
    latitude = data.get("latitude", 0.0)
    longitude = data.get("longitude", 0.0)
    timezone = data.get("timezone", "UTC")
    device_info = data.get("device_info", "Unknown")
    typing_speed = max(0.0, data.get("typing_speed", 0.0))
    mouse_speed = max(0.0, data.get("mouse_speed", 0.0))
    login_time = datetime.utcnow()
    login_hour = login_time.hour

    last_attempt = LoginAttempts.query.filter_by(user_id=user_id).order_by(LoginAttempts.login_time.desc()).first()
    prev_latitude = last_attempt.latitude if last_attempt else None
    prev_longitude = last_attempt.longitude if last_attempt else None
    prev_login_time = last_attempt.login_time if last_attempt else None

    if prev_latitude is not None and prev_longitude is not None and prev_login_time is not None:
        time_diff = (login_time - prev_login_time).total_seconds() / 3600  # Hours
        distance = haversine(prev_latitude, prev_longitude, latitude, longitude)
        geo_velocity = distance / time_diff if time_diff > 0 else 0
    else:
        geo_velocity = 0

    is_anomalous, error_score = detect_anomalies(typing_speed, mouse_speed, latitude, longitude, ip_address, geo_velocity, login_hour)

    if error_score >= 0.5:
        risk_decision = "block"
        reason = "Impossible travel detected" if geo_velocity > 1000 else "Highly unusual login behavior"
    elif 0.28 <= error_score < 0.5:
        risk_decision = "mfa"
        reason = "Moderate anomaly detected"
    elif 0.17 <= error_score < 0.28:
        risk_decision = "allow"
        reason = "Normal login"
    else:
        risk_decision = "mfa"
        reason = "Unusual typing/mouse speed or slight location anomaly"

    if risk_decision in ["mfa", "block"]:
        logging.warning(f"⚠️ Login from {user_id} flagged! (Error Score: {error_score:.5f}, Decision: {risk_decision}, Reason: {reason})")
        return jsonify({
            "status": risk_decision,
            "reason": reason,
            "risk_score": error_score,
            "geo_velocity": geo_velocity
        }), 401 if risk_decision == "mfa" else 403

    new_attempt = LoginAttempts(
        user_id=user_id,
        ip_address=ip_address,
        latitude=latitude,
        longitude=longitude,
        timezone=timezone,
        device_info=device_info,
        typing_speed=typing_speed,
        mouse_speed=mouse_speed,
        geo_velocity=geo_velocity,  
        login_time=login_time
    )
    db.session.add(new_attempt)
    db.session.commit()

    return jsonify({"status": "success", "message": "Login recorded", "geo_velocity": geo_velocity, "risk_score": error_score}), 200

with app.app_context():
    db.create_all()

if __name__ == '__main__':
    print("\n Flask API is running at: http://127.0.0.1:5000/\n")
    app.run(debug=True)
