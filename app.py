from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from math import radians, sin, cos, sqrt, atan2
import logging
import tensorflow as tf
import numpy as np
import joblib  

app = Flask(__name__)


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+mysqlconnector://root:root@localhost/risk_auth_db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)


with app.app_context():
    print(" Loading Autoencoder Model, Scaler, and IP Frequency Data...")
    autoencoder_model = tf.keras.models.load_model("autoencoder_model.keras")
    scaler = joblib.load("scaler.pkl")  # Load trained scaler
    ip_frequency_dict = joblib.load("ip_frequencies.pkl")  # Load IP frequency data
    print(" Model and data loaded successfully!")


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

# Haversine formula to calculate distance (in km) between two coordinates
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth's radius in km
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

# Function to detect anomalies using Autoencoder
def detect_anomalies(typing_speed, mouse_speed, latitude, longitude, ip_address, geo_velocity, login_hour):
    ip_freq = ip_frequency_dict.get(ip_address, 0)  # Default frequency is 0
    input_data = np.array([[latitude, longitude, typing_speed, mouse_speed, geo_velocity, login_hour, ip_freq]])
    input_data = scaler.transform(input_data)  # Normalize input
    reconstructed = autoencoder_model.predict(input_data)
    reconstruction_error = np.mean(np.abs(input_data - reconstructed))
    anomaly_threshold = 0.5  # Tuned threshold (not directly used in decision here)
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
    latitude = float(data.get("latitude", 0.0))
    longitude = float(data.get("longitude", 0.0))
    timezone = data.get("timezone", "UTC")
    device_info = data.get("device_info", "Unknown")
    typing_speed = max(0.0, float(data.get("typing_speed", 0.0)))
    mouse_speed = max(0.0, float(data.get("mouse_speed", 0.0)))
    login_time = datetime.utcnow()
    login_hour = login_time.hour

    # Fetch last login attempt for the user
    last_attempt = LoginAttempts.query.filter_by(user_id=user_id).order_by(LoginAttempts.login_time.desc()).first()
    prev_latitude = last_attempt.latitude if last_attempt else None
    prev_longitude = last_attempt.longitude if last_attempt else None
    prev_ip = last_attempt.ip_address if last_attempt else None
    prev_device = last_attempt.device_info if last_attempt else None
    prev_timezone = last_attempt.timezone if last_attempt else None
    prev_login_time = last_attempt.login_time if last_attempt else None

    # Calculate geo-velocity (travel speed in km/h)
    geo_velocity = 0
    if prev_latitude is not None and prev_longitude is not None and prev_login_time is not None:
        distance = haversine(prev_latitude, prev_longitude, latitude, longitude)
        time_diff = (login_time - prev_login_time).total_seconds() / 3600  # in hours
        if time_diff > 0:
            geo_velocity = distance / time_diff

        if geo_velocity > 1000:
            return jsonify({
                "status": "block",
                "reason": "Impossible travel detected (geo-velocity too high)",
                "geo_velocity": geo_velocity
            }), 403

    # Apply rule-based risk scoring as specified:
    # +2 for IP change, +3 for device change, +3 for timezone change, +5 for location change
    risk_score = 0
    changes = []
    if prev_ip and (prev_ip != ip_address):
        risk_score += 2
        changes.append("IP Address Changed")
    if prev_device and (prev_device != device_info):
        risk_score += 3
        changes.append("Device Info Changed")
    if prev_timezone and (prev_timezone != timezone):
        risk_score += 3
        changes.append("Timezone Changed")
    if prev_latitude is not None and prev_longitude is not None and ((latitude != prev_latitude) or (longitude != prev_longitude)):
        risk_score += 5
        changes.append("Location Changed")
    
    # Anomaly detection using Autoencoder (behavioral features including speeds)
    is_anomalous, error_score = detect_anomalies(typing_speed, mouse_speed, latitude, longitude, ip_address, geo_velocity, login_hour)
    
    # Decision logic depends on whether any rule-based changes occurred:
    if not changes:
        # No changes: use behavioral thresholds
        # Here we interpret error_score directly as the risk score
        total_risk_score = error_score
        if total_risk_score < 0.101 or total_risk_score >= 0.5:
            risk_decision = "block"
            reason = "High-risk login detected (behavioral anomaly)"
        elif 0.101 <= total_risk_score < 0.28:
            risk_decision = "allow"
            reason = "Normal login (behavioral anomaly within acceptable range)"
        elif 0.28 <= total_risk_score < 0.5:
            risk_decision = "mfa"
            reason = "Moderate anomaly detected (behavioral anomaly)"
    else:
        # Rule-based changes exist: combine autoencoder error and rule-based risk
        total_risk_score = error_score + risk_score
        if total_risk_score >= 8:
            risk_decision = "block"
            reason = "High-risk login detected"
        elif total_risk_score >= 3:
            risk_decision = "mfa"
            reason = "Moderate anomaly detected"
        else:
            risk_decision = "allow"
            reason = "Normal login"
    
    
    if risk_decision == "allow":
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
    else:
        logging.info(f"Login attempt not stored due to decision: {risk_decision}")

    
    breakdown = {
        "autoencoder_error": error_score,
        "rule_based_risk": risk_score if changes else 0,
        "total_risk_score": total_risk_score
    }
    return jsonify({
        "status": risk_decision,
        "reason": reason,
        "risk_score": total_risk_score,
        "changes": changes,
        "geo_velocity": geo_velocity,
        "breakdown": breakdown
    }), 401 if risk_decision == "mfa" else 403 if risk_decision == "block" else 200


with app.app_context():
    db.create_all()

if __name__ == '__main__':
    print("\n Flask API is running at: http://127.0.0.1:5000/\n")
    app.run(debug=True)
