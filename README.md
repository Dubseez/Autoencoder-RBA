🚀 Real-Time Anomaly Detection for Login Security
Overview
This project implements a Risk-Based Authentication (RBA) system using an Autoencoder to detect anomalous login attempts. It analyzes user behavior, location, and device information to determine whether to allow, challenge (MFA), or block a login.

Features
✔️ Anomaly Detection: Uses an Autoencoder to identify suspicious logins.
✔️ Geo-Velocity Analysis: Flags impossible travel speeds.
✔️ Behavioral Analysis: Detects unusual typing and mouse speed patterns.
✔️ Risk Scoring: Assigns risk scores to login attempts.
✔️ Flask API: Provides real-time risk assessment.
✔️ MySQL Integration: Stores login attempts and risk scores.
✔️ Threshold-Based Decision Making: Allows fine-tuning of risk-based authentication.
