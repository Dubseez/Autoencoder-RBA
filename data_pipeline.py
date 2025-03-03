import pandas as pd
import mysql.connector
from sklearn.preprocessing import MinMaxScaler

# Connect to MySQL Database
db_connection = mysql.connector.connect(
    host='localhost',
    user='root',
    password='root',
    database='risk_auth_db'
)
cursor = db_connection.cursor()

# Fetch login data
query = """
SELECT user_id, ip_address, latitude, longitude, timezone, device_info, typing_speed, mouse_speed, login_time 
FROM login_attempts
"""
cursor.execute(query)
data = cursor.fetchall()

# Close DB connection
cursor.close()
db_connection.close()

# Convert to DataFrame
columns = ["user_id", "ip_address", "latitude", "longitude", "timezone", "device_info", "typing_speed", "mouse_speed", "login_time"]
df = pd.DataFrame(data, columns=columns)

# Handle missing values
df.fillna({
    "latitude": 0.0,
    "longitude": 0.0,
    "typing_speed": df["typing_speed"].median(),
    "mouse_speed": df["mouse_speed"].median(),
    "timezone": "UTC",
    "device_info": "Unknown"
}, inplace=True)


# Save processed data for model training
df.to_csv("processed_login_data.csv", index=False)

print("Data pipeline setup complete. Processed data saved as processed_login_data.csv")
