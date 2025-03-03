import pandas as pd
import mysql.connector

# Connect to MySQL
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="root",
    database="risk_auth_db"
)

query = """
SELECT latitude, longitude, typing_speed, mouse_speed, geo_velocity, 
       HOUR(login_time) AS login_hour, 
       (SELECT COUNT(*) FROM login_attempts la WHERE la.ip_address = l.ip_address) AS ip_frequency
FROM login_attempts l;
"""

df = pd.read_sql(query, conn)
conn.close()

# Save to CSV
df.to_csv("real_login_data.csv", index=False)
print("✅ real_login_data.csv has been created successfully.")
