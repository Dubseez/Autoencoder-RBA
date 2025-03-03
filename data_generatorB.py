import pandas as pd
import random
from datetime import datetime, timedelta

# Load the existing dataset
file_path = "augmented_login_data.csv"
df = pd.read_csv(file_path)

# Define global locations with timezones
locations = [
    ("New York, USA", 40.7128, -74.0060, "America/New_York"),
    ("London, UK", 51.5074, -0.1278, "Europe/London"),
    ("Tokyo, Japan", 35.6895, 139.6917, "Asia/Tokyo"),
    ("Sydney, Australia", -33.8688, 151.2093, "Australia/Sydney"),
    ("Berlin, Germany", 52.5200, 13.4050, "Europe/Berlin"),
    ("Cape Town, South Africa", -33.9249, 18.4241, "Africa/Johannesburg"),
    ("Toronto, Canada", 43.651070, -79.347015, "America/Toronto"),
    ("São Paulo, Brazil", -23.5505, -46.6333, "America/Sao_Paulo"),
    ("Dubai, UAE", 25.276987, 55.296249, "Asia/Dubai"),
    ("Mumbai, India", 19.0760, 72.8777, "Asia/Mumbai")
]

# Generate a more varied set of IP addresses
def generate_ip():
    return f"{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}"

# Define varied device types
devices = ["Windows", "macOS", "Linux", "iOS", "Android"]

# Generate new entries with major shifts
new_entries = []
user_groups = df.groupby("user_id")

for user_id, group in user_groups:
    latest_entry = group.iloc[-1].to_dict()  # Get the last login for each user
    new_location = random.choice(locations)  # Select a new country
    
    old_time = datetime.strptime(latest_entry["login_time"], "%Y-%m-%d %H:%M:%S")
    new_time = old_time + timedelta(hours=random.randint(48, 72))  # Shift time by at least 48 hours
    
    new_entry = {
        "user_id": user_id,
        "ip_address": generate_ip(),
        "latitude": new_location[1],
        "longitude": new_location[2],
        "timezone": new_location[3],
        "device_info": random.choice(devices),
        "typing_speed": round(random.uniform(2, 15), 2),  # CPS
        "mouse_speed": round(random.uniform(300, 2000), 2),  # PPS
        "login_time": new_time.strftime("%Y-%m-%d %H:%M:%S")
    }
    new_entries.append(new_entry)

# Append new data to the original DataFrame
new_df = pd.DataFrame(new_entries)
final_df = pd.concat([df, new_df], ignore_index=True)

# Ensure user IDs remain sequential
final_df = final_df.sort_values(by=["user_id", "login_time"]).reset_index(drop=True)

# Save the updated dataset
final_df.to_csv("augmented_login_data_v2.csv", index=False)

print("✅ New dataset with major location shifts saved as 'augmented_login_data_v2.csv'")
