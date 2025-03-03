import pandas as pd
import random
import datetime

# Load the dataset
df = pd.read_csv("augmented_login_data_v2.csv")

# Define new locations, timezones, and devices for major shifts
major_locations = [
    ("New York, USA", 40.7128, -74.0060, "America/New_York"),
    ("London, UK", 51.5074, -0.1278, "Europe/London"),
    ("Tokyo, Japan", 35.6895, 139.6917, "Asia/Tokyo"),
    ("Sydney, Australia", -33.8688, 151.2093, "Australia/Sydney"),
    ("Cape Town, South Africa", -33.9249, 18.4241, "Africa/Johannesburg"),
    ("Dubai, UAE", 25.276987, 55.296249, "Asia/Dubai"),
    ("Berlin, Germany", 52.5200, 13.4050, "Europe/Berlin"),
    ("Toronto, Canada", 43.65107, -79.347015, "America/Toronto"),
    ("São Paulo, Brazil", -23.5505, -46.6333, "America/Sao_Paulo"),
    ("Mumbai, India", 19.0760, 72.8777, "Asia/Kolkata"),
]

device_types = ["Windows", "macOS", "Linux", "iOS", "Android"]

# Function to generate a random IP address
def generate_random_ip():
    return ".".join(str(random.randint(1, 255)) for _ in range(4))

# Group data by user_id and find the latest login
new_entries = []
for user_id, user_data in df.groupby("user_id"):
    latest_entry = user_data.iloc[-1]  # Get the latest login for the user

    # Pick a random new location that is different from the current
    new_location = random.choice(major_locations)
    new_lat, new_lon, new_timezone = new_location[1], new_location[2], new_location[3]

    # Ensure at least 48-hour shift from latest login time
    old_time = datetime.datetime.strptime(latest_entry["login_time"], "%Y-%m-%d %H:%M:%S")
    new_time = old_time + datetime.timedelta(hours=random.randint(48, 72))

    # Create new entry with major location shift
    new_entry = {
        "user_id": latest_entry["user_id"],
        "ip_address": generate_random_ip(),
        "latitude": new_lat,
        "longitude": new_lon,
        "timezone": new_timezone,
        "device_info": random.choice(device_types),
        "typing_speed": round(random.uniform(2, 15), 2),  # CPS
        "mouse_speed": round(random.uniform(300, 2000), 2),  # PPS
        "login_time": new_time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    new_entries.append(new_entry)

# Append new entries to the original dataset
new_df = pd.DataFrame(new_entries)
final_df = pd.concat([df, new_df], ignore_index=True)

# Sort by user_id to maintain sequential order
final_df = final_df.sort_values(by=["user_id", "login_time"]).reset_index(drop=True)

# Save the new dataset
final_df.to_csv("augmented_login_data_v3.csv", index=False)

print("✅ New dataset with major location shifts saved as 'augmented_login_data_v3.csv'")
