import pandas as pd
import random
from datetime import datetime, timedelta
from geopy.distance import geodesic

# Load existing data
file_path = "processed_login_data.csv"
df = pd.read_csv(file_path)

# Define typing and mouse speed ranges
def realistic_typing_speed():
    return round(random.uniform(2.5, 15), 2)  # Characters per second (CPS)

def realistic_mouse_speed():
    return round(random.uniform(300, 2000), 2)  # Pixels per second (PPS)

# Generate new entries
new_entries = []
existing_user_ids = sorted(df['user_id'].unique())   #gives list of unique user ids
user_id_counter = max(existing_user_ids) + 1

for user_id in existing_user_ids:
    user_data = df[df['user_id'] == user_id].sort_values(by='login_time', ascending=True)   #retrieves all records of that userid, sorts by logintime
    latest_entry = user_data.iloc[-1]          #gets latest value
    
    new_latitude = latest_entry['latitude'] + random.uniform(-0.05, 0.05)
    new_longitude = latest_entry['longitude'] + random.uniform(-0.05, 0.05)
    
    distance_shift_km = geodesic((latest_entry['latitude'], latest_entry['longitude']), (new_latitude, new_longitude)).km  #calcualtes distance between latest and update locations
    time_shift_minutes = distance_shift_km * 10
    
    new_login_time = datetime.strptime(latest_entry['login_time'], "%Y-%m-%d %H:%M:%S") + timedelta(minutes=time_shift_minutes)
    
    new_entries.append({
        "user_id": user_id,
        "ip_address": latest_entry['ip_address'],
        "latitude": round(new_latitude, 6),
        "longitude": round(new_longitude, 6),
        "timezone": latest_entry['timezone'],
        "device_info": latest_entry['device_info'],
        "typing_speed": realistic_typing_speed(),
        "mouse_speed": realistic_mouse_speed(),
        "login_time": new_login_time.strftime("%Y-%m-%d %H:%M:%S")
    })

# Convert new data to DataFrame
new_df = pd.DataFrame(new_entries)

# Merge and sort by user_id
final_df = pd.concat([df, new_df]).sort_values(by=['user_id', 'login_time']).reset_index(drop=True)

# Save to CSV
final_df.to_csv("augmented_login_data.csv", index=False)

print("Augmented data saved successfully!")
