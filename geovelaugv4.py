import pandas as pd
import numpy as np
from haversine import haversine

# Load synthetic dataset
df = pd.read_csv("augmented_login_data_v4.csv")

# Ensure data is sorted by user_id and login_time
df['login_time'] = pd.to_datetime(df['login_time'])
df = df.sort_values(by=['user_id', 'login_time'])

# Compute geo-velocity (speed in km/h)
df['geo_velocity'] = np.nan  # Initialize column

for user_id in df['user_id'].unique():
    user_logins = df[df['user_id'] == user_id].sort_values('login_time')
    
    prev_location = None
    prev_time = None
    
    for idx, row in user_logins.iterrows():
        if prev_location is not None:
            distance = haversine(prev_location, (row['latitude'], row['longitude']))  # in km
            time_diff = (row['login_time'] - prev_time).total_seconds() / 3600  # in hours
            
            if time_diff > 0:
                df.at[idx, 'geo_velocity'] = distance / time_diff  # speed in km/h
        
        prev_location = (row['latitude'], row['longitude'])
        prev_time = row['login_time']

# Fill NaN values with 0 (for first login attempts)
df['geo_velocity'].fillna(0, inplace=True)

# Save updated file
df.to_csv("augmented_login_data_v4_with_geo_velocity.csv", index=False)
print("✅ Geo-velocity column added successfully!")
