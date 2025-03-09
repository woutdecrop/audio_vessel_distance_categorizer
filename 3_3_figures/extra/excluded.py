import pandas as pd

hourly_intervals_28434 = [
    (pd.to_datetime('2022-08-20 04:00:00'), pd.to_datetime('2022-08-22 04:30:00')),
    (pd.to_datetime('2022-08-22 03:00:00'), pd.to_datetime('2022-08-22 14:00:00')),
    (pd.to_datetime('2022-08-22 18:00:00'), pd.to_datetime('2022-08-22 21:00:00')),
    (pd.to_datetime('2022-08-24 07:40:00'), pd.to_datetime('2022-08-24 07:55:00')),
    (pd.to_datetime('2022-08-28 23:25:00'), pd.to_datetime('2022-08-29 00:00:00')),
    (pd.to_datetime('2022-09-03 08:45:00'), pd.to_datetime('2022-09-03 09:00:00')),
    (pd.to_datetime('2022-09-05 06:00:00'), pd.to_datetime('2022-09-05 20:00:00')),
    # Add more hourly intervals or single hours as needed
]

hourly_intervals_29187 = [
    (pd.to_datetime('2022-10-28 06:00:00'), pd.to_datetime('2022-10-28 13:00:00')),
    # Add more hourly intervals or single hours as needed
]

# (pd.to_datetime('2022-01-24 08:50:00'), pd.to_datetime('2022-01-24 09:10:00')),
# (pd.to_datetime('2022-01-24 15:00:00'), pd.to_datetime('2022-01-24 15:30:00')),

hourly_intervals_15811 = [
    (pd.to_datetime('2022-01-20 08:00:00'), pd.to_datetime('2022-01-20 08:30:00')),
    (pd.to_datetime('2022-01-20 19:30:00'), pd.to_datetime('2022-01-20 20:00:00')),
    (pd.to_datetime('2022-01-24 08:50:00'), pd.to_datetime('2022-01-24 09:10:00')),
    (pd.to_datetime('2022-01-24 15:00:00'), pd.to_datetime('2022-01-24 15:30:00')),
    # (pd.to_datetime('2022-01-26 11:45:00'), pd.to_datetime('2022-01-26 12:05:00')),
    (pd.to_datetime('2022-01-28 07:00:00'), pd.to_datetime('2022-01-28 07:30:00')),
    (pd.to_datetime('2022-01-28 20:30:00'), pd.to_datetime('2022-01-28 22:15:00')),
    # (pd.to_datetime('2022-01-28 23:00:00'), pd.to_datetime('2022-01-28 23:30:00')),
    # (pd.to_datetime('2022-01-30 05:00:00'), pd.to_datetime('2022-01-30 05:30:00')),
    (pd.to_datetime('2022-01-30 17:00:00'), pd.to_datetime('2022-01-30 18:30:00')),
    (pd.to_datetime('2022-01-30 19:45:00'), pd.to_datetime('2022-01-30 20:00:00')),
    (pd.to_datetime('2022-01-30 20:45:00'), pd.to_datetime('2022-01-30 21:00:00')),
    (pd.to_datetime('2022-02-01 01:00:00'), pd.to_datetime('2022-02-01 04:30:00')),
    (pd.to_datetime('2022-02-01 20:00:00'), pd.to_datetime('2022-02-01 22:30:00')),
    (pd.to_datetime('2022-02-03 02:30:00'), pd.to_datetime('2022-02-04 03:00:00')),
    (pd.to_datetime('2022-02-03 06:30:00'), pd.to_datetime('2022-02-04 07:00:00')),
    (pd.to_datetime('2022-02-03 19:30:00'), pd.to_datetime('2022-02-04 00:00:00')),
    (pd.to_datetime('2022-02-03 19:30:00'), pd.to_datetime('2022-02-04 00:00:00')),
    (pd.to_datetime('2022-02-05 05:30:00'), pd.to_datetime('2022-02-05 08:30:00')),
    (pd.to_datetime('2022-02-05 18:00:00'), pd.to_datetime('2022-02-05 18:30:00')),
    (pd.to_datetime('2022-02-07 01:00:00'), pd.to_datetime('2022-02-07 01:30:00')),
    (pd.to_datetime('2022-02-07 07:30:00'), pd.to_datetime('2022-02-07 07:50:00')),
    (pd.to_datetime('2022-02-07 18:00:00'), pd.to_datetime('2022-02-07 18:15:00')),
    # (pd.to_datetime('2022-02-09 15:00:00'), pd.to_datetime('2022-02-09 17:00:00')),
    (pd.to_datetime('2022-02-11 04:00:00'), pd.to_datetime('2022-02-11 05:00:00')),
    (pd.to_datetime('2022-02-11 14:10:00'), pd.to_datetime('2022-02-11 15:00:00')),
    (pd.to_datetime('2022-02-13 23:00:00'), pd.to_datetime('2022-02-13 23:30:00')),
    (pd.to_datetime('2022-02-15 00:45:00'), pd.to_datetime('2022-02-15 01:30:00')),
    (pd.to_datetime('2022-02-15 11:30:00'), pd.to_datetime('2022-02-15 18:00:00')),
    (pd.to_datetime('2022-02-17 00:00:00'), pd.to_datetime('2022-02-17 06:00:00')),
    (pd.to_datetime('2022-02-17 13:30:00'), pd.to_datetime('2022-02-17 14:30:00')),
    (pd.to_datetime('2022-02-19 10:00:00'), pd.to_datetime('2022-02-19 11:30:00')),
    (pd.to_datetime('2022-02-19 14:30:00'), pd.to_datetime('2022-02-19 16:55:00')),
    (pd.to_datetime('2022-02-19 20:00:00'), pd.to_datetime('2022-02-19 20:10:00')),
    (pd.to_datetime('2022-02-19 23:30:00'), pd.to_datetime('2022-02-20 00:00:00')),
    (pd.to_datetime('2022-02-21 08:15:00'), pd.to_datetime('2022-02-21 08:45:00')),
    (pd.to_datetime('2022-02-21 11:00:00'), pd.to_datetime('2022-02-21 16:00:00')),
    (pd.to_datetime('2022-02-21 18:00:00'), pd.to_datetime('2022-02-21 19:30:00')),
    (pd.to_datetime('2022-02-21 06:00:00'), pd.to_datetime('2022-02-21 06:15:00')),
    (pd.to_datetime('2022-02-23 09:00:00'), pd.to_datetime('2022-02-23 12:00:00')),
    (pd.to_datetime('2022-02-23 14:00:00'), pd.to_datetime('2022-02-23 14:20:00')),
    (pd.to_datetime('2022-02-23 18:50:00'), pd.to_datetime('2022-02-23 20:00:00')),
    (pd.to_datetime('2022-02-25 10:30:00'), pd.to_datetime('2022-02-25 10:50:00')),
    (pd.to_datetime('2022-02-27 07:00:00'), pd.to_datetime('2022-02-27 07:10:00')),
    (pd.to_datetime('2022-02-27 10:00:00'), pd.to_datetime('2022-02-27 11:00:00')),
    (pd.to_datetime('2022-03-01 20:00:00'), pd.to_datetime('2022-03-01 20:30:00')),
    (pd.to_datetime('2022-03-01 10:30:00'), pd.to_datetime('2022-03-01 11:00:00')),
    # (pd.to_datetime('2022-03-03 16:30:00'), pd.to_datetime('2022-03-03 17:30:00')),
    (pd.to_datetime('2022-03-07 09:00:00'), pd.to_datetime('2022-03-07 15:00:00')),
    (pd.to_datetime('2022-03-09 06:00:00'), pd.to_datetime('2022-03-09 11:00:00')),
    (pd.to_datetime('2022-03-17 00:00:00'), pd.to_datetime('2022-03-17 03:00:00')),
    (pd.to_datetime('2022-03-17 17:00:00'), pd.to_datetime('2022-03-17 19:30:00')),
    (pd.to_datetime('2022-03-21 03:45:00'), pd.to_datetime('2022-03-21 06:00:00')),
    (pd.to_datetime('2022-03-29 02:20:00'), pd.to_datetime('2022-03-29 02:40:00')),
    (pd.to_datetime('2022-03-29 05:00:00'), pd.to_datetime('2022-03-29 06:30:00')),
    (pd.to_datetime('2022-03-29 10:00:00'), pd.to_datetime('2022-03-29 11:00:00')),
    (pd.to_datetime('2022-04-06 15:40:00'), pd.to_datetime('2022-04-06 16:30:00')),
    (pd.to_datetime('2022-04-22 09:55:00'), pd.to_datetime('2022-04-22 10:05:00')),
    (pd.to_datetime('2022-04-28 15:30:00'), pd.to_datetime('2022-04-28 17:00:00')),
    (pd.to_datetime('2022-04-30 09:30:00'), pd.to_datetime('2022-04-30 10:00:00')),
    (pd.to_datetime('2022-05-02 11:10:00'), pd.to_datetime('2022-05-02 12:00:00')),
    (pd.to_datetime('2022-05-02 22:30:00'), pd.to_datetime('2022-05-02 22:40:00')),
    (pd.to_datetime('2022-05-04 08:00:00'), pd.to_datetime('2022-05-04 09:00:00')),
    (pd.to_datetime('2022-05-10 23:40:00'), pd.to_datetime('2022-05-11 00:00:00')),
    (pd.to_datetime('2022-05-12 15:45:00'), pd.to_datetime('2022-05-10 16:00:00')),
    (pd.to_datetime('2022-05-16 04:00:00'), pd.to_datetime('2022-05-16 05:00:00')),
    (pd.to_datetime('2022-05-16 11:50:00'), pd.to_datetime('2022-05-16 12:15:00')),
    (pd.to_datetime('2022-05-16 19:40:00'), pd.to_datetime('2022-05-16 20:15:00')),
    (pd.to_datetime('2022-05-18 06:00:00'), pd.to_datetime('2022-05-17 14:05:00')),
    # (pd.to_datetime('2022-05-18 06:00:00'), pd.to_datetime('2022-05-18 16:15:00')),
    (pd.to_datetime('2022-05-18 20:00:00'), pd.to_datetime('2022-05-18 21:00:00')),
    (pd.to_datetime('2022-05-20 13:00:00'), pd.to_datetime('2022-05-20 13:15:00')),
]
# hourly_intervals_15811=[]

hourly_intervals_15810 = [
    (pd.to_datetime('2022-01-20 15:10:00'), pd.to_datetime('2022-01-20 16:15:00')),  # Adding a missing entry
    (pd.to_datetime('2022-01-28 06:00:00'), pd.to_datetime('2022-01-28 21:00:00')),
    (pd.to_datetime('2022-02-13 17:00:00'), pd.to_datetime('2022-02-13 17:30:00')),
    (pd.to_datetime('2022-02-15 17:00:00'), pd.to_datetime('2022-02-16 00:00:00')),
    (pd.to_datetime('2022-02-15 12:30:00'), pd.to_datetime('2022-02-15 12:45:00')),
    (pd.to_datetime('2022-02-15 09:30:00'), pd.to_datetime('2022-02-15 10:45:00')),
    # (pd.to_datetime('2022-02-19 00:00:00'), pd.to_datetime('2022-02-19 20:00:00')),
    (pd.to_datetime('2022-02-21 05:50:00'), pd.to_datetime('2022-02-21 06:00:00')),
    (pd.to_datetime('2022-02-21 15:30:00'), pd.to_datetime('2022-02-21 16:00:00')),
    (pd.to_datetime('2022-02-23 22:10:00'), pd.to_datetime('2022-02-23 23:00:00')),
    (pd.to_datetime('2022-03-03 06:00:00'), pd.to_datetime('2022-03-03 12:00:00')),
    (pd.to_datetime('2022-03-05 01:10:00'), pd.to_datetime('2022-03-05 01:50:00')),
    # (pd.to_datetime('2022-03-09 12:00:00'), pd.to_datetime('2022-03-09 14:30:00')),
    (pd.to_datetime('2022-03-11 18:30:00'), pd.to_datetime('2022-03-11 19:30:00')),
    (pd.to_datetime('2022-03-11 11:30:00'), pd.to_datetime('2022-03-11 14:00:00')),
    (pd.to_datetime('2022-03-13 08:30:00'), pd.to_datetime('2022-03-13 08:45:00')),
    (pd.to_datetime('2022-03-17 06:00:00'), pd.to_datetime('2022-03-17 06:00:00')),
    (pd.to_datetime('2022-03-19 03:30:00'), pd.to_datetime('2022-03-19 04:10:00')),
    (pd.to_datetime('2022-03-21 21:30:00'), pd.to_datetime('2022-03-21 22:00:00')),
    (pd.to_datetime('2022-03-23 08:00:00'), pd.to_datetime('2022-03-23 08:30:00')),
    (pd.to_datetime('2022-03-29 06:00:00'), pd.to_datetime('2022-03-29 08:00:00')),
    (pd.to_datetime('2022-03-31 16:00:00'), pd.to_datetime('2022-03-31 23:59:00')),
    (pd.to_datetime('2022-03-29 07:00:00'), pd.to_datetime('2022-03-29 08:00:00')),
    (pd.to_datetime('2022-04-08 10:00:00'), pd.to_datetime('2022-04-08 11:00:00')),
    (pd.to_datetime('2022-04-08 23:00:00'), pd.to_datetime('2022-04-08 23:59:00')),
    (pd.to_datetime('2022-04-12 01:20:00'), pd.to_datetime('2022-04-12 01:40:00')),
    (pd.to_datetime('2022-04-28 15:00:00'), pd.to_datetime('2022-04-29 00:00:00')),
    (pd.to_datetime('2022-04-30 10:00:00'), pd.to_datetime('2022-04-30 11:00:00')),
    (pd.to_datetime('2022-05-02 12:15:00'), pd.to_datetime('2022-05-02 13:00:00')),
    (pd.to_datetime('2022-05-04 06:00:00'), pd.to_datetime('2022-05-04 17:00:00')),
]
# hourly_intervals_15810=[]
hourly_intervals_26981 = [
    (pd.to_datetime('2022-06-24 15:15:00'), pd.to_datetime('2022-06-24 16:00:00')),
    (pd.to_datetime('2022-06-26 00:00:00'), pd.to_datetime('2022-06-26 00:30:00')),
    (pd.to_datetime('2022-06-26 02:15:00'), pd.to_datetime('2022-06-26 02:45:00')),
    (pd.to_datetime('2022-06-26 19:45:00'), pd.to_datetime('2022-06-26 20:10:00')),  # Single hour
]
def calculate_total_hours(intervals):
    total_hours = 0
    for start, end in intervals:
        if start == end:
            total_hours += 24.0  # Add 24 hours (in seconds: 24 * 3600)
        else:
            total_hours += (end - start).total_seconds() / 3600
    return total_hours
# Calculate total hours for each set of intervals
total_hours_28434 = calculate_total_hours(hourly_intervals_28434)
total_hours_29187 = calculate_total_hours(hourly_intervals_29187)
total_hours_15811 = calculate_total_hours(hourly_intervals_15811)
total_hours_15810 = calculate_total_hours(hourly_intervals_15810)
total_hours_26981 = calculate_total_hours(hourly_intervals_26981)

total_hours = total_hours_28434 + total_hours_29187 + total_hours_15811 + total_hours_15810 + total_hours_26981

total_hours



import pandas as pd
from tqdm import tqdm
file_path = r'..\..UC6\figures\samples_time.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)
# Assuming df is already your DataFrame
df['event_time'] = pd.to_datetime(df['event_time'])  # Ensure event_time is in datetime format

# Sort by deployment and event_time
df = df.sort_values(by=['deployment', 'event_time'])
df['date'] = df['event_time'].dt.date
# Initialize a total accumulated time variable
total_accumulated_time = pd.Timedelta(0)

# Set a threshold for gap (5 minutes)
gap_threshold = pd.Timedelta(minutes=5)

# Function to find consecutive hours without large gaps
def find_consecutive_hours(day_group):
    consecutive_hours = []
    current_hour_start = None
    for _, row in day_group.iterrows():
        if current_hour_start is None:
            current_hour_start = row['event_time']
        elif row['event_time'] - current_hour_start <= gap_threshold:
            continue
        else:
            if current_hour_start is not None:
                consecutive_hours.append((current_hour_start, row['event_time']))
            current_hour_start = row['event_time']
    # Append the last consecutive segment
    if current_hour_start is not None:
        consecutive_hours.append((current_hour_start, day_group['event_time'].iloc[-1]))
    return consecutive_hours

# Group by deployment and date
deployments = df['deployment'].unique()

for deployment in tqdm(deployments, desc="Processing deployments"):
    daily_groups = df[df['deployment'] == deployment].groupby('date')
    for _, day_group in daily_groups:
        consecutive_hours = find_consecutive_hours(day_group)
        for start_time, end_time in consecutive_hours:
            total_accumulated_time += (end_time - start_time)

print(f"Total accumulated time without gaps larger than 5 minutes: {total_accumulated_time}")
print(f"removed time {total_hours} devided py total time {total_accumulated_time} is", total_hours/total_accumulated_time)


import pandas as pd

# Given data
total_accumulated_time = pd.Timedelta('113 days 07:50:36.868277')
total_hours = 337.75

# Convert total_accumulated_time to hours
total_accumulated_hours = total_accumulated_time.total_seconds() / 3600

# Divide total_accumulated_hours by total_hours
result = total_accumulated_hours / total_hours

print(result)
