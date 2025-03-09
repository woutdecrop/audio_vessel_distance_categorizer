import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import calendar
import random
import matplotlib.cm as cm

# Path to the CSV file
file_path = r'..\..UC6\figures\samples_time.csv'
random.seed(42)  # Set a specific seed for reproducibility
# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Convert event_time to datetime
df['event_time'] = pd.to_datetime(df['event_time'])

# Extract date
df['date'] = df['event_time'].dt.date

df['DayOfYear'] = df['event_time'].dt.dayofyear

# Group data by station, day of year, and deployment
grouped = df.groupby(['station', 'DayOfYear', 'deployment']).size().unstack(fill_value=0)

# Additional dates
grafton_dates = [
    "2022-01-22",
    "2022-02-07",
    "2022-02-03",
    '2022-02-11',
    "2022-02-23",
    "2022-03-15",
    "2022-04-18",
    "2022-08-28",
    "2022-09-15",
    "2022-10-30"
]

gardencity_dates = [
    "2022-01-22",
    "2022-01-26",
    "2022-02-11",
    "2022-02-25",
    "2022-03-13",
    "2022-03-25",
    "2022-04-24",
    "2022-05-04",
    "2022-05-10",
    "2022-05-20"
]

random.shuffle(grafton_dates)
random.shuffle(gardencity_dates)

half_grafton = len(grafton_dates) // 2
half_gardencity = len(gardencity_dates) // 2

grafton_val = grafton_dates[:half_grafton]
grafton_val.append('2022-04-06')
grafton_test = grafton_dates[half_grafton:]
grafton_test.append('2022-09-13')

gardencity_val = gardencity_dates[:half_gardencity]
gardencity_val.append('2022-04-06')

gardencity_test = gardencity_dates[half_gardencity:]

desired_pairs_val = [('Grafton', date) for date in grafton_val] + [('GardenCity', date) for date in gardencity_val]
desired_pairs_test = [('Grafton', date) for date in grafton_test] + [('GardenCity', date) for date in gardencity_test]

# Function to create circular plot
def create_year_clock(ax, data, station, desired_pairs_val, desired_pairs_test, title_fontsize, legend_fontsize):
    num_days = 365
    angles = np.linspace(np.pi/2, -3*np.pi/2, num_days, endpoint=False)
    
    # Get unique deployments
    deployments = sorted(df['deployment'].unique())

    # Create a categorical colormap with unique colors for each deployment
    cmap = plt.cm.get_cmap('tab20', len(deployments))
    deployment_text_added = set()  # Set to track if text is already added for deployment
    colormap = cm.get_cmap('tab10')
    train_color = colormap(0)  # Blue
    val_color = colormap(2)    # Green
    test_color = colormap(1)   # Orange
    for index, count in data.items():
        deployment, day = index
        if count > 5:
            angle = angles[day - 1]
            
            # Determine color based on training, validation, and test pairs
            date_str = pd.Timestamp('2022') + pd.to_timedelta(day - 1, unit='D')
            date_str = date_str.strftime('%Y-%m-%d')
            pair = (station, date_str)
            if pair in desired_pairs_val:
                color = val_color
            elif pair in desired_pairs_test:
                color = test_color
            else:
                color = train_color
            ax.plot([0, np.cos(angle)], [0, np.sin(angle)], color=color, linewidth=3.5)

    for i in range(12):
        # Calculate the angle for the current month (clock position)
        month_angle = np.pi/2 - i * (2 * np.pi / 12)
        
        # Calculate the x and y positions for the month label
        x_pos = 1.10 * np.cos(month_angle)
        y_pos = 1.10 * np.sin(month_angle)
        
        # Determine the rotation angle based on the clock position
        if 0 <= i < 4:  # January to April (outward)
            rotation_angle = np.degrees(month_angle) - 90  # Rotate outward by 90 degrees
        elif 4 <i <=7:
            rotation_angle = np.degrees(month_angle) + 90 
            
        elif 7 <= i < 10:  # August to October (outward)
            rotation_angle = np.degrees(month_angle) + 90  # Rotate outward by 90 degrees

        else: # May, June, July, November, December (inward, flipped 180 degrees)
            rotation_angle = np.degrees(month_angle) - 90  # Rotate inward by 90 degrees (or 270)
            print(month_angle,i)
        # Place the text with the calculated rotation
        ax.text(x_pos, y_pos, calendar.month_abbr[i+1], ha='center', va='center', fontsize=legend_fontsize, rotation=rotation_angle)
        
        # Plot the month lines (dashed)
        ax.plot([0, 1.1 * np.cos(month_angle)], [0, 1.1 * np.sin(month_angle)], color='gray', linestyle='--', linewidth=0.5)


    # Create legend for train, validation, and test
    train_line = plt.Line2D([0], [0], color=train_color, linewidth=3.5, label='Train')
    val_line = plt.Line2D([0], [0], color=val_color, linewidth=3.5, label='Validation')
    test_line = plt.Line2D([0], [0], color=test_color, linewidth=3.5, label='Test')

    # ax.legend(handles=[train_line, val_line, test_line], loc='upper left', fontsize=legend_fontsize)  # Legend font size
    # Hide the ticks and labels on both x and y axes
    ax.set_xticks([])  # Remove x-axis ticks
    ax.set_yticks([])  # Remove y-axis ticks
    return ax

# Create circular plots for each station
stations = grouped.index.get_level_values('station').unique()
fig, axs = plt.subplots(1, 2, figsize=(16, 8))  # Larger figure size

title_fontsize = 30  # Set title font size
legend_fontsize = 30  # Set legend font size
under=30
for ax, station in zip(axs, stations):
    data = grouped.loc[station].unstack(fill_value=0)
    ax = create_year_clock(ax, data, station, desired_pairs_val, desired_pairs_test, title_fontsize, legend_fontsize)
    
    # Add title to each subplot
    ax.set_title(station, fontsize=title_fontsize)

deployments = sorted(df['deployment'].unique())
cmap = plt.cm.get_cmap('tab20', len(deployments))
deployment_text_added = set()  # Set to track if text is already added for deployment
colormap = cm.get_cmap('tab10')
train_color = colormap(0)  # Blue
val_color = colormap(2)    # Green
test_color = colormap(1)   # Orange
# Create a single figure-wide legend below the plots
train_line = plt.Line2D([0], [0], color=train_color, linewidth=3.5, label='Train')
val_line = plt.Line2D([0], [0], color=val_color, linewidth=3.5, label='Validation')
test_line = plt.Line2D([0], [0], color=test_color, linewidth=3.5, label='Test')


fig.legend(handles=[train_line, val_line, test_line], loc='upper center', fontsize=under, ncol=3, bbox_to_anchor=(0.5, 0.13))  # Legend below the figure

# Adjust the layout to make space for the legend at the bottom
plt.subplots_adjust(wspace=0.02, bottom=0.15)  # Adjust bottom to make room for the legend

# Set the aspect ratio to 'equal' to maintain the plot's proportions
ax.set_aspect('equal')

# Save the figure with 'bbox_inches' to tightly fit the content
save_path = r'3_3_figures\data_collection\deployments_over_time_bigger.png'
plt.savefig(save_path, dpi=350, bbox_inches="tight")


