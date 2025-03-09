import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# Parameters
R = 10  # Radius of hydrophone's hearing range in km
c1 = 3  # y-intercept of the first ship's path in km
c2 = -3  # y-intercept of the second ship's path in km
x_min, x_max = -20, 20  # Range of x values for the ship's path in km
frames = 200  # Number of frames for the animation
ship1_start_frame = 0  # Frame when the first ship starts
ship2_start_frame = 100  # Frame when the second ship starts

# Initialize distance lists for plotting
distances1 = []
distances2 = []
min_distances = []

# Figure setup
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw={'width_ratios': [2, 1]})  # Adjust the width ratio of subplots

# Main plot (circle and ship path)
circle = plt.Circle((0, 0), R, color='blue', fill=False)
ax1.add_patch(circle)
ship_path1, = ax1.plot([], [], 'g-', label='Ship 1 Path')
ship_path2, = ax1.plot([], [], 'orange', label='Ship 2 Path')
distance_line1, = ax1.plot([], [], 'r-', label='Distance to Ship 1')
distance_line2, = ax1.plot([], [], 'purple', label='Distance to Ship 2')
ship_point1, = ax1.plot([], [], 'g^', label='Ship 1', markersize=10)  # Ship 1 marker as triangle
ship_point2, = ax1.plot([], [], 'y^', label='Ship 2', markersize=10)  # Ship 2 marker as triangle

# Distance-time plot
distance_plot1, = ax2.plot([], [], 'r-', label='Distance to Ship 1 over Time')
distance_plot2, = ax2.plot([], [], 'purple', label='Distance to Ship 2 over Time')
min_distance_plot, = ax2.plot([], [], 'k-', label='Closest Distance over Time')
horizontal_line = ax2.axhline(y=10, color='blue', linestyle='--', label='Hearing Range (10 km)')

ax2.set_xlim(0, frames)
ax2.set_ylim(0, 15)  # Set y-axis limit to 15 km
ax2.set_xlabel('Time (frames)')
ax2.set_ylabel('Distance (km)')

# Move the legends under the plot
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)  # Set the number of columns for the first subplot's legend to 3
ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25))  # Adjust the position of the legend

# Initialize plot limits for the main plot
ax1.set_xlim(x_min, x_max)
ax1.set_ylim(-R-2, R+2)
ax1.set_aspect('equal')

# Initialize function
def init():
    ship_path1.set_data([], [])
    ship_path2.set_data([], [])
    distance_line1.set_data([], [])
    distance_line2.set_data([], [])
    ship_point1.set_data([], [])
    ship_point2.set_data([], [])
    distance_plot1.set_data([], [])
    distance_plot2.set_data([], [])
    min_distance_plot.set_data([], [])
    return (ship_path1, ship_path2, distance_line1, distance_line2, 
            ship_point1, ship_point2, distance_plot1, distance_plot2, min_distance_plot, horizontal_line)

# Animation function
def animate(i):
    x = np.linspace(x_min, x_max, 400)
    y1 = np.full_like(x, c1)
    y2 = np.full_like(x, c2)
    
    ship_path1.set_data(x, y1)
    ship_path2.set_data(x, y2)
    
    # Ship 1's position at frame i
    if i >= ship1_start_frame:
        x_ship1 = x_min + (i - ship1_start_frame) * (x_max - x_min) / (frames - ship1_start_frame)
        y_ship1 = c1
        ship_point1.set_data(x_ship1, y_ship1)
        distance_to_center1 = np.sqrt(x_ship1**2 + y_ship1**2)
    else:
        x_ship1, y_ship1, distance_to_center1 = np.nan, np.nan, np.inf
    
    # Ship 2's position at frame i
    if i >= ship2_start_frame:
        x_ship2 = x_min + (i - ship2_start_frame) * (x_max - x_min) / (frames - ship2_start_frame)
        y_ship2 = c2
        ship_point2.set_data(x_ship2, y_ship2)
        distance_to_center2 = np.sqrt(x_ship2**2 + y_ship2**2)
    else:
        x_ship2, y_ship2, distance_to_center2 = np.nan, np.nan, np.inf
    
    # Append distances to lists
    distances1.append(distance_to_center1)
    distances2.append(distance_to_center2)
    min_distances.append(min(distance_to_center1, distance_to_center2))
    
    # Update distance lines
    if i >= ship1_start_frame:
        distance_line1.set_data([0, x_ship1], [0, y_ship1])
    if i >= ship2_start_frame:
        distance_line2.set_data([0, x_ship2], [0, y_ship2])
    
    # Update distance-time plots
    distance_plot1.set_data(range(len(distances1)), distances1)
    distance_plot2.set_data(range(len(distances2)), distances2)
    min_distance_plot.set_data(range(len(min_distances)), min_distances)
    
    return (ship_path1, ship_path2, distance_line1, distance_line2, 
            ship_point1, ship_point2, distance_plot1, distance_plot2, min_distance_plot, horizontal_line)

# Create animation
ani = FuncAnimation(fig, animate, init_func=init, frames=frames, interval=50, blit=True)  # Increase speed by decreasing interval

# Save animation as GIF
ani.save('ships_animation.gif', writer=PillowWriter(fps=20))

# Show animation
plt.tight_layout()

plt.show()

