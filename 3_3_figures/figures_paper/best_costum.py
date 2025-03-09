import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
def custom_growth(x, a=5, b=0.625):
    if x < b:
        return (0.2 / b) * x
    elif x <= 1:
        return 0.2 + (1 - 0.2) * (1 - np.exp(-a * (x - b))) / (1 - np.exp(-a * (1 - b)))
    else:
        return 1

def exponential_growth(x):
    return x ** 2

x_values = np.linspace(0, 1.0001, 500)
highlight_x = np.arange(0.1, 1.0001, 0.1)

plt.figure(figsize=(20, 10))

b_values = np.array([ 0.4, 0.5, 0.6, 0.7])
# b_values = np.array([0.5])

# Updated ranges for a based on selected values
a_ranges = {
    0.4: np.array([-3.0, 1.0]),
    0.5: np.array([-2,-1,0.1,1, 2.0]),
    0.6: np.array([-2.0, 2.0]),
    0.7: np.array([-3.0, 1.0]),
}


# Updated b_values
b_values = np.array([ 0.5])
# b_values = np.array([0.5])

# Updated ranges for a based on selected values
a_ranges = {
    0.5: np.array([1.4]),
}

# Fixed alpha value and line thickness
alpha_value = 0.8
line_thickness = 3.0

# Create a list to store legend handles
legend_handles = []

colormap = cm.get_cmap('tab10')
train_color = colormap(0)  # Blue
val_color = colormap(2)    # Green

# Define color map for green and red
def get_color(a):
    if a > 0:  # For positive 'a'
        if a == 0.9:
            return '#b2e0b2'  # Light green for x < 0.1 (less fluorescent)
        elif a == 0.2:
            return '#66cdaa'  # Normal green for 0.1 <= x < 1
        elif a == 1.4:
            return '#3cb371'  # Darker green for 1 <= x < 2
        else:
            return val_color
    else:  # For negative 'a'
        if a == -1:
            return '#add8e6'  # Light blue for -1 < x < 0
        elif a == -2:
            return '#4682b4'  # Normal blue for -1 <= x < -2
        else:
            return '#000080'  # Dark blue for x <= -2

# Iterate over b and corresponding a values
for b in b_values:
    a_range = a_ranges.get(np.round(b, 1), np.array([5]))  # Default to a=5 if b not in a_ranges

    for a in a_range:
        y_custom_values = np.array([custom_growth(x, a, b) for x in x_values])
        highlight_y_custom = np.array([custom_growth(x, a, b) for x in highlight_x])
        
        # Color based on x values and the sign of a
        color = get_color(a)  # Get color for the current value of b and a

        # Plot using a consistent thick line and color based on the defined function
        line, = plt.plot(x_values, y_custom_values, label=rf'$f(x)$ with $a$={"+ " + str(a) if a > 0 else a}, $b$={round(b, 2)}', 
                         linestyle='-', linewidth=line_thickness, color=color, alpha=alpha_value)  # Set alpha for transparency
        
        # Highlight the specific point at 0.1 with light green
        # plt.scatter(0.1, custom_growth(0.1, a, b), color='#90ee90', s=80, alpha=alpha_value, zorder=5)

        # Use scatter for highlighting other points
        plt.scatter(highlight_x, highlight_y_custom, color=color, s=20, alpha=alpha_value)

        # Create a legend entry with a fixed size
        legend_handles.append(line)

# Plotting exponential growth for reference
y_exponential_values = exponential_growth(x_values)
highlight_y_exponential = exponential_growth(highlight_x)

# Create a handle for the exponential growth line in the legend
exp_growth_line, = plt.plot(x_values, y_exponential_values, label=f'$g(x)$ (MSE)', color='black', linestyle='-', linewidth=3)
plt.scatter(highlight_x, highlight_y_exponential, color='black', s=80, zorder=5)

plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(True, linestyle='--', alpha=0.7)

# Update x and y axis labels
plt.xlabel('1 - Normalized Distance Difference', fontsize=30)  # Clearer x-axis label
plt.ylabel('Similarity Score', fontsize=30) 

# Adjusted title position with new information
# plt.title('Custom Similarity Transformation with best (a,b) values vs. MSE ', fontsize=22, loc='center', pad=20)

# Increase the tick label sizes
plt.tick_params(axis='both', which='major', labelsize=30)  # Increased tick label sizes

# Create a custom legend with consistent size
# legend_labels = []
legend_labels = [r'$FT_{(1.4,0.5)}$']


plt.legend(legend_handles + [exp_growth_line], legend_labels + [f'$FE_{{L2}}$'], fontsize=33, loc='upper left', bbox_to_anchor=(0, 1), frameon=True, handlelength=2, borderaxespad=1)

# Ensure layout fits the legend inside the plot
plt.tight_layout(rect=[0, 0, 0.85, 1])

# Save the plot
save_path = fr'3_3_figures\loss_function\loss_functions_FT_MSE.png'
plt.savefig(save_path, dpi=350, bbox_inches="tight")

plt.show()
