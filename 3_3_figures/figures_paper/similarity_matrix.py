import numpy as np

import matplotlib.pyplot as plt

def custom_growth(x):
    if x < 0.625:
        # Linear growth from 0 to 0.625, reaching a value of 0.2 at x = 0.625
        return (0.2 / 0.625) * x
    elif x <= 1:
        # Exponential growth from 0.625 to 1, reaching a value of 1 at x = 1
        a = 5  # Adjust this parameter to control the steepness of the exponential growth
        return 0.2 + (1 - 0.2) * (1 - np.exp(-a * (x - 0.625))) / (1 - np.exp(-a * (1 - 0.625)))
    else:
        # Beyond x = 1, keep the function constant at 1
        return 1

def L2_loss(y_true, y_pred):
    return ((1-abs(y_true - y_pred)/10)) ** 2

list_cat= ['0-1 km', '1-2 km', '2-3 km', '3-4 km', '4-5 km', '5-6 km', '6-7 km', '7-8 km', '8-9 km', '9-10 km', '10+ km']

ids={lbl: i for i, lbl in enumerate(list_cat)}
classes = ids 

# Create a matrix to hold the similarity values
num_classes = len(classes)
similarity_matrix = np.zeros((num_classes, num_classes))
    
for i, class_i in enumerate(classes):
        distance_i  = float(class_i.split('-')[0].replace("-", ".").replace("+ km", ""))

        # (x.split('km')[0].split('_')[-1].replace("-", "."))
        for j, class_j in enumerate(classes):
            distance_j = float(class_j.split('-')[0].replace("-", ".").replace("+ km", ""))
            # print(distance_i, distance_j)
            distance_similarity = 1 - abs(distance_i - distance_j)/10 

            distance_similarity = custom_growth(distance_similarity)
            distance_similarity=L2_loss(distance_i, distance_j)
            similarity_matrix[i, j] = distance_similarity
from matplotlib.colors import LinearSegmentedColormap

blues = plt.get_cmap('Blues')
blues_lighter = LinearSegmentedColormap.from_list('Blues_lighter', blues(np.linspace(0, 0.8, 256)))

plt.figure(figsize=(10, 8))
plt.imshow(similarity_matrix, cmap=blues_lighter, interpolation='nearest')
plt.colorbar(label='Similarity')

plt.xticks(range(num_classes), list_cat, rotation=45, fontsize=16)
plt.yticks(range(num_classes), list_cat, fontsize=16)
# plt.title('Similarity Matrix Visualization for $f(x)$')

# # Set axis labels with matching font size
# plt.xlabel('Predicted distance', fontsize=18)
# plt.ylabel('True distance', fontsize=18)

# Add similarity values inside the cells
for i in range(num_classes):
    for j in range(num_classes):
        plt.text(j, i, f"{similarity_matrix[i, j]:.2f}",fontsize=12, ha='center', va='center', color='black')
save_path = r'3_3_figures\model\similarity_matrix_L2.png'
plt.savefig(save_path, dpi=350, bbox_inches="tight")
plt.show()
