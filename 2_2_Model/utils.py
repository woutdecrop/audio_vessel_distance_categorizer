import soundfile as sf
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader, dataloader
import torchaudio
import torch
import numpy as np
from scipy.signal import resample
import torchvision.transforms.functional as F
import torch.nn.functional as F_general
import scipy
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import csv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

import pathlib

from datetime import datetime

import os
import shutil

def delete_all_subfolders(parent_dir):
    """
    Deletes all subfolders in the specified parent directory.
    
    Args:
        parent_dir (str): Path to the parent directory.
    """
    # print(f"Deleting all subfolders in: {parent_dir}")
    for subfolder in os.listdir(parent_dir):
        subfolder_path = os.path.join(parent_dir, subfolder)
        if os.path.isdir(subfolder_path):  # Check if it's a directory
            # print(f"Deleting subfolder: {subfolder_path}")
            shutil.rmtree(subfolder_path)  # Delete the subfolder



def save_layer_weights(model, model_path):
    """
    Saves the current weights of all defined layers to the given path.
    """
    for layer_name, layer in model.named_children():
        layer_weights_path = os.path.join(model_path, f"{layer_name}.pth")
        try:
            torch.save(layer.state_dict(), layer_weights_path)
            print(f"Saved weights for layer '{layer_name}' to {layer_weights_path}")
        except Exception as e:
            print(f"Failed to save weights for layer '{layer_name}': {e}")


def convert_labels_to_km(labels):
    km_labels = []
    for label in labels:
        if label == max(labels):
            km_labels.append("10+ km")
        else:
            km_labels.append(f"{label}-{label+1} km")
    return km_labels



def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues, save_path=None):
    """
    This function prints and plots the confusion matrix.
    The colors are based on the percentage of each row.
    """
    # Calculate the percentage values for coloring based on the total sum of the confusion matrix
    total = np.sum(cm)
    cm_percentage = cm.astype('float') / total  # Calculate percentage of the entire matrix
    cm_percentage = np.nan_to_num(cm_percentage)  

    plt.figure(figsize=(8, 6))
    plt.imshow(cm_percentage, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=22)  # Increased title font size
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=16)  # Increased tick label font size
    plt.yticks(tick_marks, classes, fontsize=16)

    fmt = 'd'  # Display raw counts
    thresh = cm_percentage.max() / 2.  # Set threshold for text color
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     ha="center", va="center",
                     fontsize=14,
                     color="white" if cm_percentage[i, j] > thresh else "black")  # Adjust text color based on percentage

            # Display the percentage in the cell as a higher value (of the row)
            percentage_text = f"{cm_percentage[i, j] * 100:.1f}%"  # Format percentage
            # plt.text(j, i, percentage_text, ha="center", va="bottom", fontsize=10, color="black")  # Display percentage

    plt.tight_layout()
    plt.xlabel('Predicted label', fontsize=18)  # Increased x-axis label font size
    plt.ylabel('True label', fontsize=18)  # Increased y-axis label font size

    if save_path:
        plt.savefig(save_path, bbox_inches= 'tight' )
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()

def _sort_description(desc):
    """Helper function to sort descriptions based on distance."""
    if '10+ km' in desc:
        return 10
    start, end = map(int, re.findall(r'\d+', desc))
    return (start + end) / 2
    
def float_to_string(value):
    return re.sub(r'\.', '-', str(value))


class DatasetEmbeddings(DataLoader):
    def __init__(self, embeddings_folder,device, desc="None"):
        self.folder_path = embeddings_folder
        self.desc=desc
        self.device=device
    def __len__(self):
        embedding_type_folder=os.path.join(self.folder_path, self.desc)
        return len(list(pathlib.Path(embedding_type_folder).glob('*.pt')))


    def __getitem__(self, idx):
        embedding_path=os.path.join(self.folder_path, self.desc, f'embedding_{idx}.pt')
        row = torch.load(embedding_path, map_location=self.device)
        
        # row = pickle.load(os.path.join(self.folder_path, self.desc, f'embedding_{idx}.pt'))
        return row




class DatasetLoadEmbeddings(DataLoader):
    def __init__(self, df, ids, device):
        self.df = df.copy()
        self.device = device
        self.label_to_id = ids

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        embedding_path = row['embedding']  # Directly access the path from the DataFrame

        # Load the embedding
        embedding = torch.load(embedding_path, map_location=self.device)
        
        # Get the label ID
        label_tensor = torch.tensor(self.label_to_id[row['label']], dtype=torch.long)
        
        return embedding, label_tensor

        
class DatasetWaveform(DataLoader):
    def __init__(self, df, wavs_folder, desired_fs, max_duration,ids, channel=0):
        self.file_list = os.listdir(wavs_folder)
        self.df = df.copy()
        self.wavs_folder = wavs_folder
        self.desired_fs = desired_fs
        self.channel = channel
        self.max_duration = max_duration
        self.label_to_id = ids

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        wav_path =  row['filename'] #list(self.wavs_folder.glob('**/' + row['filename']))[0]
        # wav_path = self.wavs_folder.joinpath(row['Begin File'])
        waveform_info = torchaudio.info(wav_path)

        # If the selection is in between two files, open both and concatenate them
        waveform, fs = torchaudio.load(wav_path) #,
                                        # num_frames=461472)
        # waveform, fs = torchaudio.load(wav_path,
        #                                 frame_offset=row['begin_sample'],
        #                                 num_frames=row['end_sample'] - row[
        #                                     'begin_sample'])
        if waveform_info.sample_rate != self.desired_fs:
            transform = torchaudio.transforms.Resample(fs, self.desired_fs)
            waveform = transform(waveform)
        else:
            waveform = waveform

        max_samples = self.max_duration * self.desired_fs
        waveform = waveform[self.channel, :max_samples]
        if waveform.shape[0] < max_samples:
            waveform = F_general.pad(waveform, (0, max_samples - waveform.shape[0]))
        # return wav_path torch.tensor(self.label_to_id[row['label']])
        return waveform, torch.tensor(self.label_to_id[row['label']]),wav_path


        # return self.get_metric()['acc']
        
def max_finder(logits, ids):
    predicted_values = []
    for tensor in logits:
        max_index = torch.argmax(tensor).item()  # Get the index of the maximum value in the tensor
        for key, val in ids.items():
            if val == max_index:
                predicted_values.append(key)
                break
    return predicted_values




import os


def categorize_speed(speed):
    if speed >= 0 and speed < 8:
        return '0-8'
    elif speed >= 8 and speed < 14:
        return '5-14'
    else:
        return '14+'



def categorize_distance(distance):
    if distance >= 0 and distance < 1000:
        return '0-1 km'
    elif distance >= 1000 and distance < 2000:
        return '1-2 km'
    elif distance >= 2000 and distance < 3000:
        return '2-3 km'
    elif distance >= 3000 and distance < 4000:
        return '3-4 km'
    elif distance >= 4000 and distance < 5000:
        return '4-5 km'
    elif distance >= 5000 and distance < 6000:
        return '5-6 km'
    elif distance >= 6000 and distance < 7000:
        return '6-7 km'
    elif distance >= 7000 and distance < 8000:
        return '7-8 km'
    elif distance >= 8000 and distance < 9000:
        return '8-9 km'
    elif distance >= 9000 and distance <= 10000:
        return '9-10 km'
    else:
        return '10+ km'


def process_filenames(d_train):
    # Create DataFrame with filenames
    df = pd.DataFrame({'filename': d_train})
    
    # Extract distance from the filename
    df['distance'] = df['filename'].apply(lambda x: float(x.split('_')[-1].split('.wav')[0]))

    # Extract speed from the filename
    df['speed'] = df['filename'].apply(lambda x: float(x.split('_')[-2].replace('-', '.')))
    df = df[df['speed'] <= 30]

    # Extract ship type from the filename
    df['ship_type'] = df['filename'].apply(lambda x: x.split('_')[-4])

    # Apply the function to create a new column 'distance_category'
    df['distance_category'] = df['distance'].apply(categorize_distance)



    # Create a combined_info column
    # df['label'] = df['ship_type'] + ' at distance ' + df['distance_category'] + ' with speed ' + df['speed_category'] + ' is ' + df['activity']
    df['label'] = ["ship"] * len(df)
    # df['activity'] = ["activity"] * len(df['activity'])
    df['speed_category'] = ["speed_category"] * len(df)
    df['activity'] = ["activity"] * len(df)
    df['label'] = df['label'] + ' at distance ' + df['distance_category'] + ' with speed ' + df['speed_category'] + ' is ' + df['activity']

    # Apply the function to create a new column 'speed_category'
    df['speed_category'] = df['speed'].apply(categorize_speed)
    # df['label']= df['distance'].apply(categorize_distance)

    # Extract activity from the filename
    df['activity'] = df['filename'].apply(lambda x: x.split('_')[-3])
    return df




def extract_speed(speed_str):
    if '-' in speed_str:
        lower, upper = map(int, speed_str.split('-'))
        return (lower + upper) / 2
    else:
        return 17 #if speed_str == '15+' else int(speed_str.split('-')[0])
        

def extract_features(class_string):
    parts = class_string.split(' ')
    distance_str = parts[3]
    speed_str = parts[-3]
    distance_str_cleaned = distance_str.replace('+', '')

    distance = int(distance_str_cleaned.split('-')[0])
    speed = extract_speed(speed_str)
    activity = parts[-1]
    vessel_type = parts[0]
    return distance, speed, activity, vessel_type
    

import math



def custom_growth(x,param_a,param_b):
    if x < 0.625:
        # Linear growth from 0 to 0.625, reaching a value of 0.2 at x = 0.625
        return (0.2 / param_b) * x
    elif x <= 1:
        # Exponential growth from 0.625 to 1, reaching a value of 1 at x = 1
        # a = 5  # Adjust this parameter to control the steepness of the exponential growth
        return 0.2 + (1 - 0.2) * (1 - np.exp(-param_a * (x - param_b))) / (1 - np.exp(-param_a * (1 - param_b)))
    else:
        # Beyond x = 1, keep the function constant at 1
        return 1

def L2_loss(y_true, y_pred):
    return (abs(y_true - y_pred)/10) ** 2

def similarity(label_to_id,device,param_a,param_b, L2=False, distance_weight = 0, speed_weight = 0,activity_weight = 0,vessel_type_weight=0):

    classes = label_to_id 


    # Create a matrix to hold the similarity values
    num_classes = len(classes)
    similarity_matrix = np.zeros((num_classes, num_classes))

    # Calculate similarity between each pair of classes
    for i, class_i in enumerate(classes):
        distance_i, speed_i, activity_i, vessel_type_i = extract_features(class_i)
        for j, class_j in enumerate(classes):
            distance_j, speed_j, activity_j, vessel_type_j = extract_features(class_j)
            distance_similarity = 1 - abs(distance_i - distance_j) / 10
            # distance_similarity = sim_calculator(distance_similarity)
            if L2:
                distance_similarity=1-L2_loss(distance_i, distance_j)
            else:
                distance_similarity = custom_growth(distance_similarity,float(param_b),float(param_b))


            speed_similarity = 1 if speed_i == speed_j else 0
            activity_similarity = 1 if activity_i == activity_j else 0
            vessel_type_similarity = 1 if vessel_type_i == vessel_type_j else 0
            # Similarity is a combination of all attributes
            similarity = (distance_similarity * distance_weight +
                          speed_similarity * speed_weight +
                          activity_similarity * activity_weight +
                          vessel_type_similarity * vessel_type_weight)
            similarity_matrix[i, j] = similarity
    
    return torch.tensor(similarity_matrix).to(device)

# def metrics_calculator(similarity_matrix,logits,metrics,y):
#     values_tensor=similarity_matrix[y]
#     max_positions = torch.argmax(values_tensor, dim=1)  # This should be shape [16]

#     predics = values_tensor[torch.arange(values_tensor.size(0)), max_positions]
#     print("Corrected max_positions shape:", max_positions.shape)  # Should print torch.Size([16])
#     metrics.extend(predics.tolist())
#     # values_tensor=similarity_matrix[y]
#     # max_positions=torch.argmax(logits, dim=1)
#     # print("values_tensor shape:", values_tensor.shape)
#     # print("max_positions shape:", max_positions.shape)
#     # print("values_tensor size(0):", values_tensor.size(0))

#     # predics=values_tensor[torch.arange(values_tensor.size(0)), max_positions]
#     # metrics.extend(predics.tolist())
#     return metrics
    
def metrics_calculator(similarity_matrix,logits,metrics,y):
    
    values_tensor=similarity_matrix[y]
    max_positions=torch.argmax(logits, dim=1)


    predics=values_tensor[torch.arange(values_tensor.size(0)), max_positions]

    # Sum each row in the `values` tensor
    row_sums = values_tensor.sum(dim=1)  # Sum along columns (dim=1)
    
    # Divide each number in `pred` by the corresponding row sum
    percentage = predics / row_sums

    
    metrics.extend(percentage.tolist())
    return metrics