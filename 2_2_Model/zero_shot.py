import os
import re
import pandas as pd
import torch
import torchaudio
from pathlib import Path
from transformers import pipeline
from tqdm import tqdm

# def classify_distance_label(distance_km):
#     if distance_km < 3:
#         return 'boat closeby'
#     elif distance_km < 7:
#         return 'boat medium distance'
#     else:
#         return 'boat far away'

def classify_distance_label(distance_km):
    if distance_km < 3:
        return 'vessel noise from nearby vessel'
    elif distance_km < 7:
        return 'vessel noise from distant vessel'
    else:
        return 'distance vessel noise from very distant vessel'



def extract_distance_from_filename(filename):
    distance = filename.split("_")[-1][:-4]
    return int(distance)/1000

root_folder="/storage/CLAP_paper/data/data_per_station_6_paper-window-6_10seconds-efficient_paper_split"
output_name = 'zero_shot_space'
features_path = Path(root_folder).joinpath(output_name + '.pkl')

if features_path.exists():
    pd.read_pickle(features_path)

audio_classifier = pipeline(task="zero-shot-audio-classification", model="davidrrobinson/BioLingual")
features_list = []
idxs = []
metadata = []

# Collect all wav file paths first
all_files = []
for subfolder in os.listdir(root_folder):
    full_subfolder_path = os.path.join(root_folder, subfolder)
    if not os.path.isdir(full_subfolder_path):
        continue

    for filename in os.listdir(full_subfolder_path):
        if filename.endswith(".wav"):
            all_files.append((filename, os.path.join(full_subfolder_path, filename)))
    break


import soundfile as sf
import torch
from tqdm import tqdm
# Iterate with tqdm
for filename, filepath in tqdm(all_files, desc="Classifying audio files"):
    distance_km = extract_distance_from_filename(filename)
    if distance_km is None:
        print(f"Skipping {filename}: could not parse distance.")
        continue

    label = classify_distance_label(distance_km)
    # data, sr = sf.read(filepath)
#     waveform = torch.tensor(data.T) if data.ndim > 1 else torch.tensor(data).unsqueeze(0)
#     waveform = waveform.mean(dim=0).unsqueeze(0)  # mono
# import soundfile as sf
# import torch

    data, sr = sf.read(filepath)

    if data.size == 0:
        print(f"Skipping {filename}: empty audio data.")
        continue

    
    # Convert to [channels, time] if multi-channel, or [1, time] if mono
    if data.ndim == 1:
        waveform = torch.tensor(data).unsqueeze(0)  # [1, time]
    else:
        waveform = torch.tensor(data.T)  # [channels, time]
    
    # Ensure mono by averaging over channels
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)  # [1, time]


    output = audio_classifier(waveform.squeeze(0).numpy(), candidate_labels=[
        'vessel noise from nearby vessel',
        'vessel noise from distant vessel',
        'distance vessel noise from very distant vessel'
    ])



    predicted_label = max(output, key=lambda x: x['score'])['label']
    print(predicted_label,label)
    # except Exception as e:
    #     print(f"Error processing {filename}: {e}")
    #     continue

    features_list.append(predicted_label)
    idxs.append(filename)
    metadata.append({'distance': distance_km, 'true_label': label, 'file': filename})
    # break
df = pd.DataFrame(features_list, index=idxs, columns=['predicted_label'])
meta_df = pd.DataFrame(metadata).set_index('file')
total_df = pd.merge(df, meta_df, left_index=True, right_index=True)

# total_df.to_pickle(features_path)
# return total_df

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# # Merge predictions with metadata
# df = pd.DataFrame(features_list, index=idxs, columns=['predicted_label'])
# meta_df = pd.DataFrame(metadata).set_index('file')
# total_df = pd.merge(df, meta_df, left_index=True, right_index=True)

# # Save results
# total_df.to_pickle(features_path)

# --- Validation ---

y_true = total_df['true_label']
y_pred = total_df['predicted_label']

# Define the correct order of distance labels
label_order = [
        'vessel noise from nearby vessel',
        'vessel noise from distant vessel',
        'distance vessel noise from very distant vessel'
    ]

# Basic accuracy
acc = accuracy_score(y_true, y_pred)
print(f"Accuracy: {acc:.3f}")

# Classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, labels=label_order))


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


label_order = [
        'vessel noise from nearby vessel',
        'vessel noise from distant vessel',
        'distance vessel noise from very distant vessel'
    ]


cm = confusion_matrix(y_true, y_pred, labels=label_order)

plt.figure(figsize=(10, 8))
ax = sns.heatmap(cm,
                 annot=True,
                 fmt='d',
                 xticklabels=label_order,
                 yticklabels=label_order,
                 cmap='Blues',
                 cbar=True,
                 annot_kws={"size": 16},  # Bigger text
                 linewidths=0,            # No internal grid
                 linecolor='black')

# Add thin black border around the full plot
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(0.8)
    spine.set_color('black')

plt.xlabel('Predicted Label', fontsize=14)
plt.ylabel('True Label', fontsize=14)
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()

# 🔽 Save to PNG file
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()  # Close the figure to avoid displaying in notebooks/scripts
