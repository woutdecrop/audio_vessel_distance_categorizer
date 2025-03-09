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


from sklearn.metrics import mean_squared_error
from datetime import datetime
# from sliceguard.embeddings import generate_image_embeddings


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return dataloader.default_collate(batch)


class Dataset(Dataset):
    def __init__(self, df, audiopath, sr, sampleDur, channel=0):
        super(Dataset, self)
        self.audiopath, self.df, self.sr, self.sampleDur, self.channel = audiopath, df, sr, sampleDur, channel
        self.file_list = os.listdir(audiopath)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sig = self.read_snippet(row)
        if len(sig) < self.sampleDur * self.sr:
            sig = np.concatenate([sig, np.zeros(int(self.sampleDur * self.fs) - len(sig))])

        return Tensor(norm(sig)).float(), row.name

    def _get_duration(self, row):
        return self.sampleDur

    def read_snippet(self, row):
        info = sf.info(self.audiopath + '/' + row.filename)
        dur, fs = info.duration, info.samplerate
        sample_dur = self._get_duration(row)
        start = int(np.clip(row.pos - sample_dur / 2, 0, max(0, dur - sample_dur)) * fs)
        if row.two_files:
            stop = info.frames
            extra_dur = sample_dur - (info.frames - start) / fs
        else:
            stop = start + int(sample_dur * fs)
        try:
            sig, fs = sf.read(self.audiopath + '/' + row.filename, start=start, stop=stop, always_2d=True)
            if row.two_files:
                second_file = self.file_list[self.file_list.index(row.filename) + 1]
                stop2 = int(extra_dur * fs)
                sig2, fs2 = sf.read(self.audiopath + '/' + second_file, start=0, stop=stop2, always_2d=True)
                sig = np.concatenate([sig, sig2])
            sig = sig[:, self.channel]
        except Exception as e:
            print(f'Failed to load sound from row {row.name} with filename {row.filename}', e)

        if fs != self.sr:
            sig = resample(sig, int(len(sig)/fs*self.sr))
        return sig


class DatasetCropsDuration(Dataset):
    def __init__(self, df, audiopath, sr, sampleDur, winsize, win_overlap, n_mel, channel=0):
        super(Dataset, self)
        self.audiopath, self.df, self.sr, self.channel = audiopath, df, sr, channel
        self.winsize = winsize
        self.win_overlap = win_overlap
        self.n_mel = n_mel
        # self.norm = nn.InstanceNorm2d(1)
        self.file_list = os.listdir(audiopath)
        self.sampleDur = sampleDur

    def _get_duration(self, row):
        return row.duration + 0.2

    def get_spectrogram(self, sig):
        hopsize = int((len(sig) - self.winsize) / 128)
        f, t, sxx = scipy.signal.spectrogram(sig, fs=self.sr, window=('hamming'),
                                             nperseg=self.winsize,
                                             noverlap=self.winsize - hopsize, nfft=self.winsize,
                                             detrend=False,
                                             return_onesided=True, scaling='density', axis=-1,
                                             mode='magnitude')
        return f, t, sxx

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sig = self.read_snippet(row)
        f, t, sxx = self.get_spectrogram(sig)
        sxx = sxx[:, :self.n_mel]
        sxx = Tensor(sxx).float()
        return sxx.unsqueeze(0), row.name


class DatasetCrops(DatasetCropsDuration):
    def __init__(self, df, audiopath, sr, sampleDur, winsize, win_overlap, n_mel, channel=0):
        super(Dataset, self)
        self.audiopath, self.df, self.sr, self.channel = audiopath, df, sr, channel
        self.winsize = winsize
        self.win_overlap = win_overlap
        self.n_mel = n_mel
        # self.norm = nn.InstanceNorm2d(1)
        self.file_list = os.listdir(audiopath)
        self.sampleDur = sampleDur

    def get_spectrogram(self, sig, row):
        winsize = min(int(len(sig)/2), int(128 * row.max_freq / (row.max_freq - row.min_freq)) * 2)
        hopsize = min(int((len(sig) - self.winsize) / 128), int(winsize/2))
        f, t, sxx = scipy.signal.spectrogram(sig, fs=self.sr, window=('hamming'),
                                             nperseg=winsize,
                                             noverlap=winsize - hopsize, nfft=winsize,
                                             detrend=False,
                                             return_onesided=True, scaling='density', axis=-1,
                                             mode='magnitude')

        return f, t, sxx

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sig = self.read_snippet(row)
        f, t, sxx = self.get_spectrogram(sig, row)

        sxx = Tensor(sxx).float()
        max_freq = min(int(row.max_freq / (self.sr / 2) * sxx.shape[0]) + 1, sxx.shape[0] - 1)
        min_freq = max(0, int(row.min_freq / (self.sr / 2) * sxx.shape[0]) - 1)

        # min_dur = max(int(((self.sampleDur / 2) - (row.duration / 2 + 0.2)) / self.sampleDur * sxx.shape[1]) - 1, 0)
        # max_dur = min(int(((self.sampleDur / 2) + (row.duration / 2) - 0.2) / self.sampleDur * sxx.shape[1]) + 1, sxx.shape[1] - 1)
        sxx_cropped = sxx[min_freq: max_freq, :]  # min_dur:max_dur

        # sxx_mel = sxx_cropped - torch.quantile(sxx_cropped, 0.2, dim=-1, keepdim=True)[0]

        # plt.imshow(sxx_mel, origin='lower')
        # plt.axis('off')
        # plt.savefig(
        #     '/mnt/fscompute_shared/roi/datasets/bpns/stratified_test_set/crops_ae/%s.png' % row.name)
        # plt.close()

        # plt.pcolormesh(t[min_dur:max_dur], f[min_freq:max_freq], sxx_out.numpy()[0], cmap='jet', shading='nearest')
        # plt.savefig('/mnt/fscompute_shared/roi/datasets/bpns/stratified_test_set/predictions/crops_ae/%s.png' % row.name)

        sxx_out = F.resize(sxx_cropped.unsqueeze(0), (128, 128))

        return sxx_out, row.name


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

        return waveform, torch.tensor(self.label_to_id[row['label']])


def norm(arr):
    return (arr - np.mean(arr) ) / np.std(arr)


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)


class Reshape(nn.Module):
    def __init__(self, *shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.shape[0], *self.shape)


class Croper2D(nn.Module):
    def __init__(self, *shape):
        super(Croper2D, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x[:,:,:self.shape[0],(x.shape[-1] - self.shape[1])//2:-(x.shape[-1] - self.shape[1])//2]


class Accuracy:
    def __init__(self):
        self.num_total = 0
        self.num_correct = 0

    def update(self, logits, y):
        self.num_total += logits.shape[0]
        self.num_correct += torch.sum(logits.argmax(axis=1) == y).cpu().item()

    def get_metric(self):
        return {'acc': 0. if self.num_total == 0 else self.num_correct / self.num_total}

    def get_primary_metric(self):
        return self.get_metric()['acc']
        
def max_finder(logits, ids):
    predicted_values = []
    values=[]
    for tensor in logits:
        max_index = torch.argmax(tensor).item()  # Get the index of the maximum value in the tensor
        for key, val in ids.items():
            if val == max_index:
                predicted_values.append(key)
                values.append(val)
                break
    return predicted_values,values

# Function to convert float to string with '-' instead of '.'
def float_to_string(value):
    return re.sub(r'\.', '-', str(value))


def map_category_to_number(category):
    if category == "10+":
        return 10
    else:
        lower, upper = map(int, category.split('-'))
        return (lower + upper) / 2
# import pandas as pd

def eval_pytorch_model(model, result_dir, dataloader, metric_factory, device, similarity_matrix, similarity_matrix_distance, similarity_matrix_speed, similarity_matrix_activity, similarity_matrix_type, desc, weights, comment="", ids=None):
    model.eval()
    total_loss = 0.
    steps = 0
    metrics = []
    metrics_distance = []
    metrics_speed = []
    metrics_activity = []
    metrics_type = []
    true_values_list = []
    predicted_list = []
    y_list = []
    values_list = []

    with torch.no_grad():
        for x, y in tqdm(dataloader, desc=desc):
            x = x.to(device)
            y = y.to(device)

            loss, logits = model(x, y)
            total_loss += loss.cpu().item()
            steps += 1
            metrics = metrics_calculator(similarity_matrix, logits, metrics, y)
            metrics_distance = metrics_calculator(similarity_matrix_distance, logits, metrics_distance, y)
            metrics_speed = metrics_calculator(similarity_matrix_speed, logits, metrics_speed, y)
            metrics_activity = metrics_calculator(similarity_matrix_activity, logits, metrics_activity, y)
            metrics_type = metrics_calculator(similarity_matrix_type, logits, metrics_type, y)

            if ids is not None:
                max_positions = torch.argmax(logits, dim=1)
                predicted, values = max_finder(logits, ids)
                true_values = [list(ids.keys())[list(ids.values()).index(idx)] for idx in y.tolist()]

                true_values_list.extend(true_values)
                predicted_list.extend(predicted)
                y_list.extend(y.tolist())
                values_list.extend(values)

    total_loss /= steps
    mse = float('inf')

    # Create a DataFrame with the predicted, true, y, and values columns
    results_df = pd.DataFrame({
        'predicted': predicted_list,
        'true': true_values_list,
        'y': y_list,
        'values': values_list
    })

    if desc == "test":
        actual_numeric = y_list
        predicted_numeric = values_list

        mse = mean_squared_error(actual_numeric, predicted_numeric)
        print(f"Mean Squared Error: {mse}")


    print(f"Metrics: {np.mean(metrics)} | Distance: {np.mean(metrics_distance)} | Speed: {np.mean(metrics_speed)} | Activity: {np.mean(metrics_activity)} | Type: {np.mean(metrics_type)}")

    return total_loss, np.mean(metrics), np.mean(metrics_distance), np.mean(metrics_speed), np.mean(metrics_activity), np.mean(metrics_type), mse, results_df

    
def eval_pytorch_model_dclde(model, dataloader, device,desc,weights,ids,similarity_matrix, comment=""):
    model.eval()
    total_loss = 0.
    steps = 0
    true_values_list=[]
    predicted_list=[]
    metrics=[]
    with torch.no_grad():
        for x, y in tqdm(dataloader, desc=desc):
            x = x.to(device)
            y = y.to(device)

            loss, logits = model(x, y)
            total_loss += loss.cpu().item()
            steps += 1
            
            metrics=metrics_calculator(similarity_matrix,logits,metrics,y)
            
            # metric.update(logits.to("cpu"), y.to("cpu"))
            max_positions=torch.argmax(logits, dim=1)
            predicted=max_finder(logits,ids)
            true_values = [list(ids.keys())[list(ids.values()).index(idx)] for idx in y.tolist()]

            predicted_list.extend(predicted)
            true_values_list.extend(true_values)
    # Calculate accuracy
    # train_accuracy = train_metric.get_primary_metric()

    total_loss /= steps
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    timestamp_folder = os.path.join('/srv/CLAP/roi/BioLingual/output', timestamp)
    os.makedirs(timestamp_folder, exist_ok=True)
    
    # Construct filename with timestamp and weights
    weights_str = '_'.join([f"{float_to_string(value)}" for _, value in weights.items()])
    csv_filename = f'predicted_true_values_{timestamp}_{weights_str}_{desc}_{comment}.csv'
    csv_file_path = os.path.join(timestamp_folder, csv_filename)
            
    # Write data to CSV file
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['predicted', 'true'])
        for predicted, true in zip(predicted_list, true_values_list):
            writer.writerow([predicted, true])
    
    print("CSV file saved successfully.")

    return total_loss,np.mean(metrics)



import os
# wavs_folder=r'..\..UC6\data\data_per_station_10_vessel_information_point'
# d_train_path= r"..\..UC6\ds_split-CLAP\train.txt"
# d_valid_path= r"..\..UC6\ds_split-CLAP\val.txt"

# split = np.genfromtxt(d_train_path, dtype='str', delimiter=' ')
# d_train_loc = np.array([os.path.join(wavs_folder, i) for i in split[:, 0]])

# split = np.genfromtxt(d_valid_path, dtype='str', delimiter=' ')
# d_valid_loc = np.array([os.path.join(wavs_folder, i) for i in split[:, 0]])


# def categorize_speed(speed):
#     if speed >= 0 and speed < 1.5:
#         return '0-1.5'
#     elif speed >= 1.5 and speed < 8.2:
#         return '1.5-8.2'
#     elif speed >= 8.2 and speed < 13.3:
#         return '8.2-13.3'
#     else:
#         return '13.3+'

# def categorize_speed(speed):
#     if speed >= 0 and speed < 5:
#         return '0-5'
#     elif speed >= 5 and speed < 10:
#         return '5-10'
#     elif speed >= 10 and speed < 15:
#         return '10-15'
#     else:
#         return '15+'


def categorize_speed(speed):
    if speed >= 0 and speed < 8:
        return '0-8'
    elif speed >= 8 and speed < 14:
        return '5-14'
    else:
        return '14+'

# def categorize_speed(speed):
#     if speed >= 0 and speed < 2:
#         return '0-2'
#     elif speed >= 2 and speed < 4:
#         return '2-4'
#     elif speed >= 4 and speed < 6:
#         return '4-6'
#     elif speed >= 6 and speed < 8:
#         return '6-8'
#     elif speed >= 8 and speed < 9:
#         return '8-9'
#     elif speed >= 9 and speed < 10:
#         return '9-10'
#     elif speed >= 10 and speed < 11:
#         return '10-11'
#     elif speed >= 11 and speed < 12:
#         return '11-12'
#     elif speed >= 12 and speed < 13:
#         return '12-13'
#     elif speed >= 13 and speed < 14:
#         return '13-14'
#     elif speed >= 14 and speed < 15:
#         return '14-15'
#     elif speed >= 15 and speed < 17:
#         return '15-17'
#     else:
#         return '17+'

# def categorize_speed(speed):
#     speed = round(speed / 3) * 3  # Round speed to the nearest multiple of 3 km/h
#     if speed < 17:
#         return f'{speed}-{speed+2}'
#     else:
#         return '17+'
        

# def categorize_speed(speed):
#     if speed >= 0 and speed < 2:
#         return '0-2'
#     elif speed >= 2 and speed < 4:
#         return '2-4'
#     elif speed >= 4 and speed < 6:
#         return '4-6'
#     elif speed >= 6 and speed < 8:
#         return '6-8'
#     elif speed >= 8 and speed < 9:
#         return '8-9'
#     elif speed >= 9 and speed < 10:
#         return '9-10'
#     elif speed >= 10 and speed < 11:
#         return '10-11'
#     elif speed >= 11 and speed < 12:
#         return '11-12'
#     elif speed >= 12 and speed < 13:
#         return '12-13'
#     elif speed >= 13 and speed < 14:
#         return '13-14'
#     elif speed >= 14 and speed < 15:
#         return '14-15'
#     elif speed >= 15 and speed < 17:
#         return '15-17'
#     else:
#         return '17+'
        
# Apply function to create a new column 'speed_category'

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
    # Extract activity from the filename
    df['activity'] = df['filename'].apply(lambda x: x.split('_')[-3])

    # Extract ship type from the filename
    df['ship_type'] = df['filename'].apply(lambda x: x.split('_')[-4])

    # Apply the function to create a new column 'distance_category'
    df['distance_category'] = df['distance'].apply(categorize_distance)

    # Apply the function to create a new column 'speed_category'
    df['speed_category'] = df['speed'].apply(categorize_speed)

    # Create a combined_info column
    # df['label'] = df['ship_type'] + ' at distance ' + df['distance_category'] + ' with speed ' + df['speed_category'] + ' is ' + df['activity']
    df['label'] = ["ship"] * len(df['ship_type'])
    # df['activity'] = ["activity"] * len(df['activity'])
    df['speed_category'] = ["speed_category"] * len(df['speed_category'])
    df['activity'] = ["activity"] * len(df['activity'])
    df['label'] = df['label'] + ' at distance ' + df['distance_category'] + ' with speed ' + df['speed_category'] + ' is ' + df['activity']

    # df['label']= df['distance'].apply(categorize_distance)
    return df

# float(last_part.replace("-", ".")[:-2])

def process_filenames_dclde(d_train):
    # Create DataFrame with filenames
    df = pd.DataFrame({'filename': d_train})
    
    # Extract distance from the filename
    df['distance'] = df['filename'].apply(lambda x: float(x.split('km')[0].split('_')[-1].replace("-", "."))*1000)


    # Apply the function to create a new column 'distance_category'
    # df['distance_category'] = df['distance'].apply(categorize_distance)


    df['label']= df['distance'].apply(categorize_distance)


    return df


def extract_speed(speed_str):
    if '-' in speed_str:
        lower, upper = map(int, speed_str.split('-'))
        return (lower + upper) / 2
    else:
        return 17 #if speed_str == '15+' else int(speed_str.split('-')[0])
        

# def extract_features(class_string):
#     parts = class_string.split(' ')
#     distance_str = parts[3]
#     speed_str = parts[-3]
#     distance_str_cleaned = distance_str.replace('+', '')

#     distance = int(distance_str_cleaned.split('-')[0])
#     speed = speed_str if speed_str == '0-5' else 5 if speed_str == '5-10' else 10 if speed_str == '10-15' else 15
#     activity = parts[-1]
#     vessel_type = parts[0]
#     return distance, speed, activity, vessel_type

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

# def custom_sigmoid(x, a, b):
#     """
#     Custom sigmoid function with parameters a and b
#     Returns a value between 0 and 1.
#     """

#     return 1 / (1 + math.exp(-a * (x - b)))

# def sim_calculator(x):
# # for x in input_values:
#     a = 15  # Adjust this parameter to control the steepness of the curve
#     b = 0.7
#     if x < 0.7:
#         return 0
#     elif x==1:
#         return 1
#     else:
#         return custom_sigmoid(x, a, b)

def custom_growth(x,a=5,b=0.625):
    if x < b:
        return (0.2 / b) * x
    elif x <= 1:
        # a = 5
        return 0.2 + (1 - 0.2) * (1 - np.exp(-a * (x - b))) / (1 - np.exp(-a * (1 - b)))
    else:
        return 1


# def custom_growth(x):
#     return x ** 2
        
def similarity_distance(label_to_id,device):

    classes = label_to_id 


    # Create a matrix to hold the similarity values
    num_classes = len(classes)
    similarity_matrix = np.zeros((num_classes, num_classes))
    
    # Define weights for each attribute
    # distance_weight = 0.55
    # speed_weight = 0.30
    # activity_weight = 0.10
    # vessel_type_weight=0
    
    
    # Calculate similarity between each pair of classes
    for i, class_i in enumerate(classes):
        distance_i  = float(class_i.split('km')[0].replace("-", ".").replace("+", ""))

        # (x.split('km')[0].split('_')[-1].replace("-", "."))
        for j, class_j in enumerate(classes):
            distance_j = float(class_j.split('km')[0].replace("-", ".").replace("+", ""))
            # print(distance_i, distance_j)
            distance_similarity = 1 - abs(distance_i - distance_j) / 10
            distance_similarity = sim_calculator(distance_similarity)

            similarity_matrix[i, j] = distance_similarity
            
    return torch.tensor(similarity_matrix).to(device)


# def mse_loss(y_true, y_pred):
#     return (abs(y_true - y_pred)/10) ** 2


def similarity(label_to_id,device,a=10,b=0.625,distance_weight = 0, speed_weight = 0,activity_weight = 0,vessel_type_weight=0):

    classes = label_to_id 


    # Create a matrix to hold the similarity values
    num_classes = len(classes)
    similarity_matrix = np.zeros((num_classes, num_classes))
    
    # Define weights for each attribute
    # distance_weight = 0.55
    # speed_weight = 0.30
    # activity_weight = 0.10
    # vessel_type_weight=0
    
    
    # Calculate similarity between each pair of classes
    for i, class_i in enumerate(classes):
        distance_i, speed_i, activity_i, vessel_type_i = extract_features(class_i)
        for j, class_j in enumerate(classes):
            distance_j, speed_j, activity_j, vessel_type_j = extract_features(class_j)
            distance_similarity = 1 - abs(distance_i - distance_j) / 10
            # distance_similarity = sim_calculator(distance_similarity)
            distance_similarity = custom_growth(distance_similarity,a,b)
            
            # distance_similarity = distance_similarity ** 2 

            # speed_similarity = 1 - abs(speed_i - speed_j) / 16
            # speed_similarity = sim_calculator(speed_similarity)
            # distance_similarity=mse_loss(distance_i, distance_j)
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


def metrics_calculator(similarity_matrix,logits,metrics,y):
    values_tensor=similarity_matrix[y]
    max_positions=torch.argmax(logits, dim=1)
    predics=values_tensor[torch.arange(values_tensor.size(0)), max_positions]
    metrics.extend(predics.tolist())
    return metrics