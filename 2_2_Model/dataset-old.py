import datetime
import json
import os
import pathlib
import shutil
import sys

# import fairseq
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz
import scipy
import soundfile as sf
import torch
import torchaudio
import torchaudio.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from PIL import Image
# from maad import util
from tqdm import tqdm
import torch.optim as optim
from sklearn.model_selection import train_test_split
import copy
import suntime
import pytz

from transformers import ClapModel, ClapProcessor
from transformers import pipeline

import models
import utils as u

torchaudio.set_audio_backend(backend='soundfile')

import torch.optim.lr_scheduler as lr_scheduler

# matplotlib.use('TkAgg')
# Get the color map by name:
cm = plt.get_cmap('jet')


class CLAP_Vessel_Distance:
    def __init__(self, config):
        # Spectrogram settings
        self.duration = config['duration']
        self.overlap = config['overlap']  # overlap of the chunks in %
        self.desired_fs = config['desired_fs']
        self.channel = config['channel']
        self.log = config['log']
        self.color = config['color']

        # Folders
        self.wavs_folder = pathlib.Path(config['wavs_folder'])
        self.dataset_folder = pathlib.Path(config['dataset_folder'])
        self.images_folder = self.dataset_folder.joinpath('images')
        self.labels_folder = self.dataset_folder.joinpath('labels')

        self.d_train_path=config['d_train_path']
        self.d_valid_path=config['d_valid_path']

        self.annotations_file = config['annotations_file']

        self.nfft = config['nfft']
        self.win_len = config['win_len']
        self.hop_length = int(self.win_len / config['hop_ratio'])
        self.win_overlap = self.win_len - self.hop_length

        self.normalization_style = config['normalization_style']

        if 'min_duration' in config.keys():
            self.MIN_DURATION = config['min_duration']
        else:
            self.MIN_DURATION = self.nfft / self.desired_fs

        if 'max_duration' in config.keys():
            self.MAX_DURATION = config['max_duration']
        else:
            self.MAX_DURATION = self.duration / 2
        self.MIN_SNR = 10

        self.blocksize = int(self.duration * self.desired_fs)

        self.config = config

    def __setitem__(self, key, value):
        if key in self.config.keys():
            self.config[key] = value
        self.__dict__[key] = value



    def train_clap(self, model_path='davidrrobinson/BioLingual',a=5,b=0.5, epochs=10, lr=1e-5,
                   batch_size=80, stop_shuffle=False, sample_dur=10):

        if torch.cuda.is_available():
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print("Selected CUDA device:", torch.cuda.get_device_name(device))
        else:
            print("CUDA is not available. Need GPU")
            return 
        device =torch.device('cuda:0')

        weights = {
            'distance_weight': 1,
            'speed_weight': 0,
            'activity_weight': 0,
            'vessel_type_weight': 0
        }

        comment=rf"spectrogram_PAPER_{a}_{b}"
        weights_str = '_'.join([f"{u.float_to_string(value)}" for _, value in weights.items()])+"_"+ comment
        log_path = f'roi/BioLingual/logs_{weights_str}.log'
        log_file = open(log_path, mode='w')

        # detections = self.convert_raven_to_ae_format(labels_to_exclude=None)
        # detections = detections.loc[~detections.label.isna()]

        d_train_path= "data/train.txt"
        d_valid_path= "data/val.txt"
        d_test_path= "data/test.txt"
        # print("path ", d_train_path)
        split = np.genfromtxt(d_train_path, dtype='str', delimiter=' ')
        d_train_loc = np.array([os.path.join(self.wavs_folder, i) for i in split[:, 0]])

        split = np.genfromtxt(d_valid_path, dtype='str', delimiter=' ')
        d_valid_loc = np.array([os.path.join(self.wavs_folder, i) for i in split[:, 0]])

        split = np.genfromtxt(d_test_path, dtype='str', delimiter=' ')
        d_test_loc = np.array([os.path.join(self.wavs_folder, i) for i in split[:, 0]])


        d_train=u.process_filenames(d_train_loc)#.iloc[0:100]
        d_valid=u.process_filenames(d_valid_loc)#.iloc[0:200]
        d_test=u.process_filenames(d_test_loc)#.iloc[0:200]
        
        train_labels = set(d_train["label"])
        # Count the occurrences of each label in d_train
        label_counts_train = d_train["label"].value_counts()
        
        # Filter out labels which have less than 5 samples in d_train
        valid_train_labels = label_counts_train[label_counts_train >= 5].index
        
        # Filter d_train and d_valid based on valid_train_labels
        d_train = d_train[d_train["label"].isin(valid_train_labels)]
        d_valid = d_valid[d_valid["label"].isin(valid_train_labels)]
        # Filter d_valid based on labels present in d_train
        # d_valid= d_valid[d_valid["label"].isin(train_labels)]
        test_train_labels = label_counts_train[label_counts_train >= 5].index

        # Filter d_train and d_test based on test_train_labels
        d_train = d_train[d_train["label"].isin(test_train_labels)]
        d_test = d_test[d_test["label"].isin(test_train_labels)]

        # Display the number of unique classes
        # Display the number of unique classes
        num_classes = len(valid_train_labels)
        descriptions=d_train['label'].unique()
        def extract_distance_value(description):
            if '10+ km' in description:
                return 10  # Assign a numeric value for "10+ km"
            # Extract distance range and convert to numeric value
            distance_part = description.split('distance ')[1].split(' km')[0]
            start, end = distance_part.split('-')
            return (int(end) + int(start)) / 2
        
        # Sort the array based on the numeric values
        sorted_descriptions = sorted(descriptions, key=extract_distance_value)
        
        # Convert sorted list back to numpy array
        sorted_array = np.array(sorted_descriptions)
        ids={lbl: i for i, lbl in enumerate(sorted_array)}
        

        similarity_matrix=u.similarity(ids,device,a,b,distance_weight = weights['distance_weight'], speed_weight = weights['speed_weight'],activity_weight = weights['activity_weight'],vessel_type_weight= weights['vessel_type_weight'])
        similarity_matrix_distance=u.similarity(ids,device,a,b,distance_weight = 1)
        similarity_matrix_speed=u.similarity(ids,device,a,b,speed_weight = 1)
        similarity_matrix_activity=u.similarity(ids,device,a,b,activity_weight = 1)
        similarity_matrix_type=u.similarity(ids,device,a,b,vessel_type_weight = 1)
        
        model = models.CLAPClassifier(model_path, num_classes, sr=self.desired_fs, device=device, similarity_matrix=similarity_matrix, multi_label=False)
        
        dataloader_train = torch.utils.data.DataLoader(
            dataset=u.DatasetWaveform(df=d_train, wavs_folder=self.wavs_folder, desired_fs=self.desired_fs,
                                      max_duration=sample_dur,ids=ids),
            batch_size=batch_size,
            shuffle=not stop_shuffle)

        dataloader_val = torch.utils.data.DataLoader(
            dataset=u.DatasetWaveform(df=d_valid, wavs_folder=self.wavs_folder, desired_fs=self.desired_fs,
                                      max_duration=sample_dur,ids=ids),
            batch_size=batch_size,
            shuffle=not stop_shuffle)

        dataloader_test = torch.utils.data.DataLoader(
            dataset=u.DatasetWaveform(df=d_test, wavs_folder=self.wavs_folder, desired_fs=self.desired_fs,
                                      max_duration=sample_dur,ids=ids),
            batch_size=batch_size,
            shuffle=stop_shuffle)
        
        valid_loss_best = 0
        valid_loss_previous=0
        
        best_model = model
        break_next=False
        log_file.write("lr = {}\n".format(lr))
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # print("lr = {}".format(lr), file=log_file)

        optimizer = optim.Adam(params=model.parameters(), lr=lr)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        
        for epoch in range(epochs):
            # print(f'epoch = {epoch}', file=sys.stderr)
            sys.stderr.write('epoch = {}\n'.format(epoch))
            model.train()

            train_loss = 0.
            train_steps = 0
            train_metric = u.Accuracy()
            metrics=[]
            metrics_distance=[]
            metrics_speed=[]
            metrics_activity=[]
            metrics_type=[]
            for x, y in tqdm(dataloader_train, desc='train'):
                optimizer.zero_grad()
                # print("optimizer")
                x = x.to(device)
                y = y.to(device)
                model = model.to(device)
                loss, logits = model(x, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.cpu()
                train_steps += 1
                metrics=u.metrics_calculator(similarity_matrix,logits,metrics,y)
                metrics_distance=u.metrics_calculator(similarity_matrix_distance,logits,metrics_distance,y)
                metrics_speed=u.metrics_calculator(similarity_matrix_speed,logits,metrics_speed,y)
                metrics_activity=u.metrics_calculator(similarity_matrix_activity,logits,metrics_activity,y)
                metrics_type=u.metrics_calculator(similarity_matrix_type,logits,metrics_type,y)
                # Calculate accuracy
                # train_accuracy = train_metric.get_primary_metric()
                
                # Print the accuracy
                print(f"Training metrics: {np.mean(metrics)}| Distance: {np.mean(metrics_distance)}| Speed: {np.mean(metrics_speed)} | Activity: {np.mean(metrics_activity)} | Type: {np.mean(metrics_type)}")
                # break
            valid_loss, valid_metric,valid_metric_distance,valid_metric_speed,valid_metric_activity,valid_metric_type = u.eval_pytorch_model(
                model=model,
                dataloader=dataloader_val,
                metric_factory=u.Accuracy,
                device=device,
                similarity_matrix=similarity_matrix,
                similarity_matrix_distance=similarity_matrix_distance,
                similarity_matrix_speed=similarity_matrix_speed,
                similarity_matrix_activity=similarity_matrix_activity,
                similarity_matrix_type=similarity_matrix_type, 
                desc='valid',
                weights=weights, comment=comment)

            scheduler.step()
            # Optionally print the current learning rate
            current_lr = scheduler.get_last_lr()[0]
            text=f"Learning rate after epoch {epoch}: {current_lr}"
            print(text)
            log_file.write(text + '\n')
            log_file.flush()
            
            if valid_loss > valid_loss_best:
                weights_folder = os.path.join('roi/BioLingual/model', weights_str)
                os.makedirs(weights_folder, exist_ok=True)
                valid_metric_best = valid_metric
                best_model = copy.deepcopy(model)
                model.clap.save_pretrained(weights_folder)
                model.processor.save_pretrained(weights_folder)
                torch.save(model.linear.state_dict(), os.path.join(weights_folder, 'linear.pth'))
            if min(valid_loss_previous, valid_loss) > valid_loss_best:
                print("breaking early")
                log_file.write("breaking next cycle" + '\n')
                log_file.flush()
                if break_next:
                    break
                break_next=True

            
            valid_loss_previous=valid_loss

            log_message = json.dumps({
                    'epoch': epoch,
                    'train': {
                        'loss': (train_loss / train_steps).cpu().item(),
                        'metric': np.mean(metrics),
                        'metric_distance': np.mean(metrics_distance),
                        'metric_speed': np.mean(metrics_speed),
                        'metric_activity': np.mean(metrics_activity),
                        'metric_type': np.mean(metrics_type),
                    },
                    'valid': {
                        'loss': valid_loss,
                        'metric': valid_metric,
                        'metric_distance': np.mean(valid_metric_distance),
                        'metric_speed': np.mean(valid_metric_speed),
                        'metric_activity': np.mean(valid_metric_activity),
                        'metric_type': np.mean(valid_metric_type),
                    },
                })
            log_file.write(log_message + '\n')
            log_file.flush()
        test_loss, test_metric,test_metric_distance,test_metric_speed,test_metric_activity,test_metric_type = u.eval_pytorch_model(
            model=best_model,
            dataloader=dataloader_test,
            metric_factory=u.Accuracy,
            device=device,
            similarity_matrix=similarity_matrix,
            similarity_matrix_distance=similarity_matrix_distance,
            similarity_matrix_speed=similarity_matrix_speed,
            similarity_matrix_activity=similarity_matrix_activity,
            similarity_matrix_type=similarity_matrix_type,
            desc='test',
            weights=weights,
            comment=comment,
            ids=ids)
        
        log_message = json.dumps({
            'test': {
                'loss': test_loss,
                'metric': test_metric,
                'metric_distance': np.mean(test_metric_distance),
                'metric_speed': np.mean(test_metric_speed),
                'metric_activity': np.mean(test_metric_activity),
                'metric_type': np.mean(test_metric_type),
            }
        })

        # Write the JSON string to the log file
        log_file.write(log_message + '\n')
        log_file.flush()


        return best_model, valid_metric_best

    
