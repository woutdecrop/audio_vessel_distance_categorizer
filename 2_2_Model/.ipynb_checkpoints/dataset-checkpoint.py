import datetime
import json
import os
import pathlib
import torch
import numpy as np
import copy
from tqdm import tqdm
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from transformers import ClapModel, ClapProcessor
import sys
import models
import utils as u
import pandas as pd
import re

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, mean_squared_error
class CLAP_Vessel_Distance:
    def __init__(self, config, weights=None, comment=""):
        # Set default weights if not provided
        self.weights = weights if weights is not None else {
            'distance_weight': 1,
            'speed_weight': 0,
            'activity_weight': 0,
            'vessel_type_weight': 0
        }

        self.comment=comment 
        # Set weights_str based on the provided or default weights and comment
        weights_str = '-'.join([f"{u.float_to_string(value)}" for _, value in self.weights.items()])
        if comment:
            self.weights_str = f"{weights_str}_{comment}"
        else:
            self.weights_str = weights_str

        # Spectrogram settings
        self.duration = config['duration']
        self.overlap = config['overlap']
        self.desired_fs = config['desired_fs']
        self.channel = config['channel']
        self.log = config['log']
        self.color = config['color']

        # Folders
        self.wavs_folder = pathlib.Path(config['wavs_folder'])
        self.d_train_path = config['d_train_path']
        self.d_valid_path = config['d_valid_path']
        self.d_test_path = config['d_test_path']

        self.nfft = config['nfft']
        self.win_len = config['win_len']
        self.hop_length = int(self.win_len / config['hop_ratio'])
        self.win_overlap = self.win_len - self.hop_length

        # Training parameters
        self.lr = float(config.get('lr', 1e-5))
        self.epochs = config.get('epochs', 10)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # Model paths
        self.model_path = config.get('model_path', 'davidrrobinson/BioLingual')
        self.result_dir = config.get('result_dir', f"/srv/CLAP/3_1_Results/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}_{self.weights_str}")
        self.model_folder = os.path.join(self.result_dir, "model")
        os.makedirs(self.model_folder, exist_ok=True)

    def _prepare_data(self, sample_dur, stop_shuffle):
        """Prepare data loaders for training, validation, and testing."""
        def load_data(path):
            return np.array([os.path.join(self.wavs_folder, i) for i in np.genfromtxt(path, dtype='str', delimiter=' ')[:, 0]])

        d_train_loc = load_data(self.d_train_path)
        d_valid_loc = load_data(self.d_valid_path)
        d_test_loc = load_data(self.d_test_path)

        d_train = u.process_filenames(d_train_loc)
        d_valid = u.process_filenames(d_valid_loc)
        d_test = u.process_filenames(d_test_loc)


        # d_train = u.process_filenames(d_train_loc)[0:50]
        # d_valid = u.process_filenames(d_valid_loc)[0:25]
        # d_test = u.process_filenames(d_test_loc)[0:25]

        
        label_counts_train = d_train["label"].value_counts()
        valid_train_labels = label_counts_train[label_counts_train >= 5].index

        d_train = d_train[d_train["label"].isin(valid_train_labels)]
        d_valid = d_valid[d_valid["label"].isin(valid_train_labels)]
        d_test = d_test[d_test["label"].isin(valid_train_labels)]
        self.d_test=d_test
        num_classes = len(valid_train_labels)
        descriptions = d_train['label'].unique()
        sorted_descriptions = sorted(descriptions, key=lambda desc: self._sort_description(desc))
        self.ids = {lbl: i for i, lbl in enumerate(sorted_descriptions)}

        dataloader_train = torch.utils.data.DataLoader(
            dataset=u.DatasetWaveform(df=d_train, wavs_folder=self.wavs_folder, desired_fs=self.desired_fs, max_duration=sample_dur, ids=self.ids),
            batch_size=self.batch_size,
            shuffle=not stop_shuffle
        )

        dataloader_val = torch.utils.data.DataLoader(
            dataset=u.DatasetWaveform(df=d_valid, wavs_folder=self.wavs_folder, desired_fs=self.desired_fs, max_duration=sample_dur, ids=self.ids),
            batch_size=self.batch_size,
            shuffle=not stop_shuffle
        )

        dataloader_test = torch.utils.data.DataLoader(
            dataset=u.DatasetWaveform(df=d_test, wavs_folder=self.wavs_folder, desired_fs=self.desired_fs, max_duration=sample_dur, ids=self.ids),
            batch_size=self.batch_size,
            shuffle=stop_shuffle
        )
        
        return dataloader_train, dataloader_val, dataloader_test, num_classes

    def _sort_description(self, desc):
        """Helper function to sort descriptions based on distance."""
        if '10+ km' in desc:
            return 10
        distance = desc.split('distance ')[1].split(' km')[0]
        start, end = map(int, distance.split('-'))
        return (start + end) / 2

    def _initialize_model(self, num_classes, a, b):
        """Initialize and return the model with similarity matrices."""
        similarity_matrices = {
            'default': u.similarity(self.ids, self.device, a, b, **self.weights),
            'distance': u.similarity(self.ids, self.device, a, b, distance_weight=1),
            'speed': u.similarity(self.ids, self.device, a, b, speed_weight=1),
            'activity': u.similarity(self.ids, self.device, a, b, activity_weight=1),
            'type': u.similarity(self.ids, self.device, a, b, vessel_type_weight=1)
        }

        model = models.CLAPClassifier(
            self.model_path,
            num_classes,
            sr=self.desired_fs,
            device=self.device,
            similarity_matrix=similarity_matrices['default'],
            multi_label=False
        )

        return model, similarity_matrices

    def _train_epoch(self, model, dataloader_train, optimizer, similarity_matrices):
        """Train the model for one epoch and return metrics."""
        model.train()
        train_loss = 0.0
        train_steps = 0
        metrics = {key: [] for key in similarity_matrices.keys()}

        for x, y in tqdm(dataloader_train, desc='train'):
            optimizer.zero_grad()
            x, y = x.to(self.device), y.to(self.device)
            loss, logits = model(x, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.cpu().item()
            train_steps += 1
            for key, sim_matrix in similarity_matrices.items():
                metrics[key] = u.metrics_calculator(sim_matrix, logits, metrics[key], y)

            print(f"Training metrics: {' | '.join([f'{key}: {np.mean(metrics[key])}' for key in metrics])}")

        return train_loss / train_steps, metrics


    def _save_model(self, model):
        """Save the model and its components."""
        model.clap.save_pretrained(self.model_folder)
        model.processor.save_pretrained(self.model_folder)
        torch.save(model.linear.state_dict(), os.path.join(self.model_folder, 'linear.pth'))

    
    def _evaluate(self, model, dataloader, similarity_matrices, description):
        """Evaluate the model on a dataset and return metrics."""
        results = u.eval_pytorch_model(
            model=model,
            result_dir=self.result_dir,
            dataloader=dataloader,
            metric_factory=u.Accuracy,
            device=self.device,
            similarity_matrix=similarity_matrices['default'],
            similarity_matrix_distance=similarity_matrices['distance'],
            similarity_matrix_speed=similarity_matrices['speed'],
            similarity_matrix_activity=similarity_matrices['activity'],
            similarity_matrix_type=similarity_matrices['type'],
            desc=description,
            weights=self.weights,
            comment=self.weights_str,
            ids=self.ids
        )
        return results

    def extract_timestamps(self,filename):
        # Regular expression to match timestamps in the format YYYY-MM-DD_HH-MM-SS
        timestamp_pattern = r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}'
        timestamps = re.findall(timestamp_pattern, filename)[0]
        return timestamps



    def plot_confusion_matrix(self, cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues, save_path=None):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            # print("Normalized confusion matrix")

    
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
    
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], fmt),
                         ha="center", va="center",
                         color="white" if cm[i, j] > thresh else "black")
    
        plt.tight_layout()
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
    
        if save_path:
            plt.savefig(save_path)
            print(f"Confusion matrix saved to {save_path}")
        else:
            plt.show()

    def train_clap(self, a=5, b=0.5, sample_dur=10, batch_size=8,stop_shuffle=False):
        self.batch_size=batch_size
        self.epochs=1
        dataloader_train, dataloader_val, dataloader_test, num_classes = self._prepare_data(sample_dur, stop_shuffle)
        model, similarity_matrices = self._initialize_model(num_classes, a, b)

        optimizer = optim.Adam(params=model.parameters(), lr=self.lr)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        valid_loss_best = float('inf')
        best_model = model
        break_next = False

        log_file_path = os.path.join(self.result_dir, 'logs.log')
        with open(log_file_path, mode='w') as log_file:
            log_file.write(f"lr = {self.lr}\n")

            for epoch in range(self.epochs):
                sys.stderr.write(f'epoch = {epoch}\n')

                train_loss, metrics = self._train_epoch(model, dataloader_train, optimizer,similarity_matrices)
                # Evaluate the model on the validation set
                valid_results = self._evaluate(model, dataloader_val, similarity_matrices, 'valid')
                valid_loss = valid_results[0]
                valid_metrics = valid_results[1:6]  # Extract the metrics excluding the mse

                scheduler.step()
                log_file.write(f"Learning rate after epoch {epoch}: {scheduler.get_last_lr()[0]}\n")

                if valid_loss < valid_loss_best:
                    valid_loss_best = valid_loss
                    best_model = copy.deepcopy(model)
                    self._save_model(best_model)

                if min(valid_loss, valid_loss_best) > valid_loss_best:
                    log_file.write("breaking next cycle\n")
                    if break_next:
                        break
                    break_next = True
                log_message = json.dumps({
                    'epoch': epoch,
                    'train': {'loss': train_loss, **{key: np.mean(value) for key, value in metrics.items()}},
                    'valid': {'loss': valid_loss, **{f'metric_{i+1}': np.mean(metric) for i, metric in enumerate(valid_metrics)}}
                })
                log_file.write(log_message + '\n')

            # Final evaluation on the test set
            test_results = self._evaluate(best_model, dataloader_test, similarity_matrices, 'test')
            test_loss = test_results[0]
            test_metrics = test_results[1:6]
            test_df = test_results[-1]

            # Combine the test results DataFrame with the corresponding filenames
            combined_df = pd.concat([self.d_test.reset_index(drop=True), test_df.reset_index(drop=True)], axis=1)

            # Drop rows with missing filenames
            nan_indices = combined_df[combined_df['filename'].isna()].index
            self.d_test = self.d_test.drop(nan_indices + 1)

            # Extract timestamps from filenames
            # Apply the extract_timestamps method to the 'filename' column
            combined_df['timestamp'] = combined_df['filename'].apply(self.extract_timestamps)
            
            # Replace underscores with hyphens in the 'timestamp' column
            combined_df['timestamp'] = combined_df['timestamp'].str.replace('_', '-')
            
            # Extract the predicted category from the 'predicted' column
            combined_df['predicted_category'] = combined_df["predicted"].apply(lambda x: x.split(" ")[3])
            
            # Drop the 'true' column
            combined_df = combined_df.drop(columns=["true"])



            # Assuming 'combined_df' is your DataFrame containing 'predicted_category' and 'distance_category'
            actual_values = combined_df['distance_category']
            predicted_values = combined_df['predicted_category']
            labels = sorted(set(actual_values) | set(predicted_values))
            labels.sort(key=lambda x: float(x.split('-')[0]) if x != '10+' else float('inf'))
            

            
            # Compute confusion matrix
            cm = confusion_matrix(actual_values, predicted_values, labels=labels)
            
            # Convert categories to numeric values for MSE calculation
            actual_numeric = actual_values.apply(lambda x: float(x.split('-')[0]) if x != '10+' else float('inf'))
            predicted_numeric = predicted_values.apply(lambda x: float(x.split('-')[0]) if x != '10+' else float('inf'))
            
            # Calculate MSE and RMSE
            mse = mean_squared_error(actual_numeric, predicted_numeric)
            rmse = np.sqrt(mse)
            print(f"Mean Squared Error: {mse}")
            print(f"Root Mean Squared Error (RMSE): {rmse} km")
            
            # Define the title with MSE and RMSE
            title = f'Confusion Matrix (MSE: {mse:.2f}, RMSE: {rmse:.2f} km)'
            
            # Define the path to save the confusion matrix plot
            cm_save_path = os.path.join(self.result_dir, 'confusion_matrix.png')
            self.plot_confusion_matrix(cm, classes=labels, title=title, save_path=cm_save_path)

            # Save DataFrame to CSV
            csv_file_path = os.path.join(self.result_dir, os.path.basename(self.result_dir.rstrip('/')) + '.csv')
            combined_df.to_csv(csv_file_path, index=False)
            print("CSV file saved successfully.")
            
            log_message = json.dumps({
                'test': {'loss': test_loss, **{f'metric_{i+1}': np.mean(metric) for i, metric in enumerate(test_metrics)}}
            })
            log_file.write(log_message + '\n')

        return best_model, test_loss
