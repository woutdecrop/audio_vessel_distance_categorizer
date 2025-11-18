import torchvision.models as torchmodels
from torch import nn
import utils as u
import torch
import os
import torch.nn.functional as F
from transformers import AutoProcessor, ClapModel, ClapAudioModelWithProjection, ClapProcessor
import torch.nn.functional as F

import pathlib
import soundfile as sf
import torchaudio
import torch

torch.hub.set_dir('/data/woutdecrop/torch/')
    
class CLAPClassifier(nn.Module):
    def __init__(self, model_path, num_classes, sr, device, similarity_matrix,embeddings_path, freeze_clap=False) -> None:
        super(CLAPClassifier, self).__init__()
        
        # Initialize the CLAP model and processor
        self.clap = ClapAudioModelWithProjection.from_pretrained(model_path)
        self.freeze_clap=freeze_clap
        if freeze_clap:
            self.fc1 = nn.Linear(in_features=512, out_features=256)
            self.fc2 = nn.Linear(in_features=256, out_features=num_classes)
            # Activation function and dropout
            self.relu = nn.ReLU()
        else:
            self.linear = nn.Linear(in_features=512, out_features=num_classes)
        # Load layer weights
        self._load_layer_weights(model_path, device)
            
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.device = device
        self.sr = sr
        self.embedding_dir = embeddings_path
        # Move the similarity matrix to the appropriate device
        self.loss_func = CustomLossFunction(torch.tensor(similarity_matrix).to(device))
        
        # Move model parameters to the specified device
        self.to(device)
        
    def forward(self, x, y=None, desc=None,wav_path=None, creating_embeddings=False):
        if creating_embeddings:
            os.makedirs(os.path.join(self.embedding_dir, desc), exist_ok=True)
            wav_path=pathlib.Path(wav_path)
            wav_filename = os.path.basename(wav_path)
            embedding_filename = os.path.splitext(wav_filename)[0] + '.pt'
            embedding_path = os.path.join(self.embedding_dir, desc, embedding_filename)
            # Check if the embedding is already computed and saved
            if os.path.exists(embedding_path):
                # Load the saved embedding
                out = torch.load(embedding_path, map_location=self.device)
                print(f"Loaded saved embedding for input with hash {embedding_path}")
            else:
                # Ensure input data is on the correct device
                x = [s.cpu().numpy() for s in x]
                # Process inputs with the processor
                inputs = self.processor(audios=x, return_tensors="pt", sampling_rate=self.sr, padding=True)
                inputs = {key: value.to(self.device) for key, value in inputs.items()}

                if self.freeze_clap:
                    # Get embeddings from the CLAP model
                    out = self.clap(**inputs).audio_embeds.to(self.device)
                    # Save the computed embedding
                    torch.save(out, embedding_path)
                else:
                    # Save the computed embedding
                    torch.save(inputs["input_features"], embedding_path)
            return
        else:
            # print(x)
            x=x.to(self.device).squeeze(1)
            # print(x)
            if self.freeze_clap:
                out = self.fc1(x)
                out = self.relu(out)
                out = self.fc2(out)
                
            else:
                x=self.clap(x).audio_embeds.to(self.device)
                # Pass through the linear layers without activation or dropout
                out = self.linear(x)
            # Calculate loss if labels are provided
            if y is not None:
                y = y.to(self.device)
                loss = self.loss_func(out, y)
                return loss, out
            
            return out
    def _load_layer_weights(self, model_path, device):
        """
        Loads weights for all defined layers if corresponding files exist.
        """
        for layer_name, layer in self.named_children():
            layer_weights_path = os.path.join(model_path, f"{layer_name}.pth")
            if os.path.exists(layer_weights_path):
                try:
                    layer.load_state_dict(torch.load(layer_weights_path, map_location=device))
                    print(f"Loaded weights for layer '{layer_name}' from {layer_weights_path}")
                except Exception as e:
                    print(f"Failed to load weights for layer '{layer_name}': {e}")
            else:
                print(f"No weights file found for layer '{layer_name}' at {layer_weights_path}, initializing with random weights.")

class CustomLossFunction(nn.Module):
    def __init__(self, similarity_matrix):
        super(CustomLossFunction, self).__init__()
        self.similarity_matrix = similarity_matrix

    def forward(self, outputs, target):
        # print("output", outputs)
        pred_softmax = F.softmax(outputs, dim=-1)
        pred = F.log_softmax(outputs, dim=-1)

        # Select the similarity row corresponding to the target
        similarity_row = self.similarity_matrix[target]
        # Multiply the input row-wise with the selected similarity row
        tensor = pred * similarity_row
    
        sum_over_columns = torch.sum(tensor, dim=1)
    
        # Take the average over all the rows
        average_over_rows = torch.mean(sum_over_columns)
        
        # Calculate the negative log likelihood
        return -average_over_rows/sum(sum(self.similarity_matrix))
