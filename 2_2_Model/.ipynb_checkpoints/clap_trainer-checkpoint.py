from dataset import CLAP_Vessel_Distance
import yaml
import pathlib
import numpy as np
import os
import pandas as pd

config_path = os.path.join('..', 'config.yaml')
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

weights = {
    'distance_weight': 1,
    'speed_weight': 0,
    'activity_weight': 0,
    'vessel_type_weight': 0
}

L=CLAP_Vessel_Distance(config)

L.train_clap(a=5,b=0.5, batch_size=100)
# L.train_clap(model_path='/srv/CLAP/roi/BioLingual/model/1_0_0_0_filtered_classes_window4')