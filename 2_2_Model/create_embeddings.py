from dataset import CLAP_Vessel_Distance
import yaml
import os


config_path = os.path.join('..', 'config.yaml')
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)


L=CLAP_Vessel_Distance(config,batch_size=1, epochs=1, freeze_clap=False)
L.train_CLAP(creating_embeddings=True)