from dataset import CLAP_Vessel_Distance
import yaml
import os


config_path = os.path.join('..', 'config.yaml')
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)


L = CLAP_Vessel_Distance(config, batch_size=16, epochs=20, lr=1e-4,freeze_clap=False, save_model=False, L2=True )

# Train the model
L.train_CLAP(param_a=str(1.4), param_b=str(0.5))
