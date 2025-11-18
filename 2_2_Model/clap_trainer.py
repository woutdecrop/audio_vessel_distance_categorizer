import os
import yaml
import torch
from dataset import CLAP_Vessel_Distance
from ray import tune
from ray.tune.schedulers import ASHAScheduler
import numpy as np
# Load configuration file
config_path = os.path.join('..', 'config.yaml')
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)
from utils import delete_all_subfolders
# print(config)
# Define param_a training function for Ray Tune

delete_all_subfolders("/srv/CLAP/temporary")


def train_fn(config, checkpoint_dir=None):
    # Extract hyperparameters from the config passed by Ray Tune

    param_a = str(config["param_a"])
    param_b = str(config["param_b"])
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    learning_rate = config["learning_rate"]
    L2=config["L2"]
    # Initialize the model with custom layer sizes
    L = CLAP_Vessel_Distance(config, batch_size=batch_size, epochs=epochs, lr=learning_rate,freeze_clap=True, save_model=True ,L2=L2 ) 
    
    # Train the model
    L.train_CLAP(param_a=param_a, param_b=param_b)
    

# Define the search space for the hyperparameters
def get_search_space():
    # Define param_a with values from -3 to 3 in steps of 0.1
    param_a_values = np.round(np.arange(-3, 3.1, 0.1), 1).tolist()
    param_a_values = [x for x in param_a_values if x != 0]
    # Define param_b with values from 0.4 to 0.7 in steps of 0.1
    param_b_values = np.round(np.arange(0.4, 0.8, 0.1), 1).tolist()
    
    return {
        "param_a": tune.choice(param_a_values),  # Choose from rounded values of `param_a`
        "param_b": tune.choice(param_b_values),  # Choose from rounded values of `param_b`
        "batch_size": tune.choice([8,16]),  # Batch size options
        "learning_rate": tune.choice([1e-3,1e-4, 1e-5]), # Log-uniform distribution for learning rate
        "epochs": 100, # Total number of epochs for the training
        "L2": tune.choice([True, False])
    }

# Define the number of trials and search method
num_samples = 10000 # Number of trials to run in parallel
print("Number iterations", num_samples)
# Run hyperparameter search using Ray Tune
scheduler = ASHAScheduler(
    metric="val_mse",  # Metric to optimize
    mode="min",         # Minimize validation loss
    max_t=100,          # Max number of epochs
    grace_period=3,     # Minimum number of epochs to run
    reduction_factor=2, # Reduce resources by factor of 2
)

# Merge the YAML config with the search space
search_space = get_search_space()
merged_config = {**config, **search_space}  # Merging both config and search space

analysis = tune.run(
    train_fn,
    name="tune_vessel_distance",
    config=merged_config,  # Pass the merged configuration to Ray Tune
    num_samples=num_samples,
    scheduler=scheduler,
    resources_per_trial={"cpu": 8, "gpu": 0.125},  # Adjust the resources for each trial
)

# Optionally, you can access the best trial's results
best_trial = analysis.get_best_trial("val_mse", "min", "all")
print(f"Best trial validation loss: {best_trial.last_result['val_mse']}")

# Assuming `analysis` is the result of `tune.run()`
best_config = analysis.get_best_config(metric="val_mse", mode="min")
print("Best configuration found: ", best_config)

