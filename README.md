# Vessel Distance Classification with CLAP

## Overview

This repository contains the implementation of a convolutional neural network (CNN) model for vessel distance classification using underwater acoustic data. The model is based on the Contrastive Language-Audio Pretraining (CLAP) framework and is designed for transfer learning to predict the distance of vessels from hydrophones deployed in the Belgian part of the North Sea (BPNS).

The dataset used in this project consists of synchronized acoustic and AIS (Automatic Identification System) data, allowing for supervised classification based on vessel presence and proximity. The model is described in:

**Decrop Wout, Deneudt Klaas, Parcerisas Clea, Schall Elena, Debusschere Elisabeth, 2025. Transfer learning for distance classification of marine vessels using underwater sound. Submitted to IEEE.**

---

## Folder Structure

```
├── 1_Data_Split              # Train/test/validation split information
├── 2_1_Data                  # Contains dataset text files with .wav file locations
├── 2_2_Model                 # Model training and testing scripts
├── 3_1_Results               # Stores model outputs and predictions
├── 3_2_Check_Performance     # Performance evaluation scripts and metrics
├── 3_3_figures               # Plots and visualizations
├── 4_Export                  # Exported model weights and configurations
├── __pycache__               # Cached Python files
├── config.yaml               # Configuration file for running experiments
├── requirements.txt          # Required dependencies
├── README.md                 # This file
```

---

## Model Training

The core training function is `train_fn`, which is responsible for handling the training process and hyperparameter tuning.

```python
def train_fn(config, checkpoint_dir=None):
    # Extract hyperparameters from the config passed by Ray Tune
    param_a = str(config["param_a"])
    param_b = str(config["param_b"])
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    learning_rate = config["learning_rate"]
    L2 = config["L2"]
    
    # Initialize the model with the given parameters
    L = CLAP_Vessel_Distance(config, batch_size=batch_size, epochs=epochs, lr=learning_rate,
                             freeze_clap=True, save_model=True, L2=L2)
    
    # Train the model with the specified parameters
    L.train_CLAP(param_a=param_a, param_b=param_b)
```

### Explanation:
- `param_a` and `param_b`: Custom parameters used for training variations.
- `batch_size`: Defines the number of samples per training batch.
- `epochs`: Number of training iterations.
- `learning_rate`: Defines the step size for optimization.
- `L2`: If `True`, the model uses the L2 loss function; otherwise, it follows standard loss settings.
- `freeze_clap`: Keeps pre-trained CLAP layers frozen during training.
- `save_model`: Saves the trained model when set to `True`.

### Running Training

- **Grid Search for Hyperparameters**
  ```bash
  python clap_trainer.py
  ```
  This performs a grid search to find the optimal hyperparameters.

- **Training with a Fixed Loss Function**
  ```bash
  python clap_test.py
  ```
  This runs the model with a specific loss function setup.

---

## Configuration

The `config.yaml` file contains all hyperparameter settings, including learning rate, batch size, and loss function choices. Modify this file to customize the training setup.

---

## Requirements

Install the required dependencies:
```bash
pip install -r requirements.txt
```

---

## Results & Performance

The trained models are stored in `3_1_Results/`. You can visualize results using scripts in `3_3_figures/` and analyze performance metrics in `3_2_Check_Performance/`.

---

## Citation

If you use this model, please cite:

- Decrop Wout, Deneudt Klaas, Parcerisas Clea, Schall Elena, Debusschere Elisabeth, 2025. Transfer learning for distance classification of marine vessels using underwater sound. Submitted to IEEE.

---

## License

This repository is licensed under [MIT License](LICENSE).
