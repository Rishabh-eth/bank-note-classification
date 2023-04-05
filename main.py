import os
import sys
from create_dataset import create_dataset
from train_model import train_model
from test_model import test_model
from omegaconf import DictConfig
import hydra
import wandb

# Set your API key
# api_key = ''

# Log in to WandB
# wandb.login(key=api_key)

@hydra.main(config_path=".", config_name='config.yaml', version_base=None)
def main(cfg: DictConfig):
    
    # Set path to the data folder and the label file
    data_folder = cfg.dataset.data_path
    label_file = os.path.join(data_folder, cfg.dataset.label_file)

    # Create train and test data loaders
    train_loader, test_loader= create_dataset(label_file, data_folder)

    # Train the model
    model= train_model(cfg, train_loader)

    # Test the model and obtain accuracy and F1 score
    accuracy, f1= test_model(model, test_loader)


if __name__ == '__main__':
    main()




    





