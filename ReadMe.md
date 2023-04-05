# Bank Note Classification

This project aims to classify bank notes as acceptable or non-acceptable based on their images. The dataset used in this project consists of original and altered images. The altered images are labeled as 1 (acceptable) and 0(non-acceptable). We use a pre-trained ResNet18 deep neural network architecture with a custom fully connected layer to train and evaluate the model.

## Getting Started

To run this project, follow these steps:

1. Clone this repository to your local machine:

```
git clone https://github.com/Rishabh-eth/bank-note-classification.git
cd bank-note-classification

```

2. Create conda environment:

```
conda env create -f environment.yml

```

3. Download the dataset and extract it to the data directory. Data is hosted [here](https://drive.google.com/drive/folders/1yfafwTvzidgUM-oHLNa6JzUETIOnlkpb)

4. Specify the parameteres required for training in configuration file: [config.yaml](config.yaml)

5. Train and evaluate the model:

```
python main.py

```

## Configuration
This project uses the Hydra library to manage configurations. The configuration file [config.yaml](config.yaml) contains all the hyperparameters and settings for the model and training process. You can modify this file or create new configuration files to experiment with different settings.

## Dependencies
This project requires the following dependencies:

1. Python 3.6 or higher
2. PyTorch 1.8 or higher
3. TorchVision 0.9 or higher
4. Hydra 1.1 or higher
5. WandB 0.12 or higher (optional, for logging training metrics)

# Technical Presentation

The technical presentation can be found in the file: [Technical_presentation](Technical_presentation.pdf)

# Answers

All the answers to intermediate questions asked in task 1 are in the file: [Answers](Answers.pdf)