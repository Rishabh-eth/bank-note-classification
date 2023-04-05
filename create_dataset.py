import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from dataset import BanknoteDataset

def create_dataset(label_file, data_folder):

    """
    This function loads the dataset from the given label file and data folder,
    creates a list of all image files, all labels, and splits the dataset into 
    train and test sets. It then applies the necessary image transforms to the 
    train and test sets, creates and returns data loaders for both train and 
    test sets.

    Args:
    - label_file (str): path to the file containing the labels for the dataset
    - data_folder (str): path to the folder containing the dataset images

    Returns:
    - train_loader (DataLoader): data loader for the train set
    - test_loader (DataLoader): data loader for the test set
    """


    # Load the labels for the altered images
    df_labels = pd.read_csv(label_file, header=None)
    df_labels = df_labels.iloc[1:, :]
    df_labels.columns = ['filename', 'label']
    df_labels['filename'] = df_labels['filename'].apply(lambda x: os.path.join(data_folder, 'altered', x))
    df_labels['label'] = df_labels['label'].astype(int)

    # Create a list of all image files
    original_files = [os.path.join(data_folder, 'originals', f) for f in os.listdir(os.path.join(data_folder, 'originals')) if os.path.isfile(os.path.join(data_folder, 'originals', f))]
    altered_files = df_labels['filename'].tolist()
    all_files = original_files + altered_files

    # Create a list of all labels
    all_labels = np.concatenate([np.ones(len(original_files)), df_labels['label'].values])

    # Split the dataset into train and test sets
    train_files, test_files, train_labels, test_labels = train_test_split(all_files, all_labels, test_size=0.2, stratify=all_labels)

    # Define the transforms to apply to the images
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)), #Randomly crop and resize the images during training
        transforms.RandomRotation(degrees=15), # Randomly rotate the images during training
        transforms.RandomHorizontalFlip(), # Randomly flip the images horizontally during training
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1), # Randomly adjust the color of the images during training
        # transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        # transforms.RandomVerticalFlip(p=0.5),
        # transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=10),
        # transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        # transforms.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
        # # transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(), # Convert the image to a PyTorch tensor
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Normalize the image tensors
    ])
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)), # Resize the images during testing
        transforms.ToTensor(), # Convert the image to a PyTorch tensor
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Normalize the tensor with mean and standard deviation values
    ])

    # Create the datasets
    train_dataset = BanknoteDataset(train_files, train_labels, transform=train_transform)
    test_dataset = BanknoteDataset(test_files, test_labels, transform=test_transform)

    # Create the data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

    return train_loader, test_loader