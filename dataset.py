import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class BanknoteDataset(Dataset):
    """
    Custom dataset class for banknote images.
    """
    
    def __init__(self, file_list, label_list, transform=None):
        """
        Initializes the dataset with a list of file paths and corresponding labels.

        Args:
        - file_list (list): List of file paths for the banknote images.
        - label_list (list): List of labels for the banknote images.
        - transform (callable, optional): A function/transform that takes in a PIL image and returns a transformed version. Default is None.
        """
        self.file_list = file_list
        self.label_list = label_list
        self.transform = transform
        
    def __len__(self):
        """
        Returns the length of the dataset.
        """
        return len(self.file_list)

    def __getitem__(self, index):
        """
        Returns the item at the given index.

        Args:
        - index (int): Index of the item to return.

        Returns:
        - img (tensor): Tensor of the banknote image at the given index.
        - label (tensor): Tensor of the label for the banknote image at the given index.
        """
        # Open the image at the given file path as a PIL image
        img_path = self.file_list[index]
        img = Image.open(img_path).convert('RGB')

        # Get the label for the image at the given index as a tensor of dtype float32
        label = torch.tensor(int(self.label_list[index]), dtype=torch.float32)

        # If there is a transform function, apply it to the image
        if self.transform is not None:
            img = self.transform(img)

        # Return the transformed image and its label
        return img, label
