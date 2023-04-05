import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from omegaconf import DictConfig
import hydra
from torch.utils.data import DataLoader
from focal_loss import FocalLoss
import wandb

def train_model(cfg: DictConfig, train_loader: DataLoader):
    """
    Train a neural network model using the specified configuration and data loader.

    Args:
        cfg: A dictionary-like object containing the configuration parameters.
        train_loader: A PyTorch DataLoader object representing the training data.

    Returns:
        The trained PyTorch model.
    """

    # Define model architecture
    model = models.resnet18(pretrained=cfg.model.pretrained)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, cfg.model.num_ftrs)

    # Define loss function and optimizer
    if cfg.model.loss == 'focal':
        criterion = FocalLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=cfg.training.lr, momentum=cfg.training.momentum,
                          weight_decay=cfg.training.weight_decay)

    # wandb.init(project="VU_project", entity="bank-note_clf")
    # wandb.config.update(cfg)

    # Train the model
    for epoch in range(cfg.training.num_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels.float().unsqueeze(1))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)

        print('[Epoch %d] Loss: %.3f' % (epoch + 1, avg_loss))
        
        # wandb.log({"training_loss": avg_loss})

    # Save the model
    torch.save(model.state_dict(), cfg.output_path)

    return model

