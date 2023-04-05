import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    A class for computing the focal loss, which is a modification of the cross-entropy loss
    that downweights the contribution of well-classified examples.
    """
    def __init__(self, gamma=2, alpha=None, reduction='mean'):
        """
        Constructor method for the FocalLoss class.

        Parameters:
        gamma (float): The focusing parameter. Default is 2.
        alpha (float or list): The weight for each class. If None, the weight will be calculated
                               based on the frequency of the classes. Default is None.
        reduction (string): The reduction method. Can be 'none', 'mean', or 'sum'. Default is 'mean'.
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)):
            self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list):
            self.alpha = torch.Tensor(alpha)
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Computes the focal loss.

        Parameters:
        inputs (torch.Tensor): The output tensor from the neural network.
        targets (torch.Tensor): The target tensor containing the labels.

        Returns:
        The focal loss.
        """
        CE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction=self.reduction)
        pt = torch.exp(-CE_loss)
        F_loss = ((1-pt)**self.gamma)*CE_loss
        if self.alpha is not None:
            F_loss = self.alpha[1]*F_loss*self.alpha[0]*F_loss
        return F_loss
