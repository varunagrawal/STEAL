"""Losses for use in training"""

import torch
from torch import nn


class LagrangianMomentumLoss(nn.Module):
    """
    Loss which computes the difference between
    the predicted momentum and the momentum of the trajectory.
    """

    def __init__(self, criterion=nn.MSELoss()):
        """Initialize the loss."""
        super().__init__()
        self.criterion = criterion

    def forward(self, y_pred, y_target):
        """Forward Pass"""
        f_pred, M_pred, J_subtasks = y_pred
        #TODO: This was regular inverse before!
        M_inv_pred = torch.inverse(M_pred)
        y_pred = torch.einsum('bij, bj -> bi', M_inv_pred, f_pred)
        diff_cspace = y_pred - y_target

        n_subtasks = len(J_subtasks)
        loss = 0.
        for n in range(n_subtasks):
            diff_subtask = torch.einsum('bij, bj -> bi', J_subtasks[n],
                                        diff_cspace)
            loss += self.criterion(diff_subtask,
                                   torch.zeros_like(diff_subtask))
        loss = loss / n_subtasks
        return loss
