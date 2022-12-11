"""Module for sampling LASA trajectories from GP model."""

import numpy as np
import pyLasaDataset as lasa
import torch
import gpytorch
from matplotlib import pyplot as plt

from steal.datasets import lasa
from steal.gp.multi_task import MultitaskApproximateGP

def load_trajectories(dataset_name="heee"):
    """Load the demontstration trajectories from LASA with the name"""
    if hasattr(lasa.DataSet, dataset_name):
        dataset = getattr(lasa.DataSet, dataset_name)
        return dataset.demos
    else:
        raise ValueError(
            "Invalid dataset name specified. Please check the LASA dataset repo for valid names."
        )


def concatenate_trajectories(trajectories):
    """Concatenate the input into a single array"""
    train_t = np.empty((0, ))
    train_xy = np.empty((0, 2))
    for i, trajectory in enumerate(trajectories):
        train_t = np.hstack(
            (train_t, trajectory.t[0])) if train_t.size else trajectory.t[0]
        train_xy = np.vstack(
            (train_xy,
             trajectory.pos.T)) if train_xy.size else trajectory.pos.T
    return train_t, train_xy

def get_lasa_samples(dataset_name ="heee", sample_num = 10):
    """ Get LASA trajectory samples from GP model

        input: dataset_name (String), sample_num (int)
        output: samples (torch.Tensor)

    """
    trajectories = load_trajectories()

    # Concatenating the demos
    train_t, train_xy = concatenate_trajectories(trajectories)

    # Time
    train_t = torch.tensor(train_t)
    # X, Y Position
    train_xy = torch.tensor(train_xy)

    num_latents = 3
    num_tasks = 2
    gp = MultitaskApproximateGP(num_tasks=num_tasks, num_latents=num_latents)

    num_epochs = 60
    gp.training(train_t, train_xy, training_iterations=num_epochs)

    model = gp.get_model()
    likelihood = gp.evaluation()

    # Set into eval mode
    model.eval()
    likelihood.eval()

    # Make predictions
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        test_t = torch.linspace(0, 6, 1000).double()
        preds = model(test_t)
        predictions = likelihood(preds)
        mean = predictions.mean
        lower, upper = predictions.confidence_region()
        samples = gp.sampling(preds, 10) 

    return samples 