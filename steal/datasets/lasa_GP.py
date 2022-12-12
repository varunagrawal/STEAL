"""Module for sampling LASA trajectories from GP model."""

import gpytorch
import numpy as np
import pyLasaDataset as lasa
import torch
from matplotlib import pyplot as plt

from steal.datasets import lasa
from steal.gp.multi_task import MultitaskApproximateGaussianProcess


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


def get_lasa_samples(dataset_name="heee", num_samples=10):
    """Get samples from a GP model trained on a LASA dataset.

    Args:
        dataset_name (str, optional): The specific shape dataset. Defaults to "heee".
        num_samples (int, optional): Then number of samples to return. Defaults to 10.

    Returns:
        torch.Tensor: Trajectory samples.
    """

    trajectories = load_trajectories(dataset_name=dataset_name)

    # Concatenating the demos
    train_t, train_xy = concatenate_trajectories(trajectories)

    # Time
    train_t = torch.tensor(train_t)
    # X, Y Position
    train_xy = torch.tensor(train_xy)

    num_latents = 3
    num_tasks = 2
    gp = MultitaskApproximateGaussianProcess(num_tasks=num_tasks,
                                             num_latents=num_latents)

    num_epochs = 60
    gp.train(train_t, train_xy, training_iterations=num_epochs)

    # Make predictions
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        test_t = torch.linspace(0, 6, 1000).double()
        samples = gp.samples(test_t, num_samples)

    return samples
