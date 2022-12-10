"""Unit tests for Gaussian Process Regression on Trajectories"""

#pylint: disable=unused-import

import unittest

import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import axes3d

from steal.datasets import lasa
from steal.gp import ModelGP


def load_trajectories(dataset_name="heee"):
    """Load the demontstration trajectories from LASA with the name"""
    if hasattr(lasa.DataSet, dataset_name):
        dataset = getattr(lasa.DataSet, dataset_name)
        return dataset.demos
    else:
        raise ValueError(
            "Invalid dataset name specified. Please check the LASA dataset repo for valid names."
        )


def train_data(trajectories, i):
    """
    Training data.
    Input is time and output is the position.
    """
    X = torch.tensor(trajectories[i].t[0])
    y = torch.tensor((trajectories[i].pos).T)
    return X, y


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


class TestGaussianProcess(unittest.TestCase):
    """Tests for a scalar valued Gaussian Process on the LASA dataset."""

    def test_load_trajectories(self):
        """Test if loading of trajectories is correct."""
        trajectories = load_trajectories()
        assert len(trajectories) == 7

        train_t, train_xy = concatenate_trajectories(trajectories)
        assert train_t.shape == (7000, )
        assert train_xy.shape == (7000, 2)

    def test_gp(self):
        """Test training of a scalar-valued GP."""
        trajectories = load_trajectories()
        assert len(trajectories) == 7

        # Concatenating the demos
        train_t, train_xy = concatenate_trajectories(trajectories)
        train_x = train_xy[:, 0]
        train_y = train_xy[:, 1]

        # Time
        train_t = torch.tensor(train_t)
        # X Position
        train_x = torch.tensor(train_x)
        # Y Position
        train_y = torch.tensor(train_y)

        training_iters = 22

        # GP Model for time vs X position
        # initialize GP
        m1 = ModelGP(train_t, train_x)
        model = m1.get_model()
        m1.training(train_t, train_x, training_iters)
        likelihood = m1.evaluation()

        # Test points are regularly spaced along [0,6]
        # Make predictions by feeding model through likelihood
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            test_t = torch.linspace(0, 6, 1000).double()
            observed_pred = likelihood(model(test_t))

        # GP Model for time vs Y position
        m2 = ModelGP(train_t, train_y)
        model1 = m2.get_model()
        m2.training(train_t, train_y, training_iters)
        likelihood1 = m2.evaluation()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            test_t = torch.linspace(0, 6, 1000).double()
            observed_pred1 = likelihood1(model1(test_t))

        trajectories_to_plot = [(trajectory.t[0], trajectory.pos[0, :])
                                for trajectory in trajectories]
        self.plot_gp(observed_pred,
                     trajectories_to_plot,
                     means=(test_t.numpy(), observed_pred.mean.numpy()),
                     xlim=[0, 6.0],
                     ylim=[-40, 15],
                     xlabel="Time",
                     ylabel="X-position",
                     image_name="x_time_GP.png")

        trajectories_to_plot = [(trajectory.t[0], trajectory.pos[1, :])
                                for trajectory in trajectories]
        self.plot_gp(observed_pred1,
                     trajectories_to_plot,
                     means=(test_t.numpy(), observed_pred1.mean.numpy()),
                     xlim=[0, 6.0],
                     ylim=[-25, 30],
                     xlabel="Time",
                     ylabel="Y-position",
                     image_name="y_time_GP.png")

        trajectories_to_plot = [(trajectory.pos[0, :], trajectory.pos[1, :])
                                for trajectory in trajectories]
        self.plot_gp(observed_pred1,
                     trajectories_to_plot,
                     means=(observed_pred.mean.numpy(),
                            observed_pred1.mean.numpy()),
                     xlim=[-40, 15],
                     ylim=[-25, 30],
                     xlabel="X-position",
                     ylabel="Y-position",
                     plot_intervals=False)

        self.plot_3d_traj(trajectories,
                          test_t,
                          observed_preds=(observed_pred, observed_pred1))

    def plot_gp(self,
                observed_pred,
                trajectories,
                means,
                xlim=(0, 6.0),
                ylim=(-40, 15),
                xlabel="Time",
                ylabel="X-position",
                image_name=None,
                plot_intervals=True):
        """Plot the gaussian process with confidence intervales"""
        # Plot graphs
        with torch.no_grad():
            # Initialize plot
            _, ax = plt.subplots(1, 1, figsize=(4, 3))

            # Get upper and lower confidence bounds
            lower, upper = observed_pred.confidence_region()

            # Plot training data as dotted lines
            for x, y in trajectories:
                ax.plot(x, y, '--')

            # Plot predictive means as blue line
            ax.plot(means[0], means[1], 'b')

            if plot_intervals:
                # Shade between the lower and upper confidence bounds
                ax.fill_between(means[0],
                                lower.numpy(),
                                upper.numpy(),
                                alpha=0.5)

            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.legend([
                'Observed Demo 1', 'Observed Demo 2', 'Observed Demo 3',
                'Observed Demo 4', 'Observed Demo 5', 'Observed Demo 6',
                'Observed Demo 7', 'Mean', 'Confidence'
            ])
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            if image_name:
                plt.savefig(image_name)

            plt.show()

    def plot_3d_traj(self, trajectories, test_t, observed_preds):

        with torch.no_grad():

            fig = plt.figure(figsize=(4, 4))
            ax = fig.add_subplot(111, projection='3d')

            # Plot training data as black stars
            for trajectory in trajectories:
                train_x = trajectory.pos[0, :]
                train_y = trajectory.pos[1, :]
                ax.plot(train_y, train_x, test_t, '--')

            # Plot predictive means as blue line
            ax.plot(observed_preds[1].mean.numpy(),
                    observed_preds[0].mean.numpy(), test_t, 'b')
            ax.legend([
                'Observed Demo 1',
                'Observed Demo 2',
                'Observed Demo 3',
                'Observed Demo 4',
                'Observed Demo 5',
                'Observed Demo 6',
                'Observed Demo 7',
                'Predicted trajectory',
            ])
            ax.set_xlabel('Y-position')
            ax.set_ylabel('X-position')
            ax.set_zlabel('Time')
            # plt.show()
            plt.show()