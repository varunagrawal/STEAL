"""Unit tests for Gaussian Process Regression on Trajectories"""

#pylint: disable=unused-import

import os
import unittest

import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import axes3d

from steal.datasets import lasa
from steal.gp import ModelGP, MultitaskApproximateGP, MultitaskExactGP
from steal.utils.plotting.gp import plot_3d_traj, plot_gp


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

        training_iters = 1

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

        if "PYTEST_CURRENT_TEST" not in os.environ:

            legend = [
                'Observed Demo 1', 'Observed Demo 2', 'Observed Demo 3',
                'Observed Demo 4', 'Observed Demo 5', 'Observed Demo 6',
                'Observed Demo 7', 'Mean', 'Confidence'
            ]
            trajectories_to_plot = [(trajectory.t[0], trajectory.pos[0, :])
                                    for trajectory in trajectories]
            plot_gp(observed_pred,
                    trajectories_to_plot,
                    means=(test_t.numpy(), observed_pred.mean.numpy()),
                    legend=legend,
                    xlim=[0, 6.0],
                    ylim=[-40, 15],
                    xlabel="Time",
                    ylabel="X-position",
                    image_name="x_time_GP.png")

            trajectories_to_plot = [(trajectory.t[0], trajectory.pos[1, :])
                                    for trajectory in trajectories]
            plot_gp(observed_pred1,
                    trajectories_to_plot,
                    means=(test_t.numpy(), observed_pred1.mean.numpy()),
                    legend=legend,
                    xlim=[0, 6.0],
                    ylim=[-25, 30],
                    xlabel="Time",
                    ylabel="Y-position",
                    image_name="y_time_GP.png")

            trajectories_to_plot = [(trajectory.pos[0, :],
                                     trajectory.pos[1, :])
                                    for trajectory in trajectories]
            plot_gp(observed_pred1,
                    trajectories_to_plot,
                    means=(observed_pred.mean.numpy(),
                           observed_pred1.mean.numpy()),
                    legend=legend,
                    xlim=[-40, 15],
                    ylim=[-25, 30],
                    xlabel="X-position",
                    ylabel="Y-position",
                    plot_intervals=False)

            legend = (
                'Observed Demo 1',
                'Observed Demo 2',
                'Observed Demo 3',
                'Observed Demo 4',
                'Observed Demo 5',
                'Observed Demo 6',
                'Observed Demo 7',
                'Predicted trajectory',
            )
            plot_3d_traj(trajectories,
                         test_t,
                         observed_preds=(observed_pred, observed_pred1),
                         legend=legend)


class TestMultitaskGP(unittest.TestCase):
    """Unit tests for multi-output Gaussian Processes."""

    def get_data(self):
        """Get the trajectories and training data."""
        trajectories = load_trajectories()

        # Concatenating the demos
        train_t, train_xy = concatenate_trajectories(trajectories)

        # Time
        train_t = torch.tensor(train_t).float()
        # Output
        train_xy = torch.tensor(train_xy).float()

        return trajectories, train_t, train_xy

    @unittest.skip("Exact multi-output inference is too slow, O(n^3)")
    def test_multi_output_exact_gp(self):
        """Unit test for exact inference on multi-output GP"""
        trajectories, train_t, train_xy = self.get_data()

        m = MultitaskExactGP(train_t, train_xy, num_tasks=2)
        model = m.get_model()

        training_iterations = 0
        m.training(train_t, train_xy, training_iterations)
        likelihood = m.evaluation()

        # Set into eval mode
        model.eval()
        likelihood.eval()

        # Make predictions
        with torch.no_grad(), \
            gpytorch.settings.fast_pred_var(), \
                gpytorch.settings.fast_computations():
            test_t = torch.linspace(0, 6, 1000).float()
            predictions = likelihood(model(test_t))
            mean = predictions.mean
            lower, upper = predictions.confidence_region()

        print('Mean')
        print(mean.shape)
        print(mean[0:10, :])
        print('Lower values:')
        print(lower.shape)
        print(lower[0:10, :])
        print('Upper values:')
        print(upper.shape)
        print(lower[0:10, :])

        self.plot(trajectories, test_t, mean, lower, upper)

    def test_multi_output_variational_gp(self):
        """Unit test for variational inference on multi-output GP"""
        _, train_t, train_xy = self.get_data()

        torch.random.manual_seed(1107)

        num_tasks = 2
        m = MultitaskApproximateGP(num_tasks=num_tasks, num_latents=10)

        m.training(train_t, train_xy, training_iterations=3)

        model = m.get_model()
        likelihood = m.evaluation()

        # Set into eval mode
        model.eval()
        likelihood.eval()

        # Make predictions
        with torch.no_grad(), \
            gpytorch.settings.fast_pred_var(), \
                gpytorch.settings.fast_computations():
            test_t = torch.linspace(0, 6, 1000).float()
            predictions = likelihood(model(test_t))
            mean = predictions.mean
            lower, upper = predictions.confidence_region()

        # regression tests for the mean trajectory
        torch.testing.assert_close(
            mean[0],
            torch.tensor([-26.878214315, 11.697209545]).double())
        torch.testing.assert_close(
            mean[500],
            torch.tensor([-22.503201496, -6.546301216]).double())
        torch.testing.assert_close(
            mean[-1],
            torch.tensor([-4.958472397, -4.592620590]).double())

        torch.testing.assert_close(
            lower[0],
            torch.tensor([-31.121183753, 6.925952889]).double())
        torch.testing.assert_close(
            upper[0],
            torch.tensor([-22.635244877, 16.468466201]).double())
