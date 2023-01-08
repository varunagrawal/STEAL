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

from steal.datasets.lasa import Lasa
from steal.gp import (MultitaskApproximateGaussianProcess,
                      MultitaskExactGaussianProcess, ScalarGaussianProcess)
from steal.utils.plotting.gp import plot_3d_traj, plot_gp

# set default printing precision
torch.set_printoptions(precision=9)


class TestGaussianProcess(unittest.TestCase):
    """Tests for a scalar valued Gaussian Process on the LASA dataset."""

    def test_gp(self):
        """Test training of a scalar-valued GP."""
        lasa = Lasa(shape="heee")

        # Concatenating the demos
        train_t, train_xy = lasa.concatenated_trajectories()
        train_x = train_xy[:, 0]
        train_y = train_xy[:, 1]

        # Time
        train_t = torch.tensor(train_t)
        # X Position
        train_x = torch.tensor(train_x)
        # Y Position
        train_y = torch.tensor(train_y)

        # GP Model for time vs X position
        # initialize GP
        m1 = ScalarGaussianProcess(train_t, train_x)
        m1.train(train_t, train_x, training_iterations=1)

        # Test points are regularly spaced along [0,6]
        # Make predictions by feeding model through likelihood
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            test_t = torch.linspace(0, 6, 1000).double()
            observed_pred = m1.evaluate(test_t)

        # GP Model for time vs Y position
        m2 = ScalarGaussianProcess(train_t, train_y)
        m2.train(train_t, train_y, training_iterations=1)

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            test_t = torch.linspace(0, 6, 1000).double()
            observed_pred1 = m2.evaluate(test_t)


class TestMultitaskGP(unittest.TestCase):
    """Unit tests for multi-output Gaussian Processes."""

    def get_data(self):
        """Get the trajectories and training data."""
        lasa = Lasa(shape="heee")

        # Concatenating the demos
        train_t, train_xy = lasa.concatenated_trajectories()

        # Time
        train_t = torch.tensor(train_t).float()
        # Output
        train_xy = torch.tensor(train_xy).float()

        return train_t, train_xy

    def test_multi_output_exact_gp(self):
        """Unit test for exact inference on multi-output GP"""
        torch.manual_seed(7648)

        train_t, train_xy = self.get_data()

        # subsample for efficiency
        train_t = train_t[:1000]
        train_xy = train_xy[:1000]

        m = MultitaskExactGaussianProcess(train_t, train_xy, num_tasks=2)

        m.train(train_t, train_xy, training_iterations=1)

        # Make predictions
        with torch.no_grad(), \
            gpytorch.settings.fast_pred_var(), \
                gpytorch.settings.fast_computations():
            test_t = torch.linspace(0, 6, 1000).float()
            predictions = m.evaluate(test_t)
            mean = predictions.mean
            lower, upper = predictions.confidence_region()

        assert mean.shape == (1000, 2)
        torch.testing.assert_close(
            mean[0],
            torch.tensor([-30.514565338, 15.873690225]).double())
        torch.testing.assert_close(
            lower[0],
            torch.tensor([-31.367863748, 13.964214544]).double())
        torch.testing.assert_close(
            upper[0],
            torch.tensor([-29.661266929, 17.783165906]).double())

    def test_multi_output_variational_gp(self):
        """Unit test for variational inference on multi-output GP"""
        train_t, train_xy = self.get_data()

        torch.random.manual_seed(1107)

        num_tasks = 2
        m = MultitaskApproximateGaussianProcess(num_tasks=num_tasks,
                                                num_latents=10)

        m.train(train_t, train_xy, training_iterations=3)

        # Make predictions
        with torch.no_grad(), \
            gpytorch.settings.fast_pred_var(), \
                gpytorch.settings.fast_computations():
            test_t = torch.linspace(0, 6, 1000).float()
            predictions = m.evaluate(test_t)
            mean = predictions.mean
            lower, upper = predictions.confidence_region()

        # regression tests for the mean trajectory
        torch.testing.assert_close(mean[0],
                                   torch.tensor([-26.878214315,
                                                 11.697209545]).double(),
                                   atol=1e-6,
                                   rtol=1e-6)
        torch.testing.assert_close(mean[500],
                                   torch.tensor([-22.503201496,
                                                 -6.546301216]).double(),
                                   atol=1e-6,
                                   rtol=1e-6)
        torch.testing.assert_close(mean[-1],
                                   torch.tensor([-4.958472397,
                                                 -4.592620590]).double(),
                                   atol=1e-6,
                                   rtol=1e-6)

        torch.testing.assert_close(lower[0],
                                   torch.tensor([-31.121183753,
                                                 6.925952889]).double(),
                                   atol=1e-6,
                                   rtol=1e-6)
        torch.testing.assert_close(upper[0],
                                   torch.tensor([-22.635244877,
                                                 16.468466201]).double(),
                                   atol=1e-6,
                                   rtol=1e-6)
