"""Unit tests for Gaussian Process Regression on Trajectories"""

import unittest

import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import torch
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import ExactGP
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import axes3d

from steal.datasets import lasa


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


class ExactGPModel(ExactGP):
    """We will use the simplest form of GP model, exact inference"""

    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class model_GP:
    # define a simple Exact GP model
    def __init__(self, X, y) -> None:

        # initialize likelihood and model
        self.likelihood = GaussianLikelihood()
        self.model = ExactGPModel(X, y, self.likelihood)

        # Find optimal model hyperparameters
        self.model.train()
        self.likelihood.train()

        # Use the adam optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=0.1)  # Includes GaussianLikelihood parameters

        # "Loss" for GPs - the marginal log likelihood
        self.mll = ExactMarginalLogLikelihood(self.likelihood, self.model)
        self.model.double()

    def get_model(self):
        return self.model

    def get_optimizer(self):
        return self.optimizer

    def get_likelihood(self):
        return self.likelihood

    # train the GP
    def training(self, train_input, train_output, training_iter):

        for i in range(training_iter):
            # Zero gradients from previous iteration
            self.optimizer.zero_grad()
            # Output from model
            output = self.model(train_input)

            # Calc loss and backprop gradients
            loss = -self.mll(output, train_output)
            loss.backward()
            print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' %
                  (i + 1, training_iter, loss.item(),
                   self.model.covar_module.base_kernel.lengthscale.item(),
                   self.model.likelihood.noise.item()))
            self.optimizer.step()

    # GP evaluation
    def evaluation(self):
        """Return the likelihood of the data."""
        self.model.eval()
        self.likelihood.eval()
        return self.likelihood


class TestGaussianProcess(unittest.TestCase):

    def test_load_trajectories(self):
        trajectories = load_trajectories()
        assert len(trajectories) == 7

        train_t, train_xy = concatenate_trajectories(trajectories)
        assert train_t.shape == (7000, )
        assert train_xy.shape == (7000, 2)

    def test_gp(self):
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
        m1 = model_GP(train_t, train_x)
        model = m1.get_model()
        m1.training(train_t, train_x, training_iters)
        likelihood = m1.evaluation()

        # Test points are regularly spaced along [0,6]
        # Make predictions by feeding model through likelihood
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            test_t = torch.linspace(0, 6, 1000).double()
            observed_pred = likelihood(model(test_t))

        # GP Model for time vs Y position
        m2 = model_GP(train_t, train_y)
        model1 = m2.get_model()
        m2.training(train_t, train_y, training_iters)
        likelihood1 = m2.evaluation()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            test_t = torch.linspace(0, 6, 1000).double()
            observed_pred1 = likelihood1(model1(test_t))

        # Plot graphs
        with torch.no_grad():
            # Initialize plot
            f, ax = plt.subplots(1, 1, figsize=(4, 3))

            # Get upper and lower confidence bounds
            lower, upper = observed_pred.confidence_region()
            # Plot training data as black stars
            for j, trajectory in enumerate(trajectories):
                train_t = trajectory.t[0]
                train_x = trajectory.pos[0, :]
                ax.plot(train_t, train_x, '--')

            # Plot predictive means as blue line
            ax.plot(test_t.numpy(), observed_pred.mean.numpy(), 'b')
            # Shade between the lower and upper confidence bounds
            ax.fill_between(test_t.numpy(),
                            lower.numpy(),
                            upper.numpy(),
                            alpha=0.5)
            ax.set_xlim([0, 6.0])
            # ax.set_ylim([-25, 30])
            ax.set_ylim([-40, 15])
            ax.legend([
                'Observed Demo 1', 'Observed Demo 2', 'Observed Demo 3',
                'Observed Demo 4', 'Observed Demo 5', 'Observed Demo 6',
                'Observed Demo 7', 'Mean', 'Confidence'
            ])
            ax.set_xlabel('Time')
            ax.set_ylabel('X-position')
            plt.savefig('x_time_GP.png')
            plt.show()

        with torch.no_grad():
            # Initialize plot
            f, ax = plt.subplots(1, 1, figsize=(4, 3))

            # Get upper and lower confidence bounds
            lower, upper = observed_pred1.confidence_region()
            # Plot training data as black stars
            for j, trajectory in enumerate(trajectories):
                train_t = trajectory.t[0]
                train_y = trajectory.pos[1, :]
                ax.plot(train_t, train_y, '--')

            # Plot predictive means as blue line
            ax.plot(test_t, observed_pred1.mean.numpy(), 'b')
            # Shade between the lower and upper confidence bounds
            ax.fill_between(test_t,
                            lower.numpy(),
                            upper.numpy(),
                            alpha=0.5)
            ax.set_xlim([0, 6.0])
            ax.set_ylim([-25, 30])
            ax.legend([
                'Observed Demo 1', 'Observed Demo 2', 'Observed Demo 3',
                'Observed Demo 4', 'Observed Demo 5', 'Observed Demo 6',
                'Observed Demo 7', 'Mean', 'Confidence'
            ])
            ax.set_xlabel('Time')
            ax.set_ylabel('Y-position')
            plt.savefig('y_time_GP.png')
            plt.show()

        with torch.no_grad():
            # Initialize plot
            f, ax = plt.subplots(1, 1, figsize=(4, 3))

            # Plot training data as black stars
            for j, trajectory in enumerate(trajectories):
                train_x = trajectory.pos[0, :]
                train_y = trajectory.pos[1, :]
                ax.plot(train_x, train_y, '--')

            # Plot predictive means as blue line
            ax.plot(observed_pred.mean.numpy(), observed_pred1.mean.numpy(),
                    'b')
            # Shade between the lower and upper confidence bounds
            # ax.fill_between(test_t.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
            ax.set_xlim([-40, 15])
            ax.set_ylim([-25, 30])
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
            ax.set_xlabel('X-position')
            ax.set_ylabel('Y-position')
            plt.show()

        with torch.no_grad():

            fig = plt.figure(figsize=(4, 4))
            ax = fig.add_subplot(111, projection='3d')

            # Plot training data as black stars
            for trajectory in trajectories:
                train_x = trajectory.pos[0, :]
                train_y = trajectory.pos[1, :]
                ax.plot(train_y, train_x, test_t, '--')

            # Plot predictive means as blue line
            ax.plot(observed_pred1.mean.numpy(), observed_pred.mean.numpy(),
                    test_t, 'b')
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
            plt.show()
