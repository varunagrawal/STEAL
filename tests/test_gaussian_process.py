import unittest
import matplotlib.pyplot as plt
import gpytorch
import numpy as np
import numpy.testing as npt
import pytest
import torch
import pickle as pkl
import os
from pathlib import Path
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.models import ExactGP

torch.manual_seed(1813)

def load_trajectory():
    dir_path = Path(os.getcwd()).parent
    data_path = os.path.join(dir_path,'data/LASADataset_traj_pkl/heee.pickle')
    with open(data_path, "rb") as input_file:
        dataset = pkl.load(input_file)

    return dataset

def train_x_data(trajectory):
    """Training x data is time"""
    return torch.tensor(trajectory[0][1][0])


def train_y_data(trajectory):
    """Training output data is y position"""
    return torch.tensor(trajectory[0][0][1,:])


class ExactGPModel(gpytorch.models.ExactGP):
    """We will use the simplest form of GP model, exact inference"""
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class TestGaussianProcess(unittest.TestCase):

    def test_gp(self):
        
        trajectory = load_trajectory()
        assert len(trajectory) == 7
        for i in range(len(trajectory)):
            assert len(trajectory[i]) == 4
            assert trajectory[i][0].shape == (2,1000)
            assert trajectory[i][1].shape == (1,1000)
            assert trajectory[i][2].shape == (2,1000)
            assert trajectory[i][3].shape == (2,1000)
        
        train_x = train_x_data(trajectory)
        print(train_x.dtype)
        
        assert len(train_x) == 1000

        train_y = train_y_data(trajectory)
        print(train_x.dtype)

        assert len(train_y) == 1000
        
        # visualize the data
        
        f, ax = plt.subplots(1, 1, figsize=(4, 3))

        # Plot training data as black stars
        ax.plot(train_x.numpy(), train_y.numpy(), 'black')
        ax.set_xlim([0, 4.5])
        ax.set_ylim([-20, 25])
        ax.legend(['Observed Data'])
        plt.show()

        # initialize likelihood and model
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPModel(train_x, train_y, likelihood)

        # Find optimal model hyperparameters
        model.train()
        likelihood.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.1)  # Includes GaussianLikelihood parameters

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        training_iter = 22

        model = model.double()

        for i in range(training_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = model(train_x)
            # Calc loss and backprop gradients
            loss = -mll(output, train_y)
            loss.backward()
            print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' %
                  (i + 1, training_iter, loss.item(),
                   model.covar_module.base_kernel.lengthscale.item(),
                   model.likelihood.noise.item()))
            optimizer.step()

        # Get into evaluation (predictive posterior) mode
        model.eval()
        likelihood.eval()

        # Test points are regularly spaced along [0,1]
        # Make predictions by feeding model through likelihood
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            test_x = torch.linspace(0, 3.91, 178).double()
            print("##")
            print(test_x.dtype)
            observed_pred = likelihood(model(test_x))
        
        with torch.no_grad():
            # Initialize plot
            f, ax = plt.subplots(1, 1, figsize=(4, 3))

            # Get upper and lower confidence bounds
            lower, upper = observed_pred.confidence_region()
            # Plot training data as black stars
            ax.plot(train_x.numpy(), train_y.numpy(), 'k*')
            # Plot predictive means as blue line
            ax.plot(test_x.numpy(), observed_pred.mean.numpy(), 'b')
            # Shade between the lower and upper confidence bounds
            ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
            ax.set_xlim([0, 4.5])
            ax.set_ylim([-20, 25])
            ax.legend(['Observed Data', 'Mean', 'Confidence'])
            plt.show()

        # expected_mean = torch.tensor([
        #     0.0878, 0.1926, 0.3001, 0.4078, 0.5133, 0.6143, 0.7085, 0.7938,
        #     0.8682, 0.9302, 0.9784, 1.0118, 1.0298, 1.0320, 1.0183, 0.9892,
        #     0.9451, 0.8868, 0.8153, 0.7317, 0.6373, 0.5335, 0.4217, 0.3033,
        #     0.1801, 0.0536, -0.0744, -0.2022, -0.3279, -0.4494, -0.5650,
        #     -0.6723, -0.7695, -0.8545, -0.9253, -0.9803, -1.0180, -1.0373,
        #     -1.0373, -1.0179, -0.9792, -0.9221, -0.8480, -0.7586, -0.6563,
        #     -0.5439, -0.4242, -0.3006, -0.1762, -0.0542, 0.0624
        # ])
        
        # npt.assert_allclose(observed_pred.mean.numpy(), expected_mean.numpy(), 1e-3)