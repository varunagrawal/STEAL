import unittest
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import axes3d
import gpytorch
import numpy as np
import numpy.testing as npt
import pytest
import torch
import pickle as pkl
import os
import pyLasaDataset as lasa
from pathlib import Path
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.models import ExactGP

torch.manual_seed(1813)

def load_trajectory():
    return lasa.DataSet.heee.demos

def train_x_data(trajectory, i):
    """Training x data is time"""
    return torch.tensor(trajectory[i].t[0])

def train_y_data1(trajectory, i):
    """Training output data is x position"""
    return torch.tensor(trajectory[i].pos[0,:])

def train_y_data2(trajectory, i):
    """Training output data is y position"""
    return torch.tensor(trajectory[i].pos[1,:])

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

        # for i in range(len(trajectory)):
        #     assert trajectory[i].pos.shape == (2,1000)
        #     assert trajectory[i].t.shape == (1,1000)
        #     assert trajectory[i].vel.shape == (2,1000)
        #     assert trajectory[i].acc.shape == (2,1000)
        
        #     train_x = train_x_data(trajectory, i)
        #     print(train_x.dtype)
            
        #     assert len(train_x) == 1000

        #     train_y1 = train_y_data1(trajectory, i)
        #     print(train_y1.dtype)

        #     assert len(train_y1) == 1000
            
        #     # visualize the data
        #     plt.plot(train_x.numpy(), train_y1.numpy(), label='Observed Data '+str(i))

        # plt.legend()
        # plt.show()

        train_x = np.zeros((7000))
        train_y1 = np.zeros((7000))
        train_y2 = np.zeros((7000))
        for i in range(len(trajectory)):
            train_x[i*1000:1000*(i+1)] = train_x_data(trajectory, i)
            train_y1[i*1000:1000*(i+1)] = train_y_data1(trajectory, i)
            train_y2[i*1000:1000*(i+1)] = train_y_data2(trajectory, i)
        
        
        train_x = torch.tensor(train_x)
        train_y1 = torch.tensor(train_y1)
        train_y2 = torch.tensor(train_y2)

        # initialize likelihood and model
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPModel(train_x, train_y1, likelihood)
        
        likelihood1 = gpytorch.likelihoods.GaussianLikelihood()
        model1 = ExactGPModel(train_x, train_y2, likelihood)


        # op = model(train_x)
        # prior_mean = op.mean
        # print(prior_mean.shape)
        # plt.plot(prior_mean.detach().numpy(), label='Mean')

        # prior_pred = likelihood(op)
        # lower, upper = prior_pred.confidence_region()
        # print(lower)
        # print(upper)
        # plt.fill_between(np.linspace(0,7000,1), lower.detach().numpy(), upper.detach().numpy(), alpha=0.5)

        # plt.legend()
        # plt.show()

        # prior_covar = op.covariance_matrix
        # print(prior_covar.shape)

        # Find optimal model hyperparameters
        model.train()
        likelihood.train()

        model1.train()
        likelihood1.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.1)  # Includes GaussianLikelihood parameters

        optimizer1 = torch.optim.Adam(
            model1.parameters(),
            lr=0.1)  # Includes GaussianLikelihood parameters

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        mll1 = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood1, model1)

        training_iter = 22

        model = model.double()

        model1 = model1.double()

        for i in range(training_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = model(train_x)

            # Calc loss and backprop gradients
            loss = -mll(output, train_y1)
            loss.backward()
            print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' %
                (i + 1, training_iter, loss.item(),
                model.covar_module.base_kernel.lengthscale.item(),
                model.likelihood.noise.item()))
            optimizer.step()
        
        for i in range(training_iter):
            # Zero gradients from previous iteration
            optimizer1.zero_grad()
            # Output from model
            output1 = model1(train_x)

            # Calc loss and backprop gradients
            loss1 = -mll1(output1, train_y2)
            loss1.backward()
            print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' %
                (i + 1, training_iter, loss1.item(),
                model1.covar_module.base_kernel.lengthscale.item(),
                model1.likelihood.noise.item()))
            optimizer1.step()

        # Get into evaluation (predictive posterior) mode
        model.eval()
        likelihood.eval()

        model1.eval()
        likelihood1.eval()

        # Test points are regularly spaced along [0,1]
        # Make predictions by feeding model through likelihood
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            test_x = torch.linspace(0, 6, 1000).double()
            observed_pred = likelihood(model(test_x))

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            test_x = torch.linspace(0, 6, 1000).double()
            observed_pred1 = likelihood1(model1(test_x))
        
        with torch.no_grad():
            # Initialize plot
            f, ax = plt.subplots(1, 1, figsize=(4, 3))

            # Get upper and lower confidence bounds
            lower, upper = observed_pred.confidence_region()
            # Plot training data as black stars
            for j in range(len(trajectory)):
                train_x = train_x_data(trajectory, j)
                train_y1 = train_y_data1(trajectory, j)
                ax.plot(train_x.numpy(), train_y1.numpy(), '--')

            # Plot predictive means as blue line
            ax.plot(test_x.numpy(), observed_pred.mean.numpy(), 'b')
            # Shade between the lower and upper confidence bounds
            ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
            ax.set_xlim([0, 6.0])
            #ax.set_ylim([-25, 30])
            ax.set_ylim([-40, 15])
            ax.legend(['Observed Demo 1','Observed Demo 2','Observed Demo 3',
            'Observed Demo 4','Observed Demo 5','Observed Demo 6',
            'Observed Demo 7', 'Mean', 'Confidence'])
            ax.set_xlabel('Time')
            ax.set_ylabel('X-position')
            plt.show()
        
        with torch.no_grad():
            # Initialize plot
            f, ax = plt.subplots(1, 1, figsize=(4, 3))

            # Get upper and lower confidence bounds
            lower, upper = observed_pred1.confidence_region()
            # Plot training data as black stars
            for j in range(len(trajectory)):
                train_x = train_x_data(trajectory, j)
                train_y2 = train_y_data2(trajectory, j)
                ax.plot(train_x.numpy(), train_y2.numpy(), '--')

            # Plot predictive means as blue line
            ax.plot(test_x.numpy(), observed_pred1.mean.numpy(), 'b')
            # Shade between the lower and upper confidence bounds
            ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
            ax.set_xlim([0, 6.0])
            ax.set_ylim([-25, 30])
            ax.legend(['Observed Demo 1','Observed Demo 2','Observed Demo 3',
            'Observed Demo 4','Observed Demo 5','Observed Demo 6',
            'Observed Demo 7', 'Mean', 'Confidence'])
            ax.set_xlabel('Time')
            ax.set_ylabel('Y-position')
            plt.show()
        
        with torch.no_grad():
            # Initialize plot
            f, ax = plt.subplots(1, 1, figsize=(4, 3))

            # Plot training data as black stars
            for j in range(len(trajectory)):
                train_y1 = train_y_data1(trajectory, j)
                train_y2 = train_y_data2(trajectory, j)
                ax.plot(train_y1.numpy(), train_y2.numpy(), '--')

            # Plot predictive means as blue line
            ax.plot(observed_pred.mean.numpy(), observed_pred1.mean.numpy(), 'b')
            # Shade between the lower and upper confidence bounds
            # ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
            ax.set_xlim([-40, 15])
            ax.set_ylim([-25, 30])
            ax.legend(['Observed Demo 1','Observed Demo 2','Observed Demo 3',
            'Observed Demo 4','Observed Demo 5','Observed Demo 6',
            'Observed Demo 7', 'Predicted trajectory',])
            ax.set_xlabel('X-position')
            ax.set_ylabel('Y-position')
            plt.show()

        with torch.no_grad():

            fig = plt.figure(figsize=(4,4))
            ax = fig.add_subplot(111, projection='3d')

            # Plot training data as black stars
            for j in range(len(trajectory)):
                train_y1 = train_y_data1(trajectory, j)
                train_y2 = train_y_data2(trajectory, j)
                #ax.plot(train_y1.numpy(), train_y2.numpy(), '--')
                ax.plot(train_y2.numpy(), train_y1.numpy(), test_x, '--')

            # Plot predictive means as blue line
            ax.plot(observed_pred1.mean.numpy(), observed_pred.mean.numpy(), test_x, 'b')
            
            ax.legend(['Observed Demo 1','Observed Demo 2','Observed Demo 3',
            'Observed Demo 4','Observed Demo 5','Observed Demo 6',
            'Observed Demo 7', 'Predicted trajectory',])
            ax.set_xlabel('Y-position')
            ax.set_ylabel('X-position')
            ax.set_zlabel('Time')
            plt.show()
