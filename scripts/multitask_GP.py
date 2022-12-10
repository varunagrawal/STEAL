import unittest
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import axes3d
import numpy as np
import numpy.testing as npt
import pytest
import torch
import pickle as pkl
import os
import pyLasaDataset as lasa
from pathlib import Path
import gpytorch
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.models import ExactGP

torch.manual_seed(1813)

def load_trajectory():
    return lasa.DataSet.heee.demos

def train_x_data(trajectory, i):
    '''Training x data is time'''
    return torch.tensor(trajectory[i].t[0])

def train_y_data(trajectory, i, isX = True):
        return torch.tensor((trajectory[i].pos).T)

class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=2
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(), num_tasks=2, rank=1
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


class TestMultTaskGP(unittest.TestCase):

    def test_mtgp(self):
        
        trajectory = load_trajectory()
        assert len(trajectory) == 7

        # Concatenating the demos
        train_x = np.zeros((5000))  
        train_y = np.zeros((5000,2))

        n = 5 #len(trajectory)

        for i in range(n):#len(trajectory)):
            train_x[i*1000:1000*(i+1)] = train_x_data(trajectory, i)
            train_y[i*1000:1000*(i+1),:] = train_y_data(trajectory, i)
        
        print(train_y[0:10, :])
        assert train_y.shape == (5000,2)

        # Time 
        train_x = torch.tensor(train_x)
        # Output
        train_y = torch.tensor(train_y)
        

        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)
        model = MultitaskGPModel(train_x, train_y, likelihood)

        # Find optimal model hyperparameters
        model.train()
        likelihood.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        training_iterations = 2

        model.double()

        for i in range(training_iterations):
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
            optimizer.step()
        
        # Set into eval mode
        model.eval()
        likelihood.eval()

        # Initialize plots
        f, (y1_ax, y2_ax) = plt.subplots(1, 2, figsize=(8, 3))

        # Make predictions
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            test_x = torch.linspace(0, 6, 1000).double()
            predictions = likelihood(model(test_x))
            mean = predictions.mean
            lower, upper = predictions.confidence_region()

        print('Mean')
        print(mean.shape)
        print(mean[0:10,:])
        print('Lower values:')
        print(lower.shape)
        print(lower[0:10,:])
        print('Upper values:')
        print(upper.shape)
        print(lower[0:10,:])


        # This contains predictions for both tasks, flattened out
        # The first half of the predictions is for the first task
        # The second half is for the second task

        # Plot training data as black stars
        # y1_ax.plot(train_x.detach().numpy(), train_y[:, 0].detach().numpy(), '--')
        # Plot training data as black stars
        for j in range(n):
            train_x = train_x_data(trajectory, j)
            train_y = train_y_data(trajectory, j)
            y1_ax.plot(train_x.numpy(), train_y[:,0].numpy(), '--')

        # Predictive mean as blue line
        y1_ax.plot(test_x.numpy(), mean[:, 0].numpy(), 'b')
        # Shade in confidence
        y1_ax.fill_between(test_x.numpy(), lower[:, 0].numpy(), upper[:, 0].numpy(), alpha=0.5)
        y1_ax.set_ylim([-40, 15])
        y1_ax.legend(['Observed Demo 1','Observed Demo 2','Observed Demo 3',
            'Observed Demo 4','Observed Demo 5','Observed Demo 6',
            'Observed Demo 7', 'Mean', 'Confidence'])
        y1_ax.set_title('Observed Values (Likelihood)')
        # plt.savefig('x_time.png')
        # plt.show()

        # Plot training data as black stars
        # y2_ax.plot(train_x.detach().numpy(), train_y[:, 1].detach().numpy(), '--')
        for j in range(n):
            train_x = train_x_data(trajectory, j)
            train_y = train_y_data(trajectory, j)
            y2_ax.plot(train_x.numpy(), train_y[:,1].numpy(), '--')

        # Predictive mean as blue line
        y2_ax.plot(test_x.numpy(), mean[:, 1].numpy(), 'b')
        # Shade in confidence
        y2_ax.fill_between(test_x.numpy(), lower[:, 1].numpy(), upper[:, 1].numpy(), alpha=0.5)
        y2_ax.set_ylim([-25, 30])
        y2_ax.legend(['Observed Demo 1','Observed Demo 2','Observed Demo 3',
            'Observed Demo 4','Observed Demo 5','Observed Demo 6',
            'Observed Demo 7', 'Mean', 'Confidence'])
        y2_ax.set_title('Observed Values (Likelihood)')
        plt.savefig('input_output.png')
        plt.show()