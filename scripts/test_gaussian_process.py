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
    # Training output data is position
    # isX -> true will return x otherwise y position
    if isX == True:
        return torch.tensor(trajectory[i].pos[0,:])
    else:
        return torch.tensor(trajectory[i].pos[1,:])    

# def train_y_data2(trajectory, i):
#     """Training output data is y position"""
#     return torch.tensor(trajectory[i].pos[1,:])

def conc_trajectories(trajectory, i):
    # Concatenate the input into a single array
    train = np.zeros((7000))
    for i in range(len(trajectory)):
        train[i*1000:1000*(i+1)] = train_x_data(trajectory, i)
    return train

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

class model_GP:
    # define a simple Exact GP model
    def __init__(self, input, output) -> None:

        # initialize likelihood and model
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.model = ExactGPModel(input, output, self.likelihood)

        # Find optimal model hyperparameters
        self.model.train()
        self.likelihood.train() 

        # Use the adam optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=0.1)  # Includes GaussianLikelihood parameters
        
        # "Loss" for GPs - the marginal log likelihood
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
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
        self.model.eval()
        self.likelihood.eval()
        return self.likelihood

class TestGaussianProcess(unittest.TestCase):

    def test_gp(self):
        
        trajectory = load_trajectory()
        assert len(trajectory) == 7

        # Concatenating the demos
        train_x = np.zeros((7000)) 
        train_y1 = np.zeros((7000))
        train_y2 = np.zeros((7000))

        for i in range(len(trajectory)):
            train_x[i*1000:1000*(i+1)] = train_x_data(trajectory, i)
            train_y1[i*1000:1000*(i+1)] = train_y_data(trajectory, i)
            train_y2[i*1000:1000*(i+1)] = train_y_data(trajectory, i, isX = False)
        
        # Time 
        train_x = torch.tensor(train_x)
        # X Position
        train_y1 = torch.tensor(train_y1)
        # Y Position
        train_y2 = torch.tensor(train_y2)

        training_iter = 22

        # GP Model for time vs X position
        # initialize GP
        m1 = model_GP(train_x, train_y1)
        model = m1.get_model()
        m1.training(train_x, train_y1, training_iter)
        likelihood = m1.evaluation()

        # Test points are regularly spaced along [0,6]
        # Make predictions by feeding model through likelihood
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            test_x = torch.linspace(0, 6, 1000).double()
            observed_pred = likelihood(model(test_x))

        # GP Model for time vs Y position
        m2 = model_GP(train_x, train_y2)
        model1 = m2.get_model()
        m2.training(train_x, train_y1, training_iter)
        likelihood1 = m2.evaluation()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            test_x = torch.linspace(0, 6, 1000).double()
            observed_pred1 = likelihood1(model1(test_x))

        # Plot graphs 
        with torch.no_grad():
            # Initialize plot
            f, ax = plt.subplots(1, 1, figsize=(4, 3))

            # Get upper and lower confidence bounds
            lower, upper = observed_pred.confidence_region()
            # Plot training data as black stars
            for j in range(len(trajectory)):
                train_x = train_x_data(trajectory, j)
                train_y1 = train_y_data(trajectory, j)
                ax.plot(train_x.numpy(), train_y1.numpy(), '--')

            # Plot predictive means as blue line
            ax.plot(test_x.numpy(), observed_pred.mean.numpy(), 'b')
            # Shade between the lower and upper confidence bounds
            ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
            ax.set_xlim([0, 6.0])
            # ax.set_ylim([-25, 30])
            ax.set_ylim([-40, 15])
            ax.legend(['Observed Demo 1','Observed Demo 2','Observed Demo 3',
            'Observed Demo 4','Observed Demo 5','Observed Demo 6',
            'Observed Demo 7', 'Mean', 'Confidence'])
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
            for j in range(len(trajectory)):
                train_x = train_x_data(trajectory, j)
                train_y2 = train_y_data(trajectory, j, isX = False)
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
            plt.savefig('y_time_GP.png')
            plt.show()
        
        with torch.no_grad():
            # Initialize plot
            f, ax = plt.subplots(1, 1, figsize=(4, 3))

            # Plot training data as black stars
            for j in range(len(trajectory)):
                train_y1 = train_y_data(trajectory, j)
                train_y2 = train_y_data(trajectory, j, isX = False)
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
                train_y1 = train_y_data(trajectory, j)
                train_y2 = train_y_data(trajectory, j, isX = False)
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
