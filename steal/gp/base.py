"""Module for simple scalar-valued, exact GPs."""

#pylint: disable=arguments-differ

import torch
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import ExactGP
from torch.optim import Adam


class ExactGPModel(ExactGP):
    """We will use the simplest form of GP model, exact inference"""

    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel())

    def forward(self, x):
        """Forward pass"""
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class ModelGP:
    """Define a simple Exact GP model"""

    def __init__(self, X, y) -> None:

        # initialize likelihood and model
        self.likelihood = GaussianLikelihood()
        self.model = ExactGPModel(X, y, self.likelihood)

        # Find optimal model hyperparameters
        self.model.train()
        self.likelihood.train()

        # Use the adam optimizer
        # Includes GaussianLikelihood parameters
        self.optimizer = Adam(self.model.parameters(), lr=0.1)

        # "Loss" for GPs - the marginal log likelihood
        self.mll = ExactMarginalLogLikelihood(self.likelihood, self.model)
        self.model.double()

    def get_model(self):
        """Return the GP model"""
        return self.model

    def get_optimizer(self):
        """Return the optimizer"""
        return self.optimizer

    def get_likelihood(self):
        """Return the likelihood function."""
        return self.likelihood

    # train the GP
    def training(self, train_input, train_output, training_iters):
        """Run Type II MLE to get the best prior hyperparameters."""

        for i in range(training_iters):
            # Zero gradients from previous iteration
            self.optimizer.zero_grad()
            # Output from model
            output = self.model(train_input)

            # Calc loss and backprop gradients
            loss = -self.mll(output, train_output)
            loss.backward()
            print(
                f'Iter {i+1}/{training_iters} - Loss: {loss.item():.3f}' \
                f'   lengthscale: {self.model.covar_module.base_kernel.lengthscale.item():.3f}'\
                f'   noise: {self.model.likelihood.noise.item():.3f}'
            )

            self.optimizer.step()

    # GP evaluation
    def evaluation(self):
        """Return the likelihood of the data."""
        self.model.eval()
        self.likelihood.eval()
        return self.likelihood
