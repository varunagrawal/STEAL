"""Module for multi-task/vector-valued Gaussian Processes"""

#pylint: disable=arguments-differ

import numpy as np
import torch
from gpytorch.distributions import MultitaskMultivariateNormal
from gpytorch.kernels import MultitaskKernel, RBFKernel
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from gpytorch.means import ConstantMean, MultitaskMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import ApproximateGP, ExactGP
from gpytorch.variational import (CholeskyVariationalDistribution,
                                  VariationalStrategy)
from torch.optim import Adam


class MultitaskExactGPModel(ExactGP):
    """
    A vector-valued Gaussian Process to learn interactions between output values.
    """

    def __init__(self, train_x, train_y, likelihood, num_tasks=2, rank=1):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = MultitaskMean(ConstantMean(), num_tasks=num_tasks)
        self.covar_module = MultitaskKernel(RBFKernel(),
                                            num_tasks=num_tasks,
                                            rank=rank)

    def forward(self, x):
        """Forward pass"""
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultitaskMultivariateNormal(mean_x, covar_x)


class MultitaskExactGP:
    """Define a multi-output exact GP model"""

    def __init__(self, X, y, num_tasks=2, lr=0.1):
        self.likelihood = MultitaskGaussianLikelihood(num_tasks=num_tasks)
        # self.model = MultitaskExactGPModel(train_t, train_xy, likelihood)
        self.model = MultitaskExactGPModel(X, y, self.likelihood)

        # Find optimal model hyperparameters
        self.model.train()
        self.likelihood.train()

        # Use the adam optimizer
        # Includes GaussianLikelihood parameters
        self.optimizer = Adam(self.model.parameters(), lr=lr)

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
