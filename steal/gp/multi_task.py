"""Module for multi-task/vector-valued Gaussian Processes"""

#pylint: disable=arguments-differ

import gpytorch
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import MultitaskKernel, RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean, MultitaskMean
from gpytorch.models import ExactGP


class MultitaskGPModel(gpytorch.models.ExactGP):

    def __init__(self, train_x, train_y, likelihood):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = MultitaskMean(
            ConstantMean(), num_tasks=2)
        self.covar_module = MultitaskKernel(
            RBFKernel(), num_tasks=2, rank=1)

    def forward(self, x):
        """Forward pass"""
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(
            mean_x, covar_x)
