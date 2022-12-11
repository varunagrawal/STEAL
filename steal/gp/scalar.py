"""Module for simple scalar-valued GPs."""

#pylint: disable=arguments-differ

from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import ExactGP
from torch.optim import Adam

from steal.gp.base import BaseGaussianProcess


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


class ScalarGaussianProcess(BaseGaussianProcess):
    """Define a simple Exact GP model"""

    def __init__(self, X, y) -> None:
        super().__init__()

        # Initialize likelihood and model
        self._likelihood = GaussianLikelihood()
        self._model = ExactGPModel(X, y, self._likelihood)

        self._model.double()

    def train(self, X, y, training_iterations, lr=0.1):
        """Run Type II MLE to get the best prior hyperparameters."""

        # Find optimal model hyperparameters
        self._model.train()
        self._likelihood.train()

        # Use the adam optimizer
        # Includes GaussianLikelihood parameters
        optimizer = Adam(self._model.parameters(), lr=lr)

        # "Loss" for GPs - the marginal log likelihood
        mll = ExactMarginalLogLikelihood(self._likelihood, self._model)

        for i in range(training_iterations):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = self._model(X)

            # Calc loss and backprop gradients
            loss = -mll(output, y)
            loss.backward()
            print(
                f'Iter {i+1}/{training_iterations} - Loss: {loss.item():.3f}' \
                f'   lengthscale: {self._model.covar_module.base_kernel.lengthscale.item():.3f}'\
                f'   noise: {self._model.likelihood.noise.item():.3f}'
            )

            optimizer.step()
