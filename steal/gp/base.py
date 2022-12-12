"""Gaussian Process base class"""

#pylint: disable=not-callable

import gpytorch
import torch


class BaseGaussianProcess:
    """Gaussian process base class."""

    def __init__(self):
        self._model = None
        self._likelihood = None

    def model(self):
        """Return the GP model"""
        return self._model

    def likelihood(self):
        """Return the likelihood function."""
        return self._likelihood

    def train(self, X, y, training_iterations, lr=0.1):
        """Train the GP"""

    def posterior(self, x):
        """Compute the posterior of the underlying function represented by the GP."""
        # Set into eval mode
        self._model.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            return self._model(x)

    def evaluate(self, x):
        """
        Return the Gaussian posterior at the provided test points `x`.
        Also called the Posterior Predictive Distribution.
        """
        # Set into eval mode
        self._model.eval()
        self._likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            return self._likelihood(self._model(x))

    def samples(self, x, num_samples, posterior_model=None):
        """
        Return samples (e.g. a trajectory)
        from the posterior gaussian process.
        """
        posterior_model = self.posterior(x)
        return posterior_model.sample(sample_shape=torch.Size((num_samples, )))
