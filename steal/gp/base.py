"""Gaussian Process base class"""

#pylint: disable=not-callable

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

    def evaluate(self, x):
        """Return the Gaussian posterior at the provided test points `x`."""
        # Set into eval mode
        self._model.eval()
        self._likelihood.eval()
        return self._likelihood(self._model(x))

    def samples(self, x, num_samples):
        """
        Return samples (e.g. a trajectory)
        from the posterior gaussian process.
        """
        posterior_model = self.evaluate(x)
        return posterior_model.sample(sample_shape=torch.Size((num_samples, )))
