"""Gaussian Process base class"""

#pylint: disable=not-callable


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

    def train(self, X, y, num_epochs):
        """Train the GP"""

    def evaluate(self, X):
        """GP evaluation"""
        self._model.eval()
        self._likelihood.eval()
        return self._likelihood(self._model(X))

    def sample(self):
        """
        Sample a function (e.g. a trajectory)
        from the gaussian process.
        """
