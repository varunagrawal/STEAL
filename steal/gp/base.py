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

    def train(self, X, y, training_iterations, lr=0.1):
        """Train the GP"""

    def evaluate(self, X):
        """Evaluate the gaussian process at the provided test points `X`."""
        # Set into eval mode
        self._model.eval()
        self._likelihood.eval()
        return self._likelihood(self._model(X))

    def sample(self):
        """
        Sample a function (e.g. a trajectory)
        from the gaussian process.
        """
