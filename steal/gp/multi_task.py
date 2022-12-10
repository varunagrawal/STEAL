"""Module for multi-task/vector-valued Gaussian Processes"""

#pylint: disable=arguments-differ

import torch
from gpytorch.distributions import (MultitaskMultivariateNormal,
                                    MultivariateNormal)
from gpytorch.kernels import MultitaskKernel, RBFKernel, ScaleKernel
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from gpytorch.means import ConstantMean, MultitaskMean
from gpytorch.mlls import ExactMarginalLogLikelihood, VariationalELBO
from gpytorch.models import ApproximateGP, ExactGP
from gpytorch.variational import (CholeskyVariationalDistribution,
                                  LMCVariationalStrategy, VariationalStrategy)
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset


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


class MultitaskApproximateGPModel(ApproximateGP):
    """Multi-output GP trained via Stochastic Variational Inference"""

    def __init__(self, inducing_points, num_tasks=2, num_latents=3):

        # We have to mark the CholeskyVariationalDistribution as batch
        # so that we learn a variational distribution for each task
        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(-2), batch_shape=torch.Size([num_latents]))

        # We have to wrap the VariationalStrategy in a LMCVariationalStrategy
        # so that the output will be a MultitaskMultivariateNormal rather than a batch output
        variational_strategy = LMCVariationalStrategy(VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True),
                                                      num_tasks=num_tasks,
                                                      num_latents=num_latents,
                                                      latent_dim=-1)

        super().__init__(variational_strategy)

        # The mean and covariance modules should be marked as batch
        # so we learn a different set of hyperparameters
        self.mean_module = ConstantMean(batch_shape=torch.Size([num_latents]))
        self.covar_module = ScaleKernel(
            RBFKernel(batch_shape=torch.Size([num_latents])),
            batch_shape=torch.Size([num_latents]))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class MultitaskApproximateGP:
    """Define a multi-output variational GP model"""

    def __init__(self, num_tasks=2, num_latents=3):

        # Let's use a different set of inducing points for each latent function
        inducing_points = torch.rand(num_latents, 16, 1)
        self.model = MultitaskApproximateGPModel(inducing_points, num_tasks,
                                                 num_latents)

        self.likelihood = MultitaskGaussianLikelihood(num_tasks=num_tasks)

    def get_model(self):
        """Return the GP model"""
        return self.model

    def get_optimizer(self):
        """Return the optimizer"""
        return self.optimizer

    def get_likelihood(self):
        """Return the likelihood function."""
        return self.likelihood

    def training(self, train_x, train_y, training_iterations, lr=0.1):
        """Run Type II MLE to get the best prior hyperparameters."""

        train_dataset = TensorDataset(train_x, train_y)
        train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)

        # Find optimal model hyperparameters
        self.model.train()
        self.likelihood.train()

        if torch.cuda.is_available():
            model = model.cuda()
            likelihood = likelihood.cuda()

        # Use the adam optimizer
        params = [{
            'params': self.model.parameters()
        }, {
            'params': self.likelihood.parameters()
        }]
        optimizer = torch.optim.Adam(params, lr=lr)

        # Our loss object. We're using the VariationalELBO for variational inference.
        mll = VariationalELBO(self.likelihood,
                              self.model,
                              num_data=train_y.size(0))

        self.model.double()

        for i in range(training_iterations):
            epoch_loss = 0
            for x_batch, y_batch in train_loader:
                # Zero gradients from previous iteration
                optimizer.zero_grad()

                # Output from model
                output = self.model(x_batch)

                # Calc loss and backprop gradients
                loss = -mll(output, y_batch)

                epoch_loss += loss.detach()

                loss.backward()

                optimizer.step()

            # normalize the total loss from the epoch
            epoch_loss /= len(train_loader)
            print(
                f'Iter {i+1}/{training_iterations} - Loss: {epoch_loss.item():.3f}'
            )

    # GP evaluation
    def evaluation(self):
        """Return the likelihood of the data."""
        self.model.eval()
        self.likelihood.eval()
        return self.likelihood
