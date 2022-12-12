"""Basic neural networks that are used for learning."""

import numpy as np
import torch
import torch.nn.functional as F
from torch import autograd, nn


class LinearClamped(nn.Module):
    """
    Linear layer with user-specified parameters (not to be learrned!)
    """

    __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self, in_features, out_features, weights, bias_values=None):
        super(LinearClamped, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer(
            'weight',
            torch.tensor(weights).to(dtype=torch.get_default_dtype()))
        if bias_values is not None:
            self.register_buffer(
                'bias',
                torch.tensor(bias_values).to(dtype=torch.get_default_dtype()))
        else:
            self.register_buffer('bias', None)

    def forward(self, x):
        """Forward pass"""
        if x.dim() == 1:
            return F.linear(x.view(1, -1), self.weight, self.bias)
        return F.linear(x, self.weight, self.bias)

    def extra_repr(self):
        return f"in_features={self.in_features}, "\
            "out_features={self.out_features}, "\
                "bias={self.bias is not None}"


class Sin(nn.Module):
    """
    Applies the sine function element-wise.
    """

    def forward(self, inputs):
        """Forward pass"""
        return torch.sin(inputs)


class Cos(nn.Module):
    """
    Applies the cosine function element-wise.
    """

    def forward(self, inputs):
        """Forward pass"""
        return torch.cos(inputs)


class RFFN(nn.Module):
    """
    Random Fourier features network.
    """

    def __init__(self, in_dim, out_dim, nfeat, sigma=10.):
        super(RFFN, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.sigma = np.ones(in_dim) * sigma
        self.coeff = np.random.normal(0.0, 1.0, (nfeat, in_dim))
        self.coeff = self.coeff / self.sigma.reshape(1, len(self.sigma))
        self.offset = 2.0 * np.pi * np.random.rand(1, nfeat)

        self.register_buffer(
            'coeff_tensor',
            torch.tensor(self.coeff).to(dtype=torch.get_default_dtype()))
        self.register_buffer(
            'offset_tensor',
            torch.tensor(self.offset).to(dtype=torch.get_default_dtype()))

        self.network = nn.Sequential(
            LinearClamped(in_dim, nfeat, self.coeff, self.offset), Cos(),
            nn.Linear(nfeat, out_dim, bias=False))

        self.jacobian = self.jacobian_analytic

    def forward(self, x):
        """Forward pass"""
        return self.network(x)

    def jacobian_analytic(self, x):
        """Compute analytic jacobian"""
        n = x.shape[0]
        y = F.linear(x, self.coeff_tensor, self.offset_tensor)
        y = -torch.sin(y)
        y = y.repeat_interleave(self.in_dim, dim=0)
        y = y * self.coeff_tensor.t().repeat(n, 1)
        J = F.linear(y, self.network[-1].weight, bias=None)
        J = J.reshape(n, self.out_dim, self.in_dim).permute(0, 2, 1)
        return J

    def jacobian_numeric(self, inputs):
        """Compute numerical jacobian"""
        if inputs.ndimension() == 1:
            n = 1
        else:
            n = inputs.size()[0]
        inputs = inputs.repeat(1, self.in_dim).view(-1, self.in_dim)
        inputs.requires_grad_(True)
        y = self(inputs)
        mask = torch.eye(self.in_dim).repeat(n, 1)
        J = autograd.grad(y, inputs, mask, create_graph=True)[0]
        J = J.reshape(n, self.in_dim, self.in_dim)
        return J


class FCNN(nn.Module):
    """
    2-layer fully connected neural network
    """

    def __init__(self, in_dim, out_dim, hidden_dim, act='tanh'):
        super(FCNN, self).__init__()
        activations = {
            'relu': nn.ReLU,
            'sigmoid': nn.Sigmoid,
            'tanh': nn.Tanh,
            'leaky_relu': nn.LeakyReLU,
            'elu': nn.ELU,
            'prelu': nn.PReLU,
            'softplus': nn.Softplus
        }

        act_func = activations[act]
        self.network = nn.Sequential(nn.Linear(in_dim, hidden_dim), act_func(),
                                     nn.Linear(hidden_dim, hidden_dim),
                                     act_func(),
                                     nn.Linear(hidden_dim, out_dim))

    def forward(self, x):
        """Forward pass"""
        return self.network(x)
