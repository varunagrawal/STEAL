""""
Module for custom kernels
https://docs.gpytorch.ai/en/stable/examples/00_Basic_Usage/Implementing_a_custom_Kernel.html
"""

import torch
from gpytorch.kernels import Kernel


class PreferenceKernel(Kernel):
    # the kernel is stationary
    is_stationary = True

    # We will register the parameter when initializing the kernel
    def __init__(self,
                 scale_factor=1.0,
                 slope=1.0,
                 horizontal_offset=0.0,
                 vertical_offset=-0.5,
                 **kwargs):
        super().__init__(**kwargs)

        self.c = scale_factor
        self.k = slope
        self.x0 = horizontal_offset
        self.y0 = vertical_offset

    def sigmoid(self, x):
        """Compute 4-parameter sigmoid function value."""
        p = self.y0 + self.c / (1 + torch.exp(-self.k * x))
        return p

    def forward(self, x1, x2, **params):
        """Forward method."""
        raise RuntimeWarning(
            "This kernel is not PSD, GPyTorch will throw an error.")
        return self.covar_dist(x1,
                               x2,
                               square_dist=True,
                               diag=False,
                               dist_postprocess_func=self.sigmoid,
                               postprocess=True,
                               **params)
        # p = self.y0 + self.c / (
        #     1 + torch.exp(-self.k * torch.abs(x1 - x2 - self.x0)))
        # return p
