""""
Module for custom kernels
https://docs.gpytorch.ai/en/stable/examples/00_Basic_Usage/Implementing_a_custom_Kernel.html
"""

from gpytorch.kernels import Kernel
from gpytorch.constraints import Positive
import torch
import math

class PreferenceKernel(Kernel):
    # the sinc kernel is stationary
    is_stationary = True

    # We will register the parameter when initializing the kernel
    def __init__(self, scale_prior=None, scale_constraint=None, slope_prior=None,
     slope_constraint=None, horizontal_prior=None, horizontal_constraint=None, 
     vertical_prior=None, vertical_constraint=None, **kwargs):
        super().__init__(**kwargs)

        # register the parameter: c
        self.register_parameter(
            name='raw_scale', parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1))
        )

        # register the parameter: k
        self.register_parameter(
            name='raw_slope', parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1))
        )

        # register the parameter: x0
        self.register_parameter(
            name='raw_horizontal', parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1))
        )

        # register the parameter: y0
        self.register_parameter(
            name='raw_vertical', parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1))
        )

        # set the parameter constraint to be positive, when nothing is specified
        if scale_constraint is None:
            scale_constraint = Positive()

        # register the constraint
        self.register_constraint("raw_scale", scale_constraint)

        # set the parameter prior, see
        # https://docs.gpytorch.ai/en/latest/module.html#gpytorch.Module.register_prior
        if scale_prior is not None:
            self.register_prior(
                "scale_prior",
                scale_prior,
                lambda m: m.scale,
                lambda m, v : m._set_scale(v),
            )
        
        # set the parameter constraint to be positive, when nothing is specified
        if slope_constraint is None:
            slope_constraint = Positive()

        # register the constraint
        self.register_constraint("raw_slope", slope_constraint)

        # set the parameter prior, see
        # https://docs.gpytorch.ai/en/latest/module.html#gpytorch.Module.register_prior
        if slope_prior is not None:
            self.register_prior(
                "slope_prior",
                slope_prior,
                lambda m: m.slope,
                lambda m, v : m._set_slope(v),
            )
        
        # set the parameter constraint to be positive, when nothing is specified
        if horizontal_constraint is None:
            horizontal_constraint = Positive()

        # register the constraint
        self.register_constraint("raw_horizontal", horizontal_constraint)

        # set the parameter prior, see
        # https://docs.gpytorch.ai/en/latest/module.html#gpytorch.Module.register_prior
        if horizontal_prior is not None:
            self.register_prior(
                "horizontal_prior",
                horizontal_prior,
                lambda m: m.horizontal,
                lambda m, v : m._set_horizontal(v),
            )
        
        # set the parameter constraint to be positive, when nothing is specified
        if vertical_constraint is None:
            vertical_constraint = Positive()

        # register the constraint
        self.register_constraint("raw_vertical", vertical_constraint)

        # set the parameter prior, see
        # https://docs.gpytorch.ai/en/latest/module.html#gpytorch.Module.register_prior
        if vertical_prior is not None:
            self.register_prior(
                "vertical_prior",
                vertical_prior,
                lambda m: m.vertical,
                lambda m, v : m._set_vertical(v),
            )

        

    # now set up the 'actual' paramter
    @property
    def scale(self):
        # when accessing the parameter, apply the constraint transform
        return self.raw_scale_constraint.transform(self.raw_scale)

    @scale.setter
    def scale(self, value):
        return self._set_scale(value)

    def _set_scale(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_scale)
        # when setting the paramater, transform the actual value to a raw one by applying the inverse transform
        self.initialize(raw_scale=self.raw_scale_constraint.inverse_transform(value))
    
    # now set up the 'actual' paramter
    @property
    def slope(self):
        # when accessing the parameter, apply the constraint transform
        return self.raw_slope_constraint.transform(self.raw_slope)

    @slope.setter
    def slope(self, value):
        return self._set_slope(value)

    def _set_slope(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_slope)
        # when setting the paramater, transform the actual value to a raw one by applying the inverse transform
        self.initialize(raw_slope=self.raw_slope_constraint.inverse_transform(value))
    
    # now set up the 'actual' paramter
    @property
    def horizontal(self):
        # when accessing the parameter, apply the constraint transform
        return self.raw_horizontal_constraint.transform(self.raw_horizontal)

    @horizontal.setter
    def horizontal(self, value):
        return self._set_horizontal(value)

    def _set_horizontal(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_horizontal)
        # when setting the paramater, transform the actual value to a raw one by applying the inverse transform
        self.initialize(raw_horizontal=self.raw_horizontal_constraint.inverse_transform(value))
    
    # now set up the 'actual' paramter
    @property
    def vertical(self):
        # when accessing the parameter, apply the constraint transform
        return self.raw_vertical_constraint.transform(self.raw_vertical)

    @vertical.setter
    def vertical(self, value):
        return self._set_vertical(value)

    def _set_vetical(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_vertical)
        # when setting the paramater, transform the actual value to a raw one by applying the inverse transform
        self.initialize(raw_vertical=self.raw_vertical_constraint.inverse_transform(value))

    # this is the kernel function
    def forward(self, x1, x2, **params):
        print(x1.shape)
        print(x2.shape)
        noise_level = 1.0*math.sin(x1); #params[0]
        return (self.scale / (1 + torch.exp(-self.slope*(noise_level - self.horizontal)))) + self.vertical

