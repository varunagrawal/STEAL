import numpy as np
import torch

from steal.rmpflow.kinematics import TaskMap

#pylint: disable=method-hidden


class TargetTaskMap(TaskMap):
    """
    Vector to goal point
    """

    def __init__(self, goal, device=torch.device('cpu')):

        super(TargetTaskMap, self).__init__(n_inputs=goal.shape[1],
                                            n_outputs=goal.shape[1],
                                            psi=self.psi,
                                            J=self.J,
                                            J_dot=self.J_dot,
                                            device=device)
        self.device = device
        self.register_buffer(
            'goal', goal.to(device=device, dtype=torch.get_default_dtype()))

    def psi(self, y):
        return y - self.goal

    def J(self, y):
        return torch.eye(self.n_inputs,
                         device=self.device).repeat(y.shape[0], 1, 1)

    def J_dot(self, y, y_dot):
        return torch.zeros(self.n_inputs, self.n_inputs,
                           device=self.device).repeat(y.shape[0], 1, 1)


class TargetTaskMapTF(TaskMap):

    def __init__(self, T, n_inputs=3, device=torch.device('cpu')):
        self.R = T[0:3, 0:3].to(device=device, dtype=torch.get_default_dtype())
        self.t = T[0:3, -1].reshape(1, -1).to(device=device,
                                              dtype=torch.get_default_dtype())
        self.n_inputs = n_inputs
        TaskMap.__init__(self,
                         n_inputs=self.n_inputs,
                         n_outputs=self.n_inputs,
                         psi=self.psi,
                         J=self.J,
                         J_dot=self.J_dot)

    def psi(self, x):
        return torch.einsum('ij, bj -> bi', self.R, x) + self.t

    def J(self, x):
        return self.R.repeat(x.shape[0], 1, 1)

    def J_dot(self, x, xd):
        return torch.zeros(x.shape[0],
                           self.n_inputs,
                           self.n_inputs,
                           device=self.device)


class SphericalTargetTaskMap(TaskMap):

    def __init__(self, center, radius=0.0, device=torch.device('cpu')):
        super(SphericalTargetTaskMap, self).__init__(n_inputs=center.shape[1],
                                                     n_outputs=center.shape[1],
                                                     psi=self.psi,
                                                     J=self.J,
                                                     device=device)
        self.register_buffer(
            'center', center.to(device=device,
                                dtype=torch.get_default_dtype()))
        self.register_buffer(
            'radius',
            torch.tensor(radius).to(device=device,
                                    dtype=torch.get_default_dtype()))

    def psi(self, x):
        y = x - self.center
        y = y - self.radius * y / torch.norm(y, dim=1).reshape(-1, 1)
        return y

    def J(self, x):
        y = x - self.center
        y_norm = torch.norm(y, dim=1).repeat_interleave(self.n_inputs**2).view(
            -1, self.n_inputs, self.n_inputs)
        J = (1. - (self.radius / y_norm)) * \
            torch.eye(self.n_inputs, device=self.device).repeat(x.shape[0], 1, 1) + \
                (self.radius / y_norm**2) * torch.einsum('bi, bj-> bij', y, y)
        return J


class DimSelectorTaskMap(TaskMap):
    """Select a subset of dimensions in the input."""

    def __init__(self, n_inputs, selected_dims, device=torch.device('cpu')):
        self.selected_dims = selected_dims

        if isinstance(selected_dims, int):
            selected_dims = torch.tensor([selected_dims])
            n_outputs = 1
        elif isinstance(selected_dims, (torch.Tensor, np.ndarray)):
            assert (selected_dims.ndim == 1
                    ), ValueError('Selected dims has to be 1-D array or int')
            n_outputs = selected_dims.shape[-1]
        else:
            raise ValueError('Selected dims has to be 1-D array or int')

        self.jacobian = torch.zeros(n_outputs, n_inputs, device=device)
        for n in range(n_outputs):
            self.jacobian[n, selected_dims[n]] = 1.

        super(DimSelectorTaskMap, self).__init__(n_inputs=n_inputs,
                                                 n_outputs=n_outputs,
                                                 psi=self.psi,
                                                 J=self.J,
                                                 J_dot=self.J_dot,
                                                 device=device)

    def psi(self, x):
        return x[:, self.selected_dims].reshape(x.shape[0], -1)

    def J(self, x):
        return self.jacobian.repeat(x.shape[0], 1, 1)

    def J_dot(self, x, xd):
        return torch.zeros(self.n_outputs, self.n_inputs,
                           device=self.device).repeat(x.shape[0], 1, 1)


if __name__ == '__main__':
    sphere_map = SphericalTargetTaskMap(center=torch.zeros(1, 2), radius=1.)

    tm = DimSelectorTaskMap(n_inputs=4, selected_dims=torch.arange(0, 2))
    y, yd, J, Jd = tm(torch.zeros(10, 4), torch.zeros(10, 4), order=2)
    T = torch.eye(4)
    T[0, 0] = 2.
    T[0, 1] = 3.
    tm2 = TargetTaskMapTF(T=T)

    x = torch.ones(5, 3)
    xd = torch.zeros(5, 3)
    y, yd, J, Jd = tm2(x, xd)
    print(y)
