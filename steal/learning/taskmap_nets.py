"""Networks for learning TaskMaps"""

import torch
from loguru import logger
from torch import nn

from steal.learning.base import FCNN, RFFN
from steal.rmpflow.kinematics import ComposedTaskMap, TaskMap

#pylint: disable=method-hidden,not-callable


class ElementwiseAffineTaskMap(TaskMap):
    """
    A task map which scales and adds a bias (aka applies an affine transform)
    to every element of the input.
    """

    def __init__(self,
                 n_inputs,
                 scaling=None,
                 bias=None,
                 device=torch.device('cpu')):
        super(ElementwiseAffineTaskMap, self).__init__(n_inputs=n_inputs,
                                                       n_outputs=n_inputs,
                                                       psi=self.psi,
                                                       J=self.J,
                                                       J_dot=self.J_dot,
                                                       device=device)
        if scaling is not None:
            if scaling.dim() == 1:
                scaling = scaling.reshape(1, -1)
            self.register_buffer(
                'scaling',
                scaling.to(device=device, dtype=torch.get_default_dtype()))
        else:
            self.register_buffer('scaling',
                                 torch.ones(1, n_inputs, device=device))

        if bias is not None:
            if bias.dim() == 1:
                bias = bias.reshape(1, -1)
            self.register_buffer(
                'bias', bias.to(device=device,
                                dtype=torch.get_default_dtype()))
        else:
            self.register_buffer('bias', torch.zeros(1,
                                                     n_inputs,
                                                     device=device))

    def psi(self, x):
        """Apply the task map function."""
        return x * self.scaling + self.bias

    def J(self, x):
        return torch.diag_embed(self.scaling).repeat(x.shape[0], 1, 1)

    def J_dot(self, x, xd):
        return torch.zeros(self.n_inputs, self.n_inputs,
                           device=self.device).repeat(x.shape[0], 1, 1)


class CouplingLayer(TaskMap):
    """
    An implementation of a coupling layer from Dinh et. al.'s RealNVP paper.
    (https://arxiv.org/abs/1605.08803).
    """

    def __init__(self,
                 n_inputs,
                 n_hidden,
                 mask,
                 base_network='rfnn',
                 s_act='elu',
                 t_act='elu',
                 sigma=0.45,
                 device=torch.device('cpu')):

        super().__init__(n_inputs=n_inputs,
                         n_outputs=n_inputs,
                         psi=self.psi,
                         J=self.J,
                         device=device)

        self.num_inputs = n_inputs
        self.register_buffer(
            'mask', mask.to(device=device, dtype=torch.get_default_dtype()))

        if base_network == 'fcnn':
            self.scale_net = FCNN(in_dim=n_inputs,
                                  out_dim=n_inputs,
                                  hidden_dim=n_hidden,
                                  act=s_act).to(device=device)
            self.translate_net = FCNN(in_dim=n_inputs,
                                      out_dim=n_inputs,
                                      hidden_dim=n_hidden,
                                      act=t_act).to(device=device)
            logger.info("neural network initialized with identity map!")

            nn.init.zeros_(self.translate_net.network[-1].weight.data)
            nn.init.zeros_(self.translate_net.network[-1].bias.data)

            nn.init.zeros_(self.scale_net.network[-1].weight.data)
            nn.init.zeros_(self.scale_net.network[-1].bias.data)

        elif base_network == 'rfnn':
            logger.info(
                f"Random fourier feature bandwidth = {sigma}. Change it as per data!"
            )
            self.scale_net = RFFN(in_dim=n_inputs,
                                  out_dim=n_inputs,
                                  nfeat=n_hidden,
                                  sigma=sigma).to(device=device)
            self.translate_net = RFFN(in_dim=n_inputs,
                                      out_dim=n_inputs,
                                      nfeat=n_hidden,
                                      sigma=sigma).to(device=device)

            logger.info('Initializing coupling layers as identity!')
            nn.init.zeros_(self.translate_net.network[-1].weight.data)
            nn.init.zeros_(self.scale_net.network[-1].weight.data)
        else:
            raise TypeError('The network type has not been defined')

    def psi(self, x, mode='direct'):
        """Apply the task map function."""
        mask = self.mask
        masked_inputs = x * mask

        log_s = self.scale_net(masked_inputs) * (1 - mask)
        t = self.translate_net(masked_inputs) * (1 - mask)

        if mode == 'direct':
            s = torch.exp(log_s)
            return x * s + t
        else:
            s = torch.exp(-log_s)
            return (x - t) * s

    def J(self, x, mode='direct'):
        mask = self.mask
        masked_inputs = x * mask
        if mode == 'direct':
            log_s = self.scale_net(masked_inputs) * (1 - mask)
            s = torch.exp(log_s)  #TODO: remove redundancy!
            J_s = self.scale_net.jacobian(masked_inputs)
            J_t = self.translate_net.jacobian(masked_inputs)

            J = (J_s * (x * s * (1 - mask)).unsqueeze(2) + J_t *
                 ((1 - mask).unsqueeze(1))) * mask
            J = J + torch.diag_embed(s)
            return J
        else:
            raise NotImplementedError


class InvertibleMM(TaskMap):
    """
    An implementation of a invertible matrix multiplication
    layer from Glow: Generative Flow with Invertible 1x1 Convolutions
    (https://arxiv.org/abs/1807.03039).
    """

    def __init__(self, n_inputs, device=torch.device('cpu')):
        self.device = device
        self.num_inputs = n_inputs
        self.W_nn = nn.Parameter(torch.zeros(n_inputs,
                                             n_inputs,
                                             device=self.device),
                                 requires_grad=True)
        print('Initializing invertible convolution layer as identity!')
        super(InvertibleMM, self).__init__(n_inputs=n_inputs,
                                           n_outputs=n_inputs,
                                           psi=self.psi,
                                           device=device)

    @property
    def W(self):
        """Invertible weight matrix."""
        return self.W_nn + torch.eye(self.num_inputs, device=self.device)

    def psi(self, x, mode='direct'):
        """Apply the task map function."""
        if mode == 'direct':
            return x.matmul(self.W)
        else:
            return x.matmul(torch.inverse(self.W))


class LatentTargetTaskMap(TaskMap):
    """Task map to learn to reach a goal in a latent space."""

    def __init__(self,
                 n_inputs,
                 latent_taskmap,
                 goal=None,
                 device=torch.device('cpu')):
        super(LatentTargetTaskMap, self).__init__(n_inputs=n_inputs,
                                                  n_outputs=n_inputs,
                                                  psi=self.psi,
                                                  J=self.J,
                                                  J_dot=self.J_dot,
                                                  device=device)
        self.latent_taskmap = latent_taskmap

        if goal is not None:
            if goal.ndim == 1:
                goal = goal.reshape(1, -1)
            self.register_buffer(
                'goal', goal.to(device=device,
                                dtype=torch.get_default_dtype()))
        else:
            self.register_buffer('goal',
                                 torch.zeros(1, self.n_inputs, device=device))

    def psi(self, x):
        """Apply the task map function."""
        return x - self.latent_taskmap.psi(self.goal)

    def J(self, x):
        return torch.eye(self.n_inputs).repeat(x.shape[0], 1, 1)

    def J_dot(self, x, xd):
        return torch.zeros(self.n_inputs, self.n_inputs,
                           device=self.device).repeat(x.shape[0], 1, 1)


class EuclideanizingFlow(ComposedTaskMap):
    """
    A sequential container of flows.

    num_dims: dimensions of input and output
    num_blocks: number of modules in the flow
    num_hidden: hidden layer dimensions or number of features for scaling and translation functions
    s_act: (only for coupling_layer_type='fcnn') activation fcn for scaling
    t_act: (only for coupling_layer_type:'fcnn') activation fcn for translation
    sigma: (only for coupling_layer_type:'rfnn') length scale for random for fourier features
    flow_type: (realnvp/glow) selects the submodules in each module depending on the architecture
    coupling_layer_type: (fcnn/rfnn) representatation for scaling and translation

    NOTE: Orginal RealNVP and GLOW uses normalization routines, skipped for now!
    """

    def __init__(self,
                 n_inputs,
                 n_blocks,
                 n_hidden,
                 s_act=None,
                 t_act=None,
                 sigma=None,
                 flow_type='realnvp',
                 coupling_network_type='fcnn',
                 goal=None,
                 normalization_scaling=None,
                 normalization_bias=None,
                 device=torch.device('cpu')):

        super().__init__(device=device)

        taskmap_list = []
        # data normalization step
        input_norm_taskmap = ElementwiseAffineTaskMap(
            n_inputs=n_inputs,
            bias=normalization_bias,
            scaling=normalization_scaling,
            device=device)
        taskmap_list.append(input_norm_taskmap)

        # space warping step
        logger.info(f"Using the {coupling_network_type} for coupling layer")
        if flow_type == 'realnvp':
            mask = torch.arange(0, n_inputs) % 2  # alternating inputs
            mask = mask.to(device=device, dtype=torch.get_default_dtype())
            for _ in range(n_blocks):  # TODO: Try batchnorm again
                taskmap_list += [
                    CouplingLayer(n_inputs=n_inputs,
                                  n_hidden=n_hidden,
                                  mask=mask,
                                  s_act=s_act,
                                  t_act=t_act,
                                  sigma=sigma,
                                  base_network=coupling_network_type,
                                  device=device),
                ]
                # mask = 1 - mask  # flipping mask
                mask = torch.roll(mask, shifts=-1, dims=0)
        elif flow_type == 'glow':
            mask = torch.arange(0, n_inputs) % 2  # alternating inputs
            mask = mask.to(device=device, dtype=torch.get_default_dtype())
            for _ in range(n_blocks):  # TODO: Try ActNorm again
                taskmap_list += [
                    InvertibleMM(n_inputs),
                    CouplingLayer(n_inputs=n_inputs,
                                  n_hidden=n_hidden,
                                  mask=mask,
                                  s_act=s_act,
                                  t_act=t_act,
                                  sigma=sigma,
                                  base_network=coupling_network_type,
                                  device=device),
                ]
                # TODO: Not sure if this mask needs flipping
                # mask = 1 - mask  # flipping mask
                # mask = torch.roll(mask, shifts=-1, dims=0)
        else:
            raise TypeError('Unknown Flow Type!')

        # defining a composed map for normalization and warping
        latent_taskmap = ComposedTaskMap(taskmaps=taskmap_list, device=device)

        # setting up the overall composed taskmap (euclidenization + fixing origin)
        self.taskmaps.append(latent_taskmap)

        # fixing the origin to goal in latent space
        latent_target_taskmap = LatentTargetTaskMap(
            n_inputs=n_inputs,
            latent_taskmap=latent_taskmap,
            goal=goal,
            device=device)

        self.taskmaps.append(latent_target_taskmap)

        if flow_type == 'realnvp' and coupling_network_type == 'rfnn':
            # use analytic jacobian for rfnn type
            logger.info('Using Analytic Jacobian!')
            self.use_numerical_jacobian = False
        else:
            logger.info('Using Numerical Jacobian!')
            self.use_numerical_jacobian = True
