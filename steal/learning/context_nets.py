"""Context networks which aggregate the information from all the leaves."""

from torch import nn
import torch


class ContextMomentumNet(nn.Module):
    """
    A network which predicts the momentum at each leaf of an RMP tree
    and aggregates with via the pullback to give the momentum and Riemannian metric
    for the common root node.
    """

    def __init__(self,
                 lagrangian_vel_nets,
                 cspace_dims,
                 metric_scaling=None,
                 name=None):
        super().__init__()
        self.lagrangian_vel_nets = nn.ModuleList(
            [net for net in lagrangian_vel_nets])
        self.cspace_dims = cspace_dims
        self.name = name

        if metric_scaling is None:
            self.metric_scaling = [1. for net in self.lagrangian_vel_nets]
        else:
            self.metric_scaling = metric_scaling

        assert len(self.metric_scaling) == len(self.lagrangian_vel_nets)

    def forward(self, state, q_leaves: list, jacobians: list, momentums: list,
                metrics: list):
        """Forward pass

        Args:
            state (torch.Tensor): The state of the robot (e.g. joint angles)
            q_leaves (list): The RMPTreeNode leaves for each of the joints
            jacobians (list): The jacobian for each of the leaves if it is not learned.
            momentums (list): The momentum for each of the leaves if it is not learned.
            metrics (list): The Riemannian metrics for each of the leaves if it is not learned.

        Returns:
            tuple[Tensor, Tensor, List[Tensor]]: The momentum, Riemannian metric and jacobians
            for the RMPTreeNode which is the root for each of the joint leaves.
        """
        assert state.dim() == 1 or state.dim() == 2 or state.dim() == 3
        if state.dim() == 1:
            state = state.unsqueeze(0)
        elif state.dim() == 3:
            state = state.squeeze(2)

        assert state.dim() == 2
        assert state.size()[1] == self.cspace_dims

        # Number of samples in the trajectory for the robot state (e.g. joint angles)
        n_samples = state.size(0)

        momentum_root = torch.zeros(n_samples, self.cspace_dims)
        metric_root = torch.zeros(n_samples, self.cspace_dims,
                                  self.cspace_dims)
        net_jacobians = []

        for X in zip(self.lagrangian_vel_nets, q_leaves, jacobians, momentums,
                     metrics, self.metric_scaling):
            net, x, J, momentum, metric, scaling = X
            if net is not None:
                momentum, metric = net(x)
                net_jacobians.append(J)
            momentum_root += scaling * torch.einsum('bji, bj->bi', J, momentum)
            metric_root += scaling * torch.einsum('bji, bjk, bkl->bil', J,
                                                  metric, J)
        return momentum_root, metric_root, net_jacobians
