"""
Module for creating an RMP-Tree in order to generate a global Riemannian Motion Policy
from motion policies on subtasks defined by the leaves in the tree.
"""
import abc

import torch
from torch import nn

from steal.rmpflow.kinematics.taskmaps import TaskMap


# TODO: rmp order should be an input to the initializer
class Rmp(nn.Module):
    """A Riemannian Motion Policy."""

    def __init__(self, return_natural=True):
        super(Rmp, self).__init__()
        # By default return natural form, otherwise acceleration/velocity
        self.return_natural = return_natural

    def forward(self, x, xd=None, t=None):
        """Forward Pass of Neural Network layer"""
        if x.ndim == 1:
            x = x.reshape(1, -1)
            if xd is not None and xd.ndim == 1:
                xd = xd.reshape(1, -1)
        f, M = self.eval_natural(x=x, xd=xd, t=t)

        if not self.training:
            f = f.detach()
            M = M.detach()

        if self.return_natural:
            return f, M
        xdd = torch.einsum('bij,bj->bi', torch.pinverse(M), f)
        return xdd

    @abc.abstractmethod
    def eval_natural(self, x, xd, t=None):
        """Evaluate the RMP in natural form."""
        pass

    def eval_canonical(self, x, xd, t=None):
        """Evaluate the RMP in canonical form."""
        f, M = self(x=x, xd=xd, t=t)
        xdd = torch.einsum('bij,bj->bi', torch.pinverse(M), f)
        return xdd, M


class RmpTreeNode(Rmp):
    """
    Node in the RMP tree.
    TODO: Add extra_repr for printing out the tree branching out of it
    """

    def __init__(self,
                 n_dim,
                 name="",
                 order=2,
                 return_natural=True,
                 device=torch.device('cpu')):
        super(RmpTreeNode, self).__init__(return_natural=return_natural)
        self.n_dim = n_dim  # dimension of the node task space
        self.name = name
        # list of edges connecting other nodes
        self.edges = torch.nn.ModuleList()
        # list of leaf rmps
        self.rmps = torch.nn.ModuleList()
        # order of rmp node (1: momentum-based, 2: force-based)
        self.order = order
        self.device = device

    def add_rmp(self, rmp):
        """Add an RMP to which operates on the manifold represented by this node."""
        self.rmps.append(rmp)

    def add_task_space(self, task_map: TaskMap, name=""):
        """Generate a new Task Space and add it to the RMP tree.

        Args:
            task_map (TaskMap): The task map which converts
                to the task space represented by the child node.
            name (str, optional): The name of the RMP tree node
                assigned to the task map. Defaults to "".

        Returns:
            RmpTreeNode: The child RMPTree node added.
        """
        assert (
            self.n_dim == task_map.n_inputs), ValueError('Dimension mismatch!')
        child_node = RmpTreeNode(n_dim=task_map.n_outputs,
                                 name=name,
                                 order=self.order,
                                 device=self.device)
        edge = RmpTreeEdge(task_map=task_map,
                           child_node=child_node,
                           order=self.order,
                           device=self.device)
        self.edges.append(edge)
        return child_node

    def eval_natural(self, x, xd=None, t=None):
        """Evaluate the RMPs in natural form."""
        assert (x.shape[-1] == self.n_dim), ValueError('Dimension mismatch!')

        n_pts = x.shape[0]
        f = torch.zeros(n_pts, self.n_dim, device=self.device)
        M = torch.zeros(n_pts, self.n_dim, self.n_dim, device=self.device)

        for edge in self.edges:
            f_i, M_i = edge(x=x, xd=xd, t=t)
            f += f_i
            M += M_i

        for rmp in self.rmps:
            f_i, M_i = rmp(x=x, xd=xd, t=t)
            f += f_i
            M += M_i

        return f, M

    @property
    def n_edges(self):
        """Get the number of edges to from this node."""
        return len(self.edges)

    @property
    def n_rmps(self):
        """Get the number of RMPs in this node."""
        return len(self.rmps)


class RmpTreeEdge(Rmp):
    """
    An edge in the RMP Tree.
    The edge represents a transform from one manifold (or task space) to another,
    which is achieved via the `task_map`.
    The associated child node with this edge is the RMP in the transformed manifold.
    """

    def __init__(self,
                 task_map,
                 child_node,
                 order=2,
                 return_natural=True,
                 device=torch.device('cpu')):
        super(RmpTreeEdge, self).__init__(return_natural=return_natural)
        self.task_map = task_map  # mapping from parent to child node
        self.child_node = child_node  # child node
        self.order = order
        self.device = device

        assert self.order in [1, 2], TypeError('Invalid RMP order!')

    def eval_natural(self, x, xd=None, t=None):
        """Evaluate the RMP in natural form."""
        if self.order == 2:
            # pushforward
            y, yd, J, Jd = self.task_map(x=x, xd=xd, order=self.order)
            f_y, M_y = self.child_node(x=y, xd=yd, t=t)

            # pullback
            M = torch.einsum('bji, bjk, bkl->bil', J, M_y, J)
            f = torch.einsum('bji, bj->bi', J, f_y) - torch.einsum(
                'bji, bjk, bkl, bl->bi', J, M_y, Jd, xd)
            return f, M

        elif self.order == 1:
            # pushforward
            y, J = self.task_map(x=x, order=self.order)
            p_y, M_y = self.child_node(x=y, t=t)

            # pullback
            M = torch.einsum('bji, bjk, bkl->bil', J, M_y, J)
            p = torch.einsum('bji, bj->bi', J, p_y)
            return p, M
