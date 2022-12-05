"""Class to train a deep neural network (with the help of Pytorch-Lightning)."""

#pylint: disable=arguments-differ

from itertools import chain

import pytorch_lightning as pl
from torch import nn, optim

from steal.learning.context_nets import ContextMomentumNet
from steal.learning.controllers.metric_networks import MetricCholNet
from steal.learning.loss import LagrangianMomentumLoss
from steal.learning.taskmap_nets import EuclideanizingFlow
from steal.rmpflow.controllers import (
    LogCoshPotential, NaturalGradientDescentMomentumController)
from steal.rmpflow.rmp_tree import RmpTreeNode


class Diffeomorphism(pl.LightningModule):
    """
    Base class for learning diffeomorphism based on Euclideanizing Flows.
    """

    def __init__(self, n_dim, link_names, scalings, translations, leaf_goals,
                 params):
        super().__init__()

        self.link_names = link_names

        self.lagrangian_vel_nets = self.add_link_leaf_nodes(
            n_dim, scalings, translations, leaf_goals, params)

    def add_link_leaf_nodes(self, workspace_dims, scalings, translations,
                            leaf_goals, params):
        """Generate an RMPTreeNode for each link in the robot."""

        # Each network in this list predicts momentum in the Lagrangian mechanics sense.
        # We don't need a network for the robot base, hence None.
        lagrangian_vel_nets = [None]

        for n, _ in enumerate(self.link_names):
            scaling = scalings[n]
            translation = translations[n]
            leaf_goal = leaf_goals[n]

            # Taskmap which maps to a learned latent space
            # where the dynamical system is on a Euclidean manifold.
            latent_taskmap = EuclideanizingFlow(
                n_inputs=workspace_dims,
                n_blocks=params.n_blocks_flow,
                n_hidden=params.n_hidden_flow,
                s_act=params.s_act_flow,
                t_act=params.t_act_flow,
                sigma=params.sigma_flow,
                flow_type=params.flow_type,
                coupling_network_type=params.coupling_network_type,
                goal=leaf_goal,
                normalization_scaling=scaling,
                normalization_bias=translation)

            # Create an RMP for the latent task space
            latent_metric_fn = MetricCholNet(n_dims=workspace_dims,
                                             n_hidden_1=params.n_hidden_1,
                                             n_hidden_2=params.n_hidden_2,
                                             return_cholesky=False)
            latent_potential_fn = LogCoshPotential()
            latent_rmp = NaturalGradientDescentMomentumController(
                G=latent_metric_fn, del_Phi=latent_potential_fn.grad)

            leaf_rmp = RmpTreeNode(n_dim=workspace_dims,
                                   order=params.rmp_order,
                                   return_natural=True)

            # Create a child node for the latent space of this link
            latent_space_node = leaf_rmp.add_task_space(
                task_map=latent_taskmap)
            # Add the RMP to the newly created child node.
            latent_space_node.add_rmp(latent_rmp)

            lagrangian_vel_nets.append(leaf_rmp)

        return lagrangian_vel_nets


class LatentTaskMapNetwork(Diffeomorphism):
    """Lightning module to learn a latent task space map."""

    def __init__(self,
                 index,
                 n_dim,
                 link_names,
                 scalings,
                 translations,
                 leaf_goals,
                 params,
                 loss_clip=1e3):
        super().__init__(n_dim, link_names, scalings, translations, leaf_goals,
                         params)

        # Train only the network specified by `index`.
        self.model = self.lagrangian_vel_nets[index + 1]

        self.loss = nn.SmoothL1Loss()

        self.loss_clip = loss_clip

    def training_step(self, batch):
        x, y = batch
        # forward pass
        y_pred, _ = self.model(x)
        # compute loss
        loss = self.loss(y_pred, y)

        # Logging to TensorBoard by default
        self.log("train_loss", loss)

        if loss > self.loss_clip:
            # print('loss too large, skip')
            return 0.0

        # if epoch - best_train_epoch >= stop_threshold:
        #     break

        # if train_loss < best_train_loss:
        #     best_train_epoch = epoch
        #     best_train_loss = train_loss
        #     best_model = copy.deepcopy(model)

        return loss

    def set_return_natural(self, flag: bool):
        """Set the return_natural parameter for the leaf RMP."""
        self.model.return_natural = flag

    def configure_optimizers(self, learning_rate=1e-4, weight_decay=1e-6):
        """
        Configure the use of the optimization method for the network.
        """
        learnable_params = self.model.edges[0].task_map.parameters()
        optimizer = optim.Adam(learnable_params,
                               lr=learning_rate,
                               weight_decay=weight_decay)
        return optimizer


class ContextMomentumNetwork(Diffeomorphism):
    """
    Lightning module to train the multiple leaves
    of EuclideanizingFlow as a Langrangian network.
    """

    def __init__(self,
                 n_dim,
                 cspace_dim,
                 link_names,
                 scalings,
                 translations,
                 leaf_goals,
                 params,
                 loss_clip=1e3):
        super().__init__(n_dim, link_names, scalings, translations, leaf_goals,
                         params)

        self.model = ContextMomentumNet(self.lagrangian_vel_nets,
                                        cspace_dim,
                                        metric_scaling=[1.] *
                                        len(self.lagrangian_vel_nets))

        self.loss = LagrangianMomentumLoss(criterion=nn.SmoothL1Loss())

        self.loss_clip = loss_clip

    def load_models(self, lagrangian_vel_nets):
        """Load pretrained Lagrangian Velocity Networks."""
        self.lagrangian_vel_nets = lagrangian_vel_nets
        self.model.lagrangian_vel_nets = lagrangian_vel_nets

    def training_step(self, batch):
        """
        `training_step` defines the train loop.
        It is independent of forward
        """
        x, y = batch
        # forward pass
        y_pred = self.model(**x)
        # compute loss
        loss = self.loss(y_pred, y)

        # Logging to TensorBoard by default
        self.log("train_loss", loss)

        if loss > self.loss_clip:
            # print('loss too large, skip')
            return 0.0

        # if epoch - best_train_epoch >= stop_threshold:
        #     break

        # if train_loss < best_train_loss:
        #     best_train_epoch = epoch
        #     best_train_loss = train_loss
        #     best_model = copy.deepcopy(model)

        return loss

    def configure_optimizers(self, learning_rate=1e-4, weight_decay=1e-6):
        """
        Configure the use of the optimization method for the network.
        """
        # Collect the learnable parameters from the RMP leaves.
        learnable_params = []
        for leaf_rmp in self.model.lagrangian_vel_nets:
            if leaf_rmp is None:
                continue

            latent_metric_params = leaf_rmp.edges[0].child_node.parameters()
            learnable_params.append(latent_metric_params)

        learnable_params = chain(*learnable_params)
        optimizer = optim.Adam(learnable_params,
                               lr=learning_rate,
                               weight_decay=weight_decay)
        return optimizer
