"""Utilities to help in training."""

import os
import time
from itertools import chain

import torch
from torch import nn, optim
from torch.utils.data import TensorDataset

from steal.datasets import find_mean_goal
from steal.learning import (ContextMomentumNet, EuclideanizingFlow,
                            LagrangianMomentumLoss, LogCoshPotential,
                            MetricCholNet,
                            NaturalGradientDescentMomentumController)
from steal.learning.train_utils import train
from steal.rmpflow import RmpTreeNode


class Params(object):
    """Helper class to record various parameters."""

    def __init__(self, **kwargs):
        super(Params, self).__init__()
        self.__dict__.update(kwargs)


def get_params():
    """Get learning parameters"""
    params = Params(
        n_hidden_1=
        128,  # number of hidden units in hidden layer 1 in metric cholesky net
        n_hidden_2=
        64,  # number of hidden units in hidden layer 2 in metric cholseky net
        n_blocks_flow=10,  # number of blocks in diffeomorphism net
        n_hidden_flow=
        200,  # number of hidden units in the two hidden layers in diffeomorphism net
        s_act_flow=
        'elu',  # (fcnn only) activation fcn in scaling network of coupling layers
        t_act_flow=
        'elu',  # (fcnn only) activation fcn in scaling network of coupling layers
        sigma_flow=0.45,  # (for rfnn only) length scale
        flow_type='realnvp',  # (realnvp/glow) architecture of flow
        coupling_network_type=
        'rfnn',  # (rfnn/fcnn) coupling network parameterization
        eps=1e-12,

        # Optimization params
        n_epochs=500,  # number of epochs
        stop_threshold=250,
        batch_size=None,  # size of minibatches
        learning_rate=0.0001,  # learning rate
        weight_decay=1e-6,  # weight for regularization

        # pre-processing params
        downsample_rate=4,  # data downsampling
        smoothing_window_size=
        25,  # smoothing window size for savizky golay filter
        start_cut=10,  # how much data to cut at the beginning
        end_cut=5,  # how much data to cut at the end
        workspace_dim=2,
        joint_damping_gain=1e-4,
        rmp_order=1,
    )
    return params


def get_leaf_goals(robot, link_names, joint_traj_list):
    """Get the mean goal for each trajectory for the
    leaves of the RMP tree (e.g. the link task space).

    Args:
        robot (Robot): The robot class for getting link info.
        link_names (List[str]): List of links for which to find the mean goal.
        joint_traj_list (List): List of joint trajectories.

    Returns:
        dict, dict: The dict of the mean goals and the mean goal biases.
    """
    leaf_goals = {}
    leaf_goal_biases = {}
    for link_name in link_names:
        mean_goal, goal_biases = find_mean_goal(
            robot,
            joint_traj_list,
            link_name,
            base_to_tracked_frame_transforms=None)

        leaf_goals[link_name] = mean_goal
        leaf_goal_biases[link_name] = goal_biases

    return leaf_goals, leaf_goal_biases


def get_flow_params(link_names, dataset_list, robot, joint_traj_list):
    """Get the scaling, translation and leaf goals for each link."""
    scaling_list, translation_list, leaf_goals = [], [], []
    for n, link_name in enumerate(link_names):
        # for tracked_frame in tracked_frames:
        x_train = torch.cat(
            [dataset_.q_leaf_list[n + 1] for dataset_ in dataset_list], dim=0)

        minx = torch.min(x_train, dim=0)[0].reshape(1, -1)
        maxx = torch.max(x_train, dim=0)[0].reshape(1, -1)
        scaling = 1. / (maxx - minx)
        translation = -minx / (maxx - minx) - 0.5

        scaling_list.append(scaling)
        translation_list.append(translation)

        leaf_goal, _ = find_mean_goal(robot,
                                      joint_traj_list,
                                      link_name,
                                      base_to_tracked_frame_transforms=None)
        leaf_goals.append(
            torch.from_numpy(leaf_goal).to(dtype=torch.get_default_dtype()))

    return scaling_list, translation_list, leaf_goals


def get_task_space_models(link_names, scalings, translations, leaf_goals,
                          params):
    """
    Get the latent task space model for all the leaves 
    and the model which predicts the momentum for
    all the RMP leaf nodes together.
    """

    lagrangian_vel_nets = [None]
    for n, link_name in enumerate(link_names):
        scaling = scalings[n]
        translation = translations[n]
        leaf_goal = leaf_goals[n]

        leaf_rmp = RmpTreeNode(n_dim=params.workspace_dim,
                               name=link_name,
                               order=params.rmp_order,
                               return_natural=True)
        latent_taskmap = EuclideanizingFlow(
            n_inputs=params.workspace_dim,
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

        latent_space = leaf_rmp.add_task_space(task_map=latent_taskmap)

        latent_pot_fn = LogCoshPotential()
        latent_metric_fn = MetricCholNet(n_dims=params.workspace_dim,
                                         n_hidden_1=params.n_hidden_1,
                                         n_hidden_2=params.n_hidden_2,
                                         return_cholesky=False)

        latent_rmp = NaturalGradientDescentMomentumController(
            G=latent_metric_fn, del_Phi=latent_pot_fn.grad)
        latent_space.add_rmp(latent_rmp)

        lagrangian_vel_nets.append(leaf_rmp)

    return lagrangian_vel_nets


def train_combined(lagrangian_nets, dataset, link_names, cspace_dim, params,
                   load_pretrained_models: bool, models_path: str):
    """Train the RMP metric using all the leaves together."""
    # NOTE: The taskmap is kept fixed, while the metric is learned!
    print('Training metrics only!')
    learnable_params = []

    for n, link_name in enumerate(link_names):
        leaf_rmp = lagrangian_nets[n + 1]

        if load_pretrained_models:
            model_filename = f'model_{link_name}_independent.pt'
            leaf_rmp.load_state_dict(
                torch.load(os.path.join(models_path, model_filename)))

        latent_metric_params = leaf_rmp.edges[0].child_node.parameters()
        learnable_params.append(latent_metric_params)

    learnable_params = chain(*learnable_params)
    model = ContextMomentumNet(lagrangian_nets,
                               cspace_dim,
                               metric_scaling=[1. for net in lagrangian_nets])
    model.train()

    criterion = nn.SmoothL1Loss()
    # criterion = nn.MSELoss()
    loss_fn = LagrangianMomentumLoss(criterion=criterion)
    optimizer = optim.Adam(learnable_params,
                           lr=params.learning_rate,
                           weight_decay=params.weight_decay)

    t_start = time.time()
    best_model, best_traj_loss = train(model=model,
                                       loss_fn=loss_fn,
                                       opt=optimizer,
                                       train_dataset=dataset,
                                       n_epochs=params.n_epochs,
                                       batch_size=params.batch_size,
                                       stop_threshold=params.stop_threshold)
    print(f'time elapsed: {time.time() - t_start} seconds')
    print('\n')
    return best_model


def train_independent(lagrangian_nets, link_names, dataset_list, params):
    # NOTE: The metric is kept fixed (identity) while the taskmap is learned!
    print('Training taskmaps only, with identity latent space metric!')
    for n, _ in enumerate(link_names):
        x_train = torch.cat(
            [dataset_.q_leaf_list[n + 1] for dataset_ in dataset_list], dim=0)
        J_train = torch.cat(
            [dataset_.J_list[n + 1] for dataset_ in dataset_list], dim=0)
        qd_train = torch.cat([dataset_.qd_config for dataset_ in dataset_list],
                             dim=0)
        xd_train = torch.bmm(qd_train.unsqueeze(1),
                             J_train.permute(0, 2, 1)).squeeze(1)
        leaf_dataset = TensorDataset(x_train, xd_train)

        leaf_rmp = lagrangian_nets[n + 1]
        leaf_rmp.return_natural = False
        leaf_rmp.train()

        # for the independent version, only train the taskmap
        learnable_params = leaf_rmp.edges[0].task_map.parameters()

        criterion = nn.SmoothL1Loss()
        optimizer = optim.Adam(learnable_params,
                               lr=params.learning_rate,
                               weight_decay=params.weight_decay)

        t_start = time.time()
        best_model, best_traj_loss = train(
            model=leaf_rmp,
            loss_fn=criterion,
            opt=optimizer,
            train_dataset=leaf_dataset,
            n_epochs=params.n_epochs,
            batch_size=params.batch_size,
            stop_threshold=params.stop_threshold)

        print(f'time elapsed: {time.time() - t_start} seconds')
        print('\n')
        leaf_rmp.return_natural = True

    return best_model, best_traj_loss