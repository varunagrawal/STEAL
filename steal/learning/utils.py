"""Utilities to help in training."""

import copy
import os
import time
from itertools import chain

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from steal.datasets import find_mean_goal
from steal.learning import (ContextMomentumNet, ContextMomentumNetwork,
                            EuclideanizingFlow, LagrangianMomentumLoss,
                            LatentTaskMapNetwork, LogCoshPotential,
                            MetricCholNet,
                            NaturalGradientDescentMomentumController)
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
                          params) -> list[RmpTreeNode]:
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


def train(model,
          loss_fn,
          opt,
          train_dataset,
          n_epochs=500,
          batch_size=None,
          shuffle=True,
          clip_gradient=True,
          clip_value_grad=0.1,
          clip_weight=False,
          clip_value_weight=2,
          log_freq=5,
          logger=None,
          loss_clip=1e3,
          stop_threshold=float('inf')):
    '''
    train the torch model with the given parameters
    :param model (torch.nn.Module): the model to be trained
    :param loss_fn (callable): loss = loss_fn(y_pred, y_target)
    :param opt (torch.optim): optimizer
    :param x_train (torch.Tensor): training data (position/position + velocity)
    :param y_train (torch.Tensor): training label (velocity/control)
    :param n_epochs (int): number of epochs
    :param batch_size (int): size of minibatch, if None, train in batch
    :param shuffle (bool): whether the dataset is reshuffled at every epoch
    :param clip_gradient (bool): whether the gradients are clipped
    :param clip_value_grad (float): the threshold for gradient clipping
    :param clip_weight (bool): whether the weights are clipped (not implemented)
    :param clip_value_weight (float): the threshold for weight clipping (not implemented)
    :param log_freq (int): the frequency for printing loss and saving results on tensorboard
    :param logger: the tensorboard logger
    :return: None
    '''

    # if batch_size is None, train in batch
    n_samples = len(train_dataset)
    if batch_size is None:
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=n_samples,
                                  shuffle=shuffle)
        batch_size = n_samples
    else:
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle)

    # record time elasped
    ts = time.time()

    if hasattr(loss_fn, 'reduction'):
        if loss_fn.reduction == 'mean':
            mean_flag = True
        else:
            mean_flag = False
    else:
        mean_flag = True  # takes the mean by default

    best_train_loss = float('inf')
    best_train_epoch = 0
    best_model = model

    # train the model
    model.train()
    for epoch in range(n_epochs):
        # iterate over minibatches
        train_loss = 0.
        for x_batch, y_batch in train_loader:
            # forward pass
            if isinstance(x_batch, torch.Tensor):
                y_pred = model(x_batch)
            elif isinstance(x_batch, dict):
                y_pred = model(**x_batch)
            else:
                raise ValueError
            # compute loss
            loss = loss_fn(y_pred, y_batch)
            train_loss += loss.item()

            if loss > loss_clip:
                print('loss too large, skip')
                continue

            # backward pass
            opt.zero_grad()
            loss.backward()

            # clip gradient based on norm
            if clip_gradient:
                # torch.nn.utils.clip_grad_value_(
                #     model.parameters(),
                #     clip_value_grad
                # )
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               clip_value_grad)
            # update parameters
            opt.step()

        if mean_flag:  # fix for taking mean over all data instead of mini batch!
            train_loss = float(batch_size) / float(n_samples) * train_loss

        if epoch - best_train_epoch >= stop_threshold:
            break

        if train_loss < best_train_loss:
            best_train_epoch = epoch
            best_train_loss = train_loss
            best_model = copy.deepcopy(model)

        # report loss in command line and tensorboard every log_freq epochs
        if epoch % log_freq == (log_freq - 1):
            print(
                '    Epoch [{}/{}]: current loss is {}, time elapsed {} second'
                .format(epoch + 1, n_epochs, train_loss,
                        time.time() - ts))

            if logger is not None:
                info = {'Training Loss': train_loss}

                # log scalar values (scalar summary)
                for tag, value in info.items():
                    logger.scalar_summary(tag, value, epoch + 1)

                # log values and gradients of the parameters (histogram summary)
                for tag, value in model.named_parameters():
                    tag = tag.replace('.', '/')
                    logger.histo_summary(tag,
                                         value.data.cpu().numpy(), epoch + 1)
                    logger.histo_summary(tag + '/grad',
                                         value.grad.data.cpu().numpy(),
                                         epoch + 1)
    return best_model, best_train_loss


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


def train_independent2(task_map_nets: list[LatentTaskMapNetwork], link_names,
                       dataset_list, params):
    """
    The metric is kept fixed (identity) while the taskmap is learned!
    """
    print('Training taskmaps only! Using identity latent space metric!')
    for n, link_name in enumerate(link_names):
        print(f"\n\n----- Training Flow for {link_name}")
        x_train = torch.cat(
            [dataset.q_leaf_list[n + 1] for dataset in dataset_list], dim=0)
        J_train = torch.cat(
            [dataset.J_list[n + 1] for dataset in dataset_list], dim=0)
        qd_train = torch.cat([dataset_.qd_config for dataset_ in dataset_list],
                             dim=0)
        xd_train = torch.bmm(qd_train.unsqueeze(1),
                             J_train.permute(0, 2, 1)).squeeze(1)
        leaf_dataset = TensorDataset(x_train, xd_train)

        train_loader = DataLoader(leaf_dataset,
                                  num_workers=8,
                                  batch_size=len(leaf_dataset),
                                  persistent_workers=True,
                                  pin_memory=True)

        model = task_map_nets[n + 1]

        model.set_return_natural(False)

        checkpoint_callback = ModelCheckpoint(
            dirpath=f"checkpoints/{model.leaf_rmp.name}")

        trainer = pl.Trainer(max_epochs=params.n_epochs,
                             log_every_n_steps=1,
                             callbacks=[checkpoint_callback])
        trainer.fit(model=model, train_dataloaders=train_loader)

        model.set_return_natural(True)

        task_map_nets[n + 1] = trainer.model

    return task_map_nets


def train_combined2(lagrangian_vel_nets: list[LatentTaskMapNetwork],
                    train_loader: DataLoader, max_epochs,
                    cspace_dim) -> ContextMomentumNetwork:
    """Train the Riemannian metrics now that the taskmaps have been learned."""
    print('Training metrics only!')

    model = ContextMomentumNetwork(lagrangian_vel_nets=lagrangian_vel_nets,
                                   n_dims=cspace_dim,
                                   scalings=[1.] * len(lagrangian_vel_nets))

    trainer = pl.Trainer(max_epochs=max_epochs, log_every_n_steps=1)
    trainer.fit(model=model, train_dataloaders=train_loader)
    return trainer.model
