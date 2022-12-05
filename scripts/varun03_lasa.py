import argparse
import logging
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset

from steal.datasets import (ContextMomentumDataset, find_mean_goal, lasa,
                            preprocess_dataset)
from steal.learning import ContextMomentumNetwork, LatentTaskMapNetwork, Params
from steal.rmpflow import (DampingMomemtumController, DimSelectorTaskMap,
                           RmpTreeNode)
from steal.rmpflow.kinematics.robot import Robot

logging.getLogger("lightning").setLevel(logging.ERROR)

import sys

from loguru import logger

logger.remove()
logger.add(sys.stderr, level="ERROR")


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


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    return parser.parse_args()


def get_lasa_dataset(params, cspace_dim, data_name='Sshape'):
    data = getattr(lasa.DataSet, data_name)
    data = data.demos

    n_demos = len(data)
    # n_dims = data[0].pos.shape[0]

    dt = data[0].t[0, 1] - data[0].t[0, 0]
    dt = np.round(dt * params.downsample_rate, 2)

    demo_traj_list = [data[i].pos.T for i in range(n_demos)]

    torch_traj_datasets = preprocess_dataset(
        demo_traj_list,
        dt=dt,
        start_cut=params.start_cut,
        end_cut=params.end_cut,
        downsample_rate=params.downsample_rate,
        smoothing_window_size=params.smoothing_window_size,
        vel_thresh=1.,
        goal_at_origin=True)

    # ---------------------------------------------------
    joint_traj_list = [
        traj_dataset.tensors[0].numpy() for traj_dataset in torch_traj_datasets
    ]
    # add fake zeros for the rest of joints
    joint_traj_list = [
        np.concatenate((traj, torch.zeros((traj.shape[0], cspace_dim - 2))),
                       axis=1) for traj in joint_traj_list
    ]

    # ---------------------------------------------------
    time_list = [
        np.arange(0., joint_traj.shape[0]) * dt
        for joint_traj in joint_traj_list
    ]

    return time_list, joint_traj_list, dt


def create_rmp_tree(cspace_dim, params):
    """
    Create initial RMP tree with the root node and damping nodes for each joint.
    """
    print("Setting up tree")
    root = RmpTreeNode(n_dim=cspace_dim,
                       name="cspace_root",
                       order=params.rmp_order,
                       return_natural=True)
    root.eval()

    # --------------------------------
    print("Adding damping to each joint")

    joint_damping_gain = 1e-4

    for i in range(cspace_dim):
        joint_task_map = DimSelectorTaskMap(n_inputs=cspace_dim,
                                            selected_dims=i)
        joint_node = root.add_task_space(joint_task_map, name="joint" + str(i))
        damping_rmp = DampingMomemtumController(
            damping_gain=joint_damping_gain)

        joint_node.add_rmp(damping_rmp)

    return root


def get_training_dataset(joint_traj_list, cspace_dim, link_names, robot, dt,
                         root, params):
    """Get the training dataset."""
    demo_goal_list = []
    dataset_list = []
    for cspace_traj in joint_traj_list:
        cspace_vel = np.diff(cspace_traj, axis=0) / dt
        cspace_vel = np.concatenate((cspace_vel, np.zeros((1, cspace_dim))),
                                    axis=0)

        cspace_traj_torch = torch.from_numpy(cspace_traj).to(
            torch.get_default_dtype())
        cspace_vel_torch = torch.from_numpy(cspace_vel).to(
            torch.get_default_dtype())

        N = cspace_traj_torch.shape[0]

        x_list = []
        J_list = []
        p_list = []
        M_list = []

        # NOTE: Taking the original goal! (WITHOUT CUTTING!)
        cspace_goal = torch.from_numpy(cspace_traj[-1].reshape(1, -1)).to(
            torch.get_default_dtype())
        demo_goal_list.append(cspace_goal)

        # Get momentum and metric (since it is of order 1)
        p, M = root(x=cspace_traj_torch)

        x_list.append(cspace_traj_torch)
        J_list.append(torch.eye(cspace_dim).repeat((N, 1, 1)))
        p_list.append(p)
        M_list.append(M)

        for link_name in link_names:
            link_task_map = robot.get_task_map(target_link=link_name)

            link_task_map.eval()
            x, J = link_task_map(cspace_traj_torch, order=params.rmp_order)

            x_list.append(x)
            J_list.append(J)
            p_list.append(torch.zeros(N, 1))
            M_list.append(torch.zeros(N, 1))

        dataset = ContextMomentumDataset(cspace_traj_torch, cspace_vel_torch,
                                         x_list, J_list, p_list, M_list)

        dataset_list.append(dataset)

    return dataset_list


def get_leaf_goal(link_name, robot, joint_traj_list):
    """
    Compute the mean goal location for a link
    from all the trajectories in the demonstrations.
    """

    mean_goal, goal_biases = find_mean_goal(
        robot,
        joint_traj_list,
        link_name,
        base_to_tracked_frame_transforms=None)

    return mean_goal, goal_biases


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

        leaf_goal, _ = get_leaf_goal(link_name, robot, joint_traj_list)
        leaf_goals.append(
            torch.from_numpy(leaf_goal).to(dtype=torch.get_default_dtype()))

    return scaling_list, translation_list, leaf_goals


def get_models(workspace_dims, cspace_dim, link_names, scalings, translations,
               leaf_goals, params):
    """
    Get the latent task space model for all the leaves 
    and the model which predicts the momentum for
    all the RMP leaf nodes together.
    """

    latent_models = [None]
    for index, _ in enumerate(link_names):
        rmp_leaf = LatentTaskMapNetwork(index=index,
                                        n_dim=workspace_dims,
                                        link_names=link_names,
                                        scalings=scalings,
                                        translations=translations,
                                        leaf_goals=leaf_goals,
                                        params=params)
        latent_models.append(rmp_leaf)

    context_model = ContextMomentumNetwork(n_dim=workspace_dims,
                                           cspace_dim=cspace_dim,
                                           link_names=link_names,
                                           scalings=scalings,
                                           translations=translations,
                                           leaf_goals=leaf_goals,
                                           params=params)
    return latent_models, context_model


def train_independent(model: LatentTaskMapNetwork, train_loader: DataLoader,
                      max_epochs):
    """
    The metric is kept fixed (identity) while the taskmap is learned!
    """
    print('Training taskmaps only! Using identity latent space metric!')
    model.set_return_natural(False)
    trainer = pl.Trainer(max_epochs=max_epochs, log_every_n_steps=1)
    trainer.fit(model=model, train_dataloaders=train_loader)
    model.set_return_natural(True)
    return trainer.model


def train_combined(model: ContextMomentumNetwork, train_loader: DataLoader,
                   max_epochs):
    print('Training metrics only!')
    trainer = pl.Trainer(max_epochs=max_epochs, log_every_n_steps=1)
    trainer.fit(model=model, train_dataloaders=train_loader)
    return trainer.model


def main():
    """Main runner"""

    params = get_params()

    package_path = Path(__file__).parent.parent
    # data_root = package_path / 'data'

    # Creating the robot model
    urdf_dir = package_path / 'urdf'
    robot_name = 'five_point_planar_robot'
    robot_urdf = robot_name + '.urdf'
    urdf_path = urdf_dir / robot_urdf
    robot = Robot(urdf_path=urdf_path, workspace_dim=params.workspace_dim)

    cspace_dim = robot.cspace_dim

    link_names = ('link4', 'link8')

    _, joint_traj_list, dt = get_lasa_dataset(params, cspace_dim)

    root = create_rmp_tree(cspace_dim=cspace_dim, params=params)

    dataset_list = get_training_dataset(joint_traj_list=joint_traj_list,
                                        cspace_dim=cspace_dim,
                                        link_names=link_names,
                                        robot=robot,
                                        dt=dt,
                                        root=root,
                                        params=params)
    train_dataset = ConcatDataset(dataset_list)

    scalings, translations, leaf_goals = get_flow_params(
        link_names, dataset_list, robot, joint_traj_list)

    lagrangian_nets, context_net = get_models(params.workspace_dim, cspace_dim,
                                              link_names, scalings,
                                              translations, leaf_goals, params)

    max_epochs = 500

    for n, link_name in enumerate(link_names):
        print(f"Training Flow for {link_name}")
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

        leaf_rmp = lagrangian_nets[n + 1]
        trained_model = train_independent(leaf_rmp,
                                          train_loader,
                                          max_epochs=max_epochs)

        lagrangian_nets[n + 1] = trained_model

    # rmp_nodes = []
    # for index, link_name in enumerate(link_names):
    #     leaf_rmp = get_latent_task_space_model(index, workspace_dims,
    #                                            link_names, dataset_list,
    #                                            leaf_goals, params)

    #     leaf_rmp = train_independent(leaf_rmp, train_loader, max_epochs)

    #     rmp_nodes.append(leaf_rmp)


if __name__ == "__main__":
    main()
