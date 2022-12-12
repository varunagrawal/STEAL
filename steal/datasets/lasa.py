"""Module for handling LASA dataset."""

import numpy as np
import pyLasaDataset as lasa
import torch

from steal.datasets import ContextMomentumDataset
from steal.datasets.preprocess import preprocess_dataset


def get_lasa_data(params, cspace_dim, data_name='Sshape'):
    """
    Get the trajectory data for specified `data_name` 
    in the LASA dataset.
    """
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

    return time_list, joint_traj_list, dt, n_demos


def get_dataset_list(n_demos, joint_traj_list, cspace_dim, dt, root, robot,
                     link_names, rmp_order):
    """Return all the demonstration data as a list of ContextMomentumDatasets.

    Args:
        n_demos (int): The number of demonstrations.
        joint_traj_list (List): List of joint trajectories.
        cspace_dim (int): The configuration space dimension.
        dt (float): The time delta between each trajectory sample.
        root (RmpTreeNode): The root node of the RMP Tree.
        robot (Robot): The robot for which we wish to compute the policy.
        link_names (List[str]): The list of links whose trajectory data we have in `joint_traj_list`.
        rmp_order (int): The derivative order of the RMP.

    Returns:
        The demonstration trajectories as a list of ContextMomentumDatasets.
    """
    demo_goal_list = []
    dataset_list = []
    for demo in range(n_demos):
        cspace_traj = joint_traj_list[demo]
        cspace_vel = np.diff(cspace_traj, axis=0) / dt
        cspace_vel = np.concatenate((cspace_vel, np.zeros((1, cspace_dim))),
                                    axis=0)

        cspace_traj = torch.from_numpy(cspace_traj).to(
            torch.get_default_dtype())
        cspace_vel = torch.from_numpy(cspace_vel).to(torch.get_default_dtype())

        N = cspace_traj.shape[0]

        x_list = []
        J_list = []
        p_list = []
        m_list = []

        # NOTE: Taking the original goal! (WITHOUT CUTTING!)
        cspace_goal = torch.from_numpy(joint_traj_list[demo][-1].reshape(
            1, -1)).to(torch.get_default_dtype())
        demo_goal_list.append(cspace_goal)

        p, m = root(x=cspace_traj)

        x_list.append(cspace_traj)
        J_list.append(torch.eye(cspace_dim).repeat((N, 1, 1)))
        p_list.append(p)
        m_list.append(m)

        for link_name in link_names:
            link_task_map = robot.get_task_map(target_link=link_name)

            link_task_map.eval()
            x, J = link_task_map(cspace_traj, order=rmp_order)

            x_list.append(x)
            J_list.append(J)
            p_list.append(torch.zeros(N, 1))
            m_list.append(torch.zeros(N, 1))

            # local_goal = link_task_map.psi(cspace_goal)

        dataset = ContextMomentumDataset(cspace_traj, cspace_vel, x_list,
                                         J_list, p_list, m_list)

        dataset_list.append(dataset)

    return dataset_list
