import argparse
import copy
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from matplotlib import animation
from matplotlib import pyplot as plt
from torch.utils.data import ConcatDataset, DataLoader

from steal.datasets import get_dataset_list, get_lasa_data
from steal.learning import (LatentTaskMapNetwork, get_flow_params,
                            get_leaf_goals, get_params, train_combined2,
                            train_independent2)
from steal.rmpflow import RmpTreeNode
from steal.robot import Robot, create_rmp_tree
from steal.utils import (generate_trajectories, plot_robot_2D, plot_traj_2D,
                         plot_traj_time)

logging.getLogger("lightning").setLevel(logging.ERROR)

logger.remove()
logger.add(sys.stderr, level="ERROR")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    return parser.parse_args()


def get_task_space_models2(link_names, scalings, translations, leaf_goals,
                           params) -> list[LatentTaskMapNetwork]:
    """
    Get the latent task space model for all the leaves
    and the model which predicts the momentum for
    all the RMP leaf nodes together.
    """

    lagrangian_vel_nets = [None]
    for n, link_name in enumerate(link_names):
        rmp_leaf = LatentTaskMapNetwork(n_dims=params.workspace_dim,
                                        link_name=link_name,
                                        scaling=scalings[n],
                                        translation=translations[n],
                                        leaf_goal=leaf_goals[n],
                                        params=params)
        lagrangian_vel_nets.append(rmp_leaf)
    return lagrangian_vel_nets


def add_learned_nodes(robot, root: RmpTreeNode, link_names, lagrangian_nets):
    """Add the trained RMP nodes to the RMP Tree"""
    for n, link_name in enumerate(link_names):
        link_task_map = robot.get_task_map(target_link=link_name)
        leaf_node = root.add_task_space(link_task_map, name=link_name)
        leaf_rmp = lagrangian_nets[n + 1]
        leaf_rmp.eval()
        leaf_node.add_rmp(leaf_rmp)
    return root


def plot(root: RmpTreeNode, robot, link_names, time_list, joint_traj_list, dt,
         demo_test, cspace_dim, params):
    root.eval()
    # Rolling out
    T = time_list[demo_test][-1]
    t0 = 0.
    print('--------------------------------------------')
    print(f'Rolling out for demo: {demo_test}, from t0: {t0} to T:{T}')
    print('--------------------------------------------')

    ref_cspace_traj = torch.from_numpy(joint_traj_list[demo_test]).to(
        dtype=torch.get_default_dtype()
    )  # dataset_list[demo_test].state.numpy().T
    q0 = copy.deepcopy(ref_cspace_traj[int(t0 / dt), :].reshape(1, -1))

    # q0[0, 0] = q0[0, 0] + 10.
    # q0[0, 1] = q0[0, 1] + 3.

    # dt = 0.1
    root.return_natural = False

    cspace_traj = generate_trajectories(root,
                                        x_init=q0,
                                        t_init=0.,
                                        t_final=T + dt,
                                        t_step=dt,
                                        order=params.rmp_order,
                                        method='euler',
                                        return_label=False)

    time_traj = np.arange(0., T + dt, dt)

    cspace_traj = cspace_traj.squeeze()

    ref_link_trajs = []
    link_trajs = []
    for link_name in link_names:
        link_task_map = robot.get_task_map(target_link=link_name)

        # Finding the reference trajectory (with the bias!)
        ref_link_traj = link_task_map.psi(
            ref_cspace_traj
        )  # NOTE: We are not composing with the tracked frame task map!
        ref_link_trajs.append(ref_link_traj)

        link_traj = link_task_map.psi(cspace_traj)
        link_trajs.append(link_traj)

    # ------------------------------------------
    # Plotting
    # diff_link_traj = link_trajs[1] - link_trajs[0]
    # ref_diff_link_traj = ref_link_trajs[1] - ref_link_trajs[0]

    for i, link_name in enumerate(link_names):
        fig = plt.figure()
        plot_traj_time(time_traj,
                       link_trajs[i].numpy(),
                       '--',
                       'b',
                       title=link_name)
        plot_traj_time(time_list[demo_test],
                       ref_link_trajs[i].numpy(),
                       ':',
                       'r',
                       title=link_name)
        # plot_traj_time(time_traj, diff_link_traj.numpy(), '--', 'b')
        # plot_traj_time(time_list[demo_test], ref_diff_link_traj.numpy(), ':', 'r')

    fig = plt.figure()
    plt.axis(np.array([-20, 70, -20, 70]))
    plt.gca().set_aspect('equal', 'box')

    for i in range(len(link_names)):
        plot_traj_2D(link_trajs[i].numpy(), '--', 'b', order=1)
        plot_traj_2D(ref_link_trajs[i].numpy(), ':', 'r', order=1)

    link_order = [
        (i, j)
        for i, j in zip(range(2, robot.num_links), range(3, robot.num_links))
    ]
    handles, = plot_robot_2D(robot, q0, 2, link_order=link_order)

    h, = plot_robot_2D(robot,
                       cspace_traj[0, :cspace_dim],
                       2,
                       link_order=link_order)
    plot_robot_2D(robot,
                  cspace_traj[0, :cspace_dim],
                  2,
                  handle_list=h,
                  link_order=link_order)

    def init():
        return h,

    def animate(i):
        nsteps = cspace_traj.shape[0]
        handle, = plot_robot_2D(robot,
                                cspace_traj[i % nsteps, :cspace_dim],
                                2,
                                h,
                                link_order=link_order)
        return handle,

    ani = animation.FuncAnimation(fig,
                                  animate,
                                  init_func=init,
                                  interval=30,
                                  blit=False)

    plt.show()


def get_robot(params, robot_name='five_point_planar_robot'):
    package_path = Path(__file__).parent.parent
    urdf_dir = package_path / 'urdf'
    robot_urdf = robot_name + '.urdf'
    urdf_path = urdf_dir / robot_urdf
    robot = Robot(urdf_path=urdf_path, workspace_dim=params.workspace_dim)
    return robot


def main():
    """Main runner"""

    params = get_params()

    package_path = Path(__file__).parent.parent
    # data_root = package_path / 'data'

    # Creating the robot model
    robot = get_robot(params)
    cspace_dim = robot.cspace_dim

    link_names = ('link4', 'link8')

    time_list, joint_traj_list, dt, n_demos = get_lasa_data(params,
                                                            cspace_dim,
                                                            data_name="Angle")

    #TODO remove this duplicate
    leaf_goals, _ = get_leaf_goals(robot, link_names, joint_traj_list)

    root = create_rmp_tree(cspace_dim=cspace_dim, rmp_order=params.rmp_order)

    dataset_list = get_dataset_list(n_demos,
                                    joint_traj_list=joint_traj_list,
                                    cspace_dim=cspace_dim,
                                    dt=dt,
                                    root=root,
                                    robot=robot,
                                    link_names=link_names,
                                    rmp_order=params.rmp_order)
    train_dataset = ConcatDataset(dataset_list)

    scalings, translations, leaf_goals = get_flow_params(
        link_names, dataset_list, robot, joint_traj_list)

    task_map_nets: list[LatentTaskMapNetwork] = get_task_space_models2(
        link_names, scalings, translations, leaf_goals, params)
    task_map_nets = train_independent2(task_map_nets, link_names, dataset_list,
                                       params)

    lagrangian_vel_nets = []
    for net in task_map_nets:
        if net is None:
            lagrangian_vel_nets.append(None)
        else:
            lagrangian_vel_nets.append(net.leaf_rmp)

    train_loader = DataLoader(train_dataset,
                              num_workers=8,
                              batch_size=len(train_dataset),
                              persistent_workers=True,
                              pin_memory=True)
    context_net = train_combined2(lagrangian_vel_nets=lagrangian_vel_nets,
                                  train_loader=train_loader,
                                  max_epochs=params.n_epochs,
                                  cspace_dim=cspace_dim)

    task_map_nets = lagrangian_vel_nets

    # task_map_nets: list[RmpTreeNode] = get_task_space_models(
    #     link_names, scalings, translations, leaf_goals, params)

    # train_independent(task_map_nets, link_names, dataset_list, params)
    # train_combined(task_map_nets,
    #                train_dataset,
    #                link_names,
    #                cspace_dim,
    #                params,
    #                load_pretrained_models=False,
    #                models_path="")

    print("Training complete!")

    root = add_learned_nodes(robot, root, link_names, task_map_nets)

    demo_test = 2
    plot(root, robot, link_names, time_list, joint_traj_list, dt, demo_test,
         cspace_dim, params)


if __name__ == "__main__":
    main()
