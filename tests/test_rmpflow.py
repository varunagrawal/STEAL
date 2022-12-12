"""Tests for RMPFlow on various datasets."""

import unittest

import torch

from steal.rmpflow import (NaturalGradientDescentForceController,
                           ObstacleAvoidanceForceController,
                           SphereDistanceTaskMap, TargetForceControllerUniform,
                           TargetTaskMap)
from steal.rmpflow.rmp_tree import RmpTreeNode


class TestRMPFlow(unittest.TestCase):

    def test_attractor_obstacle(self):
        """
        Simple test to check if RMPFlow can go
        towards a goal point while avoiding an obstacle.
        """
        cspace_dim = 2
        work_space_dim = 2
        obstacle_center = torch.zeros(1, work_space_dim)
        obstacle_radius = 1.0
        obs_alpha = 1e-5

        x_goal = torch.tensor([-3, 3]).reshape(1, -1)

        # setting up tree
        root = RmpTreeNode(n_dim=cspace_dim, name='root')

        # attractor
        target_taskmap = TargetTaskMap(x_goal)
        controller = TargetForceControllerUniform(damping_gain=2.,
                                                  proportional_gain=1.0,
                                                  norm_robustness_eps=1e-12,
                                                  alpha=1.0,
                                                  w_u=10.0,
                                                  w_l=1.0,
                                                  sigma=1.0)
        leaf = root.add_task_space(task_map=target_taskmap, name='target')
        leaf.add_rmp(controller)

        # obstacle
        obstacle_taskmap = SphereDistanceTaskMap(n_inputs=work_space_dim,
                                                 radius=obstacle_radius,
                                                 center=obstacle_center)
        obstacle_controller = ObstacleAvoidanceForceController(
            proportional_gain=obs_alpha, damping_gain=0.0, epsilon=0.2)
        leaf = root.add_task_space(task_map=obstacle_taskmap, name='obstacle')
        leaf.add_rmp(obstacle_controller)

        # -----------------------------------------------
        # rolling out

        root.eval()

        q = torch.tensor([2.5, -3.2]).reshape(1, -1)
        qd = torch.tensor([-1.0, 1.0]).reshape(1, -1)

        T = 20.
        dt = 0.01

        traj = q
        tt = torch.tensor([0.0])

        N = int(T / dt)
        for i in range(1, N):
            qdd, _ = root.eval_canonical(q, qd)
            # Apply Euler integration to get position and velocity.
            q = q + dt * qd
            qd = qd + dt * qdd
            traj = torch.cat((traj, q), dim=0)
            tt = torch.cat((tt, torch.tensor([dt * i])))

        start_point = torch.tensor([2.5000, -3.2000])
        end_point = torch.tensor([-2.703963756561, 2.213977813721])
        torch.testing.assert_close(traj[0], start_point)
        torch.testing.assert_close(traj[-1], end_point)


class TestForceController(unittest.TestCase):
    """Tests for the force_controllers module."""

    def test_natural_gradient_descent_force_controller(self):
        d = 2
        G = lambda x, xd: torch.einsum('bi,bj->bij', x, x * xd)
        B = torch.zeros(d, d)
        Phi = lambda x: (0.5 * torch.norm(x, dim=1)**2).reshape(-1, 1)
        gds = NaturalGradientDescentForceController(G=G, B=B, Phi=Phi)
        x = torch.ones(2, d)
        xd = torch.ones(2, d)
        f, M = gds(x, xd)