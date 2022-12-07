"""Test rmpflow.kinematics.robot module"""

import unittest

import torch

from steal.rmpflow.kinematics.robot import JointLimit, Robot


class TestJointLimit(unittest.TestCase):
    """Tests for joint limit class"""

    def test_joint_limit(self):
        """Test the properties in JointLimit"""
        joint_limit = JointLimit(0.01, 0.1)
        self.assertEqual(joint_limit.lower, 0.01)
        self.assertEqual(joint_limit.upper, 0.1)


class TestRobot(unittest.TestCase):
    """Tests for the Robot class"""

    def setUp(self):
        self.robot = Robot(
            urdf_path='tests/fixtures/lula_franka_gen_fixed_gripper.urdf')

    def test_constructor(self):
        """Test constructor for Robot class."""
        self.assertEqual(len(self.robot.joints), 17)
        self.assertEqual(len(self.robot.links), 18)

    def test_sort_link_names(self):
        """Test sort_link_names method."""
        self.assertListEqual(self.robot.link_names, [
            'base_link', 'panda_link0', 'panda_link1', 'panda_link2',
            'panda_link3', 'panda_link4', 'panda_forearm_end_pt',
            'panda_link5', 'panda_link6', 'panda_link7', 'panda_wrist_end_pt',
            'panda_link8', 'panda_hand', 'right_gripper', 'panda_leftfinger',
            'panda_rightfinger', 'panda_rightfingertip', 'panda_leftfingertip'
        ])

    def test_get_all_task_maps(self):
        """Test get_all_task_maps method."""
        all_task_maps = self.robot.get_all_task_maps()
        self.assertEqual(len(all_task_maps), 18)

    def test_task_map(self):
        """Regression test for output of robot FK task map."""
        r = self.robot
        taskmap = r.get_task_map(target_link='panda_leftfingertip',
                                 base_link='base_link')
        x = torch.zeros(1, 7)
        xd = torch.ones(1, 7)
        y, yd, J, Jd = taskmap(x, xd)

        torch.testing.assert_close(
            y, torch.tensor([[0.11627766, -0.02827764, 0.82259995]]))
        torch.testing.assert_close(
            yd, torch.tensor([[0.58295530, 0.32055533, 0.03377766]]))
        torch.testing.assert_close(
            J,
            torch.tensor([[
                [
                    2.82776430e-02, 4.89600003e-01, 2.82776561e-02,
                    -1.73599988e-01, 2.82776393e-02, 2.10400015e-01,
                    -2.82776579e-02
                ],  #
                [
                    1.16277665e-01, -5.08265829e-09, 1.16277665e-01,
                    -1.47646850e-09, 1.16277665e-01, -5.08265829e-09,
                    -2.82776617e-02
                ],
                [
                    0.00000000e+00, -1.16277665e-01, 0.00000000e+00,
                    3.37776616e-02, 0.00000000e+00, 1.16277665e-01,
                    -2.47211163e-09
                ]
            ]]))
        torch.testing.assert_close(
            Jd,
            torch.tensor([[
                [
                    -0.32055533, 0.03377768, -0.32055533, -0.11627766,
                    -0.32055533, -0.11627764, 0.05655532
                ],  #
                [
                    0.58295530, 0.48959997, 0.09335532, -0.34719998,
                    0.26695529, 0.63120008, -0.05655532
                ],
                [
                    0.00000000, -0.55467767, -0.02827766, 0.21039999,
                    0.00000000, 0.18212235, -0.02827766
                ]
            ]]))
