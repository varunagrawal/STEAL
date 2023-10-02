"""Unit tests for various datasets."""
import unittest

import numpy as np

from steal.datasets import Lasa, process_data
from steal.learning import get_params


class TestLasaDataset(unittest.TestCase):
    """Unit tests for the LASA dataset and associated functions."""

    def test_lasa(self):
        """
        Test if the LASA dataset is correctly loading.
        """
        lasa = Lasa()
        self.assertIsInstance(lasa, Lasa)

    def test_load_trajectories(self):
        """Test if loading of trajectories is correct."""
        lasa = Lasa(shape="heee")
        trajectories = lasa.demos
        assert len(trajectories) == 7

        train_t, train_xy = lasa.concatenated_trajectories()
        assert train_t.shape == (7000, )
        assert train_xy.shape == (7000, 2)

    def test_process_data(self):
        """Test the process_data function."""
        lasa = Lasa()
        params = get_params()
        cspace_dim = 6

        time_list, joint_traj_list, dt, n_demos = process_data(
            lasa.demos, lasa.dt, params, cspace_dim)

        self.assertEqual(len(time_list), 7)
        self.assertEqual(time_list[0][0], 0.0)
        self.assertEqual(time_list[0][-1], 4.58)

        self.assertEqual(len(joint_traj_list), 7)
        self.assertEqual(joint_traj_list[0].shape, (230, 6))

        np.testing.assert_allclose(
            joint_traj_list[0][0],
            np.asarray([35.665249, 40.796383, 0.0, 0.0, 0.0, 0.0]))
        np.testing.assert_allclose(joint_traj_list[0][-1],
                                   np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

        self.assertEqual(dt, 0.02)
        self.assertEqual(n_demos, 7)
