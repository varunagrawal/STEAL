"""Unit tests for custom kernels."""

import unittest

import torch

from steal.gp.kernel import PreferenceKernel


class TestPreferenceKernel(unittest.TestCase):
    """Unit tests for the PreferenceKernel"""

    def test_construction(self):
        """Test if the PreferenceKernel is being constructed correctly."""
        kernel = PreferenceKernel()
        torch.testing.assert_close(kernel.x0, 0.0)
        torch.testing.assert_close(kernel.y0, -0.5)
        torch.testing.assert_close(kernel.k, 1.0)
        torch.testing.assert_close(kernel.c, 1.0)

    def test_forward(self):
        """Test the output from the forward method."""
        kernel = PreferenceKernel()

        x1, x2 = torch.ones(1, 1) * 10, torch.zeros(1, 1)

        # Test minimum difference in preferences
        torch.testing.assert_close(
            kernel(x1, x1).to_dense(), torch.zeros(1, 1))
        torch.testing.assert_close(
            kernel(x2, x2).to_dense(), torch.zeros(1, 1))

        torch.testing.assert_close(
            kernel(x1, x2).to_dense(),
            torch.ones(1, 1) * 0.4999545813)

        # Test for 2 arbitrary preferences
        torch.testing.assert_close(
            kernel(torch.ones(1, 1) * 5,
                   torch.ones(1, 1) * 8).to_dense(),
            torch.ones(1, 1) * 0.4525741339)
        torch.testing.assert_close(
            kernel(torch.ones(1, 1) * 8,
                   torch.ones(1, 1) * 5).to_dense(),
            torch.ones(1, 1) * 0.4525741339)

    def test_forward_batch(self):
        """Test the output from the forward method in batch mode."""
        kernel = PreferenceKernel()

        x1, x2 = torch.ones(10, 1, 1) * 10, torch.zeros(10, 1, 1)

        # Test minimum difference in preferences
        torch.testing.assert_close(
            kernel(x1, x1).to_dense(), torch.zeros(10, 1, 1))
        torch.testing.assert_close(
            kernel(x2, x2).to_dense(), torch.zeros(10, 1, 1))

        torch.testing.assert_close(
            kernel(x1, x2).to_dense(),
            torch.ones(10, 1, 1) * 0.4999545813)

        # Test for 2 arbitrary preferences
        batch_size = 59
        torch.testing.assert_close(
            kernel(
                torch.ones(batch_size, 1, 1) * 5,
                torch.ones(batch_size, 1, 1) * 8).to_dense(),
            torch.ones(batch_size, 1, 1) * 0.4525741339)
        torch.testing.assert_close(
            kernel(
                torch.ones(batch_size, 1, 1) * 8,
                torch.ones(batch_size, 1, 1) * 5).to_dense(),
            torch.ones(batch_size, 1, 1) * 0.4525741339)
