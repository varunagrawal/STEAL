"""Tests for metrics module"""

import torch

from steal.rmpflow.controllers.metrics import StretchMetric


def test_stretch_metric():
    """Test StretchMetric."""
    R = torch.eye(3)
    metric = StretchMetric(R)
    q = torch.tensor([[1.0, 2.0, 3.0]])
    x = metric(q)

    expected_x = torch.tensor(
        [[[1.880828738213, 0.000000000000, 0.000000000000],
          [0.000000000000, 1.000000000000, 0.000000000000],
          [0.000000000000, 0.000000000000, 1.000000000000]]])
    torch.testing.assert_close(x, expected_x)
