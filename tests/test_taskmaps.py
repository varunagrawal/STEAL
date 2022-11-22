"""Test for varios TaskMaps"""
import torch

from steal.rmpflow.kinematics.distance_taskmaps import SphereDistanceTaskMap

torch.set_printoptions(precision=12)


def test_sphere_distance_task_map():
    """Test the SphereDistanceTaskMap"""
    obstacle_taskmap = SphereDistanceTaskMap(n_inputs=2,
                                             radius=1.0,
                                             center=torch.asarray([[0, 0]]))
    x = torch.ones(1, 2)
    dist = obstacle_taskmap.psi(x)
    torch.testing.assert_close(dist, torch.tensor([[0.414213538170]]))

    actual_jacobian = obstacle_taskmap.J(x).unsqueeze(1)
    expected_jacobian = torch.autograd.functional.jacobian(
        obstacle_taskmap.psi, x, create_graph=True)
    torch.testing.assert_close(actual_jacobian, expected_jacobian)

    #TODO Test for J_d. Need x_d for this.
    print(expected_jacobian)
