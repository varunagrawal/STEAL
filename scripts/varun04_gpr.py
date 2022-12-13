"""
Script to run variational inference on a
multi-output Gaussian Process on the LASA dataset.
python scripts/shivika01_variatonal_multi.py
"""

import gpytorch
import torch

from steal.datasets.lasa_GP import concatenate_trajectories, load_trajectories
from steal.gp.scalar import ScalarPreferenceGaussianProcess
from steal.utils.plotting.gp import plot_gp


def main():
    """Main runner"""
    trajectories = load_trajectories()

    # Concatenating the demos
    train_t, train_xy = concatenate_trajectories(trajectories)

    # Time
    train_t = torch.tensor(train_t).unsqueeze(1)
    # Append preference value to input
    train_t = torch.hstack((train_t, torch.ones_like(train_t) * 10))
    train_t[0:1000, 1] = 1
    train_t[6000:7000, 1] = 1

    # X, Y Position
    train_x = torch.tensor(train_xy[:, 0])

    gp = ScalarPreferenceGaussianProcess(train_t, train_x)

    training_iterations = 60
    gp.train(train_t, train_x, training_iterations=training_iterations)

    test_t = torch.linspace(0, 6, 1000).double().unsqueeze(1)
    test_t = torch.hstack((test_t, torch.ones_like(test_t) * 10))

    # Make predictions
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        posterior = gp.evaluate(test_t)
        mean = posterior.mean

        x_lim = [0, 6.0]
        y_lims = [[-40, 15], [-25, 30]]

        trajectories_to_plot = [(trajectory.t[0], trajectory.pos[0, :])
                                for trajectory in trajectories]

        legend = [
            'Observed Demo 1', 'Observed Demo 2', 'Observed Demo 3',
            'Observed Demo 4', 'Observed Demo 5', 'Observed Demo 6',
            'Observed Demo 7', 'Mean', 'Confidence'
        ]
        plot_gp(posterior,
                trajectories_to_plot,
                [test_t[:, 0].numpy(), mean.numpy()],
                xlim=x_lim,
                ylim=y_lims[0],
                legend=legend,
                image_name='images/rbf_kernel.png')


if __name__ == "__main__":
    main()
