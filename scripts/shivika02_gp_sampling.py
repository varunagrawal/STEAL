"""
Script to train a GP and sample from it.

python scripts/shivika02_gp_sampling.py
"""

import gpytorch
import torch

from steal.datasets.lasa_GP import concatenate_trajectories, load_trajectories
from steal.gp.multi_task import MultitaskApproximateGaussianProcess
from steal.utils.plotting.gp import plot_samples


def main():
    """Main runner"""
    trajectories = load_trajectories()

    # Concatenating the demos
    train_t, train_xy = concatenate_trajectories(trajectories)

    # Time
    train_t = torch.tensor(train_t)
    # X, Y Position
    train_xy = torch.tensor(train_xy)

    num_latents = 3
    num_tasks = 2
    gp = MultitaskApproximateGaussianProcess(num_tasks=num_tasks,
                                             num_latents=num_latents)

    training_iterations = 60
    gp.train(train_t, train_xy, training_iterations=training_iterations)

    # Make predictions
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        test_t = torch.linspace(0, 6, 1000).double()
        predictions = gp.evaluate(test_t)
        mean = predictions.mean
        lower, upper = predictions.confidence_region()
        samples = gp.samples(test_t, 10)

    x_lim = [0, 6.0]
    y_lims = [[-40, 15], [-25, 30]]
    legend_list = [
        'Sample 1', 'Sample 2', 'Sample 3', 'Sample 4', 'Sample 5', 'Sample 6',
        'Sample 7', 'Sample 8', 'Sample 9', 'Sample 10', 'Mean', 'Confidence'
    ]
    # Sample visualization
    plot_samples(num_tasks, test_t, samples, mean, lower, upper, legend_list,
                 x_lim, y_lims)


if __name__ == "__main__":
    main()
