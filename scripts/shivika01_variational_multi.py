"""
Script to run variational inference on a
multi-output Gaussian Process on the LASA dataset.

python scripts/shivika01_variatonal_multi.py
"""

import gpytorch
import numpy as np
import torch

from steal.datasets import lasa

from steal.gp.multi_task import MultitaskApproximateGaussianProcess
from steal.utils.plotting.gp import plot_multi_output_gp
from steal.datasets.lasa_GP import load_trajectories, concatenate_trajectories


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
        preds = model(test_t)
        predictions = gp.evaluate(test_t)
        mean = predictions.mean
        lower, upper = predictions.confidence_region()
        samples = gp.sampling(preds, 10)

    x_lim = [0, 6.0]
    y_lims = [[-40, 15], [-25, 30]]
    plot_multi_output_gp(trajectories,
                         test_t,
                         mean,
                         lower,
                         upper,
                         num_tasks=2,
                         x_lim=x_lim,
                         y_lims=y_lims,
                         image_name='multi_VGP.png')

    legend_list = [ 
        'Sample 1', 'Sample 2', 'Sample 3', 'Sample 4', 'Sample 5',
        'Sample 6', 'Sample 7', 'Sample 8', 'Sample 9', 'Sample 10',
        'Mean', 'Confidence'
    ]
    # Sample visualization
    gp.plot_samples(num_tasks, y_lim, test_t, samples, mean, lower, upper, legend_list)


if __name__ == "__main__":
    main()
