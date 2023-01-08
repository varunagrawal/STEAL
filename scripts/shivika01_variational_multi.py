"""
Script to run variational inference on a
multi-output Gaussian Process on the LASA dataset.
python scripts/shivika01_variatonal_multi.py
"""

import gpytorch
import torch

from steal.datasets.lasa import Lasa
from steal.gp.multi_task import MultitaskApproximateGaussianProcess
from steal.utils.plotting.gp import plot_multi_output_gp


def main():
    """Main runner"""
    lasa = Lasa(shape="heee")
    # Concatenating the demos
    train_t, train_xy = lasa.concatenated_trajectories()

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

    x_lim = [0, 6.0]
    y_lims = [[-40, 15], [-25, 30]]
    plot_multi_output_gp(lasa.demos,
                         test_t,
                         mean,
                         lower,
                         upper,
                         num_tasks=2,
                         x_lim=x_lim,
                         y_lims=y_lims,
                         image_name='multi_VGP.png')


if __name__ == "__main__":
    main()
