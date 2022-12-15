"""
Script to train a GP and sample from it.

python scripts/shivika02_gp_sampling.py
"""

import argparse

import gpytorch
import torch

from steal.datasets.lasa_GP import concatenate_trajectories, load_trajectories
from steal.gp.multi_task import MultitaskApproximateGaussianProcess
from steal.utils.plotting.gp import plot_samples


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--sample_ppd',
        action='store_true',
        help=
        "Sample from the posterior predictive distribution instead of the GP posterior"
    )
    parser.add_argument('--dataset',
                        default="Sshape",
                        help="The shape name in the LASA dataset.")
    return parser.parse_args()


def main():
    """Main runner"""
    args = parse_args()

    trajectories = load_trajectories(dataset_name=args.dataset)

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

    test_t = torch.linspace(0, 4.8, 1000).double()

    # Compute posterior for the test data
    if args.sample_ppd:
        print("Sampling from PPD")
        posterior = gp.evaluate(test_t)
    else:
        posterior = gp.posterior(test_t)

    samples = gp.samples(test_t, 10)

    torch.save(samples, open("{args.dataset.lower()}_gp_samples.pt", 'wb'))

    # Compute the confidence interval for the posterior
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        mean = posterior.mean
        lower, upper = posterior.confidence_region()

    x_lim = [0, 6.0]
    y_lims = [[-15, 45], [-10, 60]]
    legend_list = [
        'Sample 1', 'Sample 2', 'Sample 3', 'Sample 4', 'Sample 5', 'Sample 6',
        'Sample 7', 'Sample 8', 'Sample 9', 'Sample 10', 'Mean', 'Confidence'
    ]
    # Sample visualization
    plot_samples(num_tasks,
                 test_t,
                 samples,
                 mean,
                 lower,
                 upper,
                 legend_list,
                 x_lim,
                 y_lims,
                 title=f"Sampled Trajectories {args.sample_ppd}")


if __name__ == "__main__":
    main()
