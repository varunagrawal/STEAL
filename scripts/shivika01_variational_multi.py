"""
Script to run variational inference on a
multi-output Gaussian Process on the LASA dataset.

python scripts/shivika01_variatonal_multi.py
"""

import gpytorch
import numpy as np
import torch
from matplotlib import pyplot as plt

from steal.datasets import lasa
from steal.gp.multi_task import MultitaskApproximateGP
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
    gp = MultitaskApproximateGP(num_tasks=num_tasks, num_latents=num_latents)

    num_epochs = 60
    gp.training(train_t, train_xy, training_iterations=num_epochs)

    model = gp.get_model()
    likelihood = gp.evaluation()

    # Set into eval mode
    model.eval()
    likelihood.eval()

    # Make predictions
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        test_t = torch.linspace(0, 6, 1000).double()
        preds = model(test_t)
        predictions = likelihood(preds)
        mean = predictions.mean
        lower, upper = predictions.confidence_region()
        samples = gp.sampling(preds, 10) 
        print(type(samples))
    
    # Initialize plots
    fig, axs = plt.subplots(1, num_tasks, figsize=(4 * num_tasks, 3))

    y_lim = [[-40, 15], [-25, 30]]
    for task, ax in enumerate(axs):

        # Plot training data as dashed lines
        for trajectory in trajectories:
            ax.plot(trajectory.t[0], trajectory.pos[task, :], '--')

        # Predictive mean as blue line
        ax.plot(test_t, mean[:, task], 'b')
        # Shade in confidence
        ax.fill_between(test_t, lower[:, task], upper[:, task], alpha=0.5)
        ax.set_xlim([0, 6.0])
        ax.set_ylim(y_lim[task])
        ax.legend([
            'Trajectory 1', 'Trajectory 2', 'Trajectory 3', 'Trajectory 4',
            'Trajectory 5', 'Trajectory 6', 'Trajectory 7', 'Mean',
            'Confidence'
        ])
        ax.set_title(f'Task {task + 1}')

    fig.tight_layout()

    plt.savefig('multi_VGP.png')
    plt.show()

    legend_list = [ 
        'Sample 1', 'Sample 2', 'Sample 3', 'Sample 4', 'Sample 5',
        'Sample 6', 'Sample 7', 'Sample 8', 'Sample 9', 'Sample 10',
        'Mean', 'Confidence'
    ]
    # Sample visualization
    gp.plot_samples(num_tasks, y_lim, test_t, samples, mean, lower, upper, legend_list)


if __name__ == "__main__":
    main()
