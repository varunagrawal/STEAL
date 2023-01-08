"""Utilites for plotting of Gaussian Processes"""

from matplotlib import pyplot as plt


def plot_gp(observed_pred,
            trajectories,
            means,
            legend=(),
            xlim=(0, 6.0),
            ylim=(-40, 15),
            xlabel="Time",
            ylabel="X-position",
            image_name=None,
            plot_intervals=True):
    """
    Plot the gaussian process with confidence intervals.
    """
    # Initialize plot
    _, ax = plt.subplots(1, 1, figsize=(4, 3))

    # Get upper and lower confidence bounds
    lower, upper = observed_pred.confidence_region()

    # Plot training data as dotted lines
    for x, y in trajectories:
        ax.plot(x, y, '--')

    # Plot predictive means as blue line
    ax.plot(means[0], means[1], 'b')

    if plot_intervals:
        # Shade between the lower and upper confidence bounds
        ax.fill_between(means[0], lower.numpy(), upper.numpy(), alpha=0.5)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.legend(legend)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if image_name:
        plt.savefig(image_name)

    plt.show()


def plot_3d_traj(trajectories, t, observed_preds, legend):

    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection='3d')

    # Plot training data as black stars
    for trajectory in trajectories:
        train_x = trajectory.pos[0, :]
        train_y = trajectory.pos[1, :]
        ax.plot(train_y, train_x, t, '--')

    # Plot predictive means as blue line
    ax.plot(observed_preds[1].mean, observed_preds[0].mean, t, 'b')
    ax.legend(legend)
    ax.set_xlabel('Y-position')
    ax.set_ylabel('X-position')
    ax.set_zlabel('Time')
    plt.show()


def plot_multi_output_gp(trajectories,
                         x,
                         mean,
                         lower,
                         upper,
                         num_tasks=2,
                         x_lim=(0, 6.0),
                         y_lims=([-40, 15], [-25, 30]),
                         legend=('Trajectory 1', 'Trajectory 2',
                                 'Trajectory 3', 'Trajectory 4',
                                 'Trajectory 5', 'Trajectory 6',
                                 'Trajectory 7', 'Mean', 'Confidence'),
                         image_name=None):
    """
    Plot the trajectory means for a multi-output GP
    with the upper and lower confidence intervals.
    """
    # This contains predictions for both tasks, flattened out
    # The first half of the predictions is for the first task
    # The second half is for the second task

    # Initialize plots
    fig, axes = plt.subplots(1, num_tasks, figsize=(4 * num_tasks, 3))

    for task, ax in enumerate(axes):
        # Plot training data as dotted line
        for trajectory in trajectories:
            ax.plot(trajectory.t[0], trajectory.pos[task, :], '--')

        # Predictive mean as blue line
        ax.plot(x, mean[:, task], 'b')

        # Shade in confidence
        ax.fill_between(x, lower[:, task], upper[:, task], alpha=0.5)

        ax.set_xlim(x_lim)
        ax.set_ylim(y_lims[task])
        ax.legend(legend)
        ax.set_title(f'Task {task + 1}')

    fig.tight_layout()

    if image_name:
        plt.savefig(image_name)

    plt.show()


def plot_samples(num_tasks,
                 test_x,
                 samples,
                 mean,
                 lower=None,
                 upper=None,
                 legend=('Sample 1', 'Sample 2', 'Sample 3', 'Sample 4',
                         'Sample 5', 'Sample 6', 'Sample 7', 'Sample 8',
                         'Sample 9', 'Sample 10', 'Mean', 'Confidence'),
                 x_lim=(0, 6.0),
                 y_lims=([-40, 15], [-25, 30]),
                 title="sampled_trajectories"):
    """Visualize the samples from a GP"""
    fig, axs = plt.subplots(1, num_tasks, figsize=(4 * num_tasks, 3))

    for task, ax in enumerate(axs):
        # Plot training data as dashed lines
        for sample in samples:
            ax.plot(test_x, sample[:, task], '-')

        # Predictive mean as blue line
        ax.plot(test_x, mean[:, task], 'b')

        if lower is not None and upper is not None:
            # Shade in confidence
            ax.fill_between(test_x, lower[:, task], upper[:, task], alpha=0.5)

        ax.set_xlim(x_lim)
        ax.set_ylim(y_lims[task])
        ax.legend(legend)
        ax.set_title(f'Task {task + 1}')

    fig.tight_layout()
    fig.suptitle(title)

    filename = '_'.join(title.lower().split()) + '.png'
    print(f"Saving plot to {filename}")
    plt.savefig(filename)
    plt.show()
