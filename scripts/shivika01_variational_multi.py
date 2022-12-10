import math
import torch
import gpytorch
import tqdm
import os
import numpy as np
from matplotlib import pyplot as plt
from steal.datasets import lasa
from steal.gp.variational_multi_output import MultitaskGPModel

def load_trajectories(dataset_name="heee"):
    """Load the demontstration trajectories from LASA with the name"""
    if hasattr(lasa.DataSet, dataset_name):
        dataset = getattr(lasa.DataSet, dataset_name)
        return dataset.demos
    else:
        raise ValueError(
            "Invalid dataset name specified. Please check the LASA dataset repo for valid names."
        )

def train_data(trajectories, i):
    """
    Training data.
    Input is time and output is the position.
    """
    X = torch.tensor(trajectories[i].t[0])
    y = torch.tensor((trajectories[i].pos).T)
    return X, y

def concatenate_trajectories(trajectories):
    """Concatenate the input into a single array"""
    train_t = np.empty((0, ))
    train_xy = np.empty((0, 2))
    for i, trajectory in enumerate(trajectories):
        train_t = np.hstack(
            (train_t, trajectory.t[0])) if train_t.size else trajectory.t[0]
        train_xy = np.vstack(
            (train_xy,
             trajectory.pos.T)) if train_xy.size else trajectory.pos.T
    return train_t, train_xy

num_latents = 3
num_tasks = 2
trajectories = load_trajectories()

# Concatenating the demos
train_t, train_xy = concatenate_trajectories(trajectories)

# Time
train_t = torch.tensor(train_t)
# X, Y Position
train_xy = torch.tensor(train_xy)

model = MultitaskGPModel(num_latents, num_tasks)
model.double()
likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks)
likelihood(model(train_t)).rsample().shape

num_epochs = 60

model.train()
likelihood.train()

optimizer = torch.optim.Adam([
    {'params': model.parameters()},
    {'params': likelihood.parameters()},
], lr=0.1)

# Our loss object. We're using the VariationalELBO, which essentially just computes the ELBO
mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_xy.size(0))

for i in range(num_epochs):
    # Within each iteration, we will go over each minibatch of data
    optimizer.zero_grad()
    output = model(train_t)
    loss = -mll(output, train_xy)
    print('Iter %d/%d - Loss: %.3f' % (i + 1, num_epochs, loss.item()))
    loss.backward()
    optimizer.step()

# Set into eval mode
model.eval()
likelihood.eval()

# Initialize plots
fig, axs = plt.subplots(1, num_tasks, figsize=(4 * num_tasks, 3))

# Make predictions
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    test_t = torch.linspace(0, 6, 1000).double()
    predictions = likelihood(model(test_t))
    mean = predictions.mean
    lower, upper = predictions.confidence_region()

y_lim = [[-40, 15],[-25, 30]]
for task, ax in enumerate(axs):
    
    # Plot training data as dashed lines
    for j, trajectory in enumerate(trajectories):
        train_t = trajectory.t[0]
        train_y = trajectory.pos[task, :]
        ax.plot(train_t, train_y, '--')

    # Predictive mean as blue line
    ax.plot(test_t.numpy(), mean[:, task].numpy(), 'b')
    # Shade in confidence 
    ax.fill_between(test_t.numpy(), lower[:, task].numpy(), upper[:, task].numpy(), alpha=0.5)
    ax.set_xlim([0, 6.0])
    ax.set_ylim(y_lim[task])
    ax.legend([ 'Trajectory 1', 'Trajectory 2', 'Trajectory 3',
                'Trajectory 4', 'Trajectory 5', 'Trajectory 6', 
                'Trajectory 7', 'Mean', 'Confidence'])
    ax.set_title(f'Task {task + 1}')

plt.savefig('multi_VGP.png')
plt.show()

