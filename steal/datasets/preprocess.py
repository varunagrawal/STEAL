"""Utilities to preprocess the dataset."""

import numpy as np
import torch
from fastdtw import fastdtw
from scipy import interpolate
from scipy.signal import medfilt, savgol_filter
from scipy.spatial.distance import euclidean
from torch.utils.data import TensorDataset


def preprocess_dataset(traj_list,
                       dt,
                       start_cut=15,
                       end_cut=10,
                       downsample_rate=1,
                       smoothing_window_size=25,
                       vel_thresh=0.,
                       goal_at_origin=False):
    """Preprocess the trajectory data.
    This function performs the following for each demonstration:
    1. Smooth the trajectory with the Savitzky-Golay filter.
    2. Downsample the dataset based on `downsample_rate`.
    3. Compute the velocities using finite differences.
    4. Trim out the edges of the trajectory based on `start_cut` and `end_cut`.

    Args:
        traj_list (_type_): _description_
        dt (_type_): _description_
        start_cut (int, optional): _description_. Defaults to 15.
        end_cut (int, optional): _description_. Defaults to 10.
        downsample_rate (int, optional): _description_. Defaults to 1.
        smoothing_window_size (int, optional): _description_. Defaults to 25.
        vel_thresh (_type_, optional): _description_. Defaults to 0..
        goal_at_origin (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    n_dims = traj_list[0].shape[1]
    n_demos = len(traj_list)

    # New downsampled time delta
    if downsample_rate > 1:
        dt = round(dt * downsample_rate, 2)

    torch_datasets = []
    for i in range(n_demos):
        demo_pos = traj_list[i]

        for j in range(n_dims):
            demo_pos[:, j] = savgol_filter(demo_pos[:, j],
                                           smoothing_window_size, 3)

        demo_pos = demo_pos[::downsample_rate, :]
        demo_vel = np.diff(demo_pos, axis=0) / dt
        demo_vel_norm = np.linalg.norm(demo_vel, axis=1)
        ind = np.where(demo_vel_norm > vel_thresh)
        demo_pos = demo_pos[np.min(ind):(np.max(ind) + 2), :]

        for j in range(n_dims):
            demo_pos[:, j] = savgol_filter(demo_pos[:, j],
                                           smoothing_window_size, 3)

        demo_pos = demo_pos[start_cut:-end_cut, :]

        if goal_at_origin:
            demo_pos = demo_pos - demo_pos[-1]

        demo_vel = np.diff(demo_pos, axis=0) / dt
        demo_vel = np.concatenate((demo_vel, np.zeros((1, n_dims))), axis=0)

        torch_datasets.append(
            TensorDataset(
                torch.from_numpy(demo_pos).to(torch.get_default_dtype()),
                torch.from_numpy(demo_vel).to(torch.get_default_dtype())))
    return torch_datasets, dt


def align_traj(source_traj, target_traj):
    """Align source and target trajectories using Dynamic Time Warping."""
    distance, path = fastdtw(source_traj, target_traj, dist=euclidean)
    target_traj_aligned = np.zeros_like(source_traj)

    for point in path:
        target_traj_aligned[point[0], :] = target_traj[point[1], :]

    return target_traj_aligned


def crop_trajectory(traj, tol):
    """
    Remove start and end parts of the trajectory which don't differ by `tol`.
    """
    init = traj[0, :]
    goal = traj[-1, :]
    try:
        idx_min = np.where(np.linalg.norm(traj - init, axis=1) > tol)[0][0]
        idx_max = np.where(np.linalg.norm(traj - goal, axis=1) > tol)[0][-1]
    except IndexError:
        return np.array([])
    return traj[idx_min:idx_max, :]


def monotonize_trajectory(traj, tol):
    """Compute monotonic version of the trajectory."""
    monotonic_traj = []
    monotonic_traj.append(traj[0, :])
    for n in range(1, len(traj)):
        if np.linalg.norm(traj[n, :] - monotonic_traj[-1]) > tol:
            monotonic_traj.append(traj[n, :])

    monotonic_traj = np.array(monotonic_traj)
    return monotonic_traj


def smooth_trajectory(traj, window_size, polyorder=3):
    """Apply Savitzky-Golay filter to smooth the trajectory `traj`."""
    smoothed_traj = np.zeros_like(traj)

    for j in range(traj.shape[1]):
        smoothed_traj[:, j] = savgol_filter(traj[:, j], window_size, polyorder)

    return smoothed_traj


def differentiate_trajectory(pos_traj, dt, normalize_vel=False):
    """Differentiate the position trajectory to get the velocity and acceleration trajectories."""
    num_dim = pos_traj.shape[1]
    vel_traj = np.diff(pos_traj, axis=0) / dt
    static_threshold = 1e-6
    static_flag = np.any(np.linalg.norm(vel_traj, axis=1) <= static_threshold)

    if static_flag:
        print(
            f"WARNING: Found velocities less than {static_threshold}. Consider Monotonizing!"
        )

    if normalize_vel:
        vel_norms = np.maximum(
            np.linalg.norm(vel_traj, axis=1),
            1e-16).reshape(-1, 1)  # lower bounding vel magnitudes to 1e-15
        vel_traj = vel_traj / vel_norms

    vel_traj = np.concatenate((vel_traj, np.zeros((1, num_dim))), axis=0)
    acc_traj = np.diff(vel_traj, axis=0) / dt
    acc_traj = np.concatenate((acc_traj, np.zeros((1, num_dim))), axis=0)

    return vel_traj, acc_traj


def transform_traj(traj, transform_mat):
    """
    Transform a 3D position trajectory using a homogenous transformation

    :param traj: Nx3 array   # Works for 3-D position trajectories only
    :param transform_mat: homogeneous transformation (list or single transform)
    :return:
    """
    traj = np.concatenate(
        (traj, np.ones((traj.shape[0], 1))),
        axis=1)  # appending a 1 at the end for transformation
    traj_transformed = np.zeros_like(traj)

    if isinstance(transform_mat, list):
        if len(transform_mat) == len(traj):
            for i in range(len(traj)):
                traj_transformed[i, :] = np.dot(transform_mat[i],
                                                traj[i, :].reshape(-1, 1)).T
        else:
            print("Error! Length of transforms list != lenght of trajectory")
            return

    else:
        traj_transformed = np.dot(transform_mat, traj.T).T

    traj_transformed = traj_transformed[:, :-1]  # removing the appened 1

    return traj_transformed


def compute_link_trajectory(joint_traj, fk_func, output_dim=3):
    """
    Carries out forward kinematics
    :param joint_traj:
    :param fk_func:  forward kinematics function (can come from the task map)
    :return:
    """

    link_traj = np.zeros((joint_traj.shape[0], output_dim))
    for i in range(joint_traj.shape[0]):
        link_traj[i, :] = fk_func(joint_traj[i, :]).flatten()

    return link_traj


def transform_inv(T):
    """
    Inverts a rigid body transformation
    :param T:
    :return:
    """
    T_inv = np.zeros_like(T)
    T_inv[-1, -1] = 1.0
    R = T[0:3, 0:3]
    p = T[0:3, -1].reshape(-1, 1)

    T_inv[0:3, 0:3] = R.T
    T_inv[0:3, -1] = -np.dot(R.T, p).flatten()

    return T_inv


def resample_trajectory(traj, num_samples):
    """Resample `traj` with `num_samples` points."""
    resampled_traj = np.zeros((num_samples, traj.shape[1]))
    x = np.linspace(0, 1, traj.shape[0], endpoint=True)
    x_new = np.linspace(0, 1, num_samples, endpoint=True)

    for i in range(traj.shape[1]):
        f = interpolate.interp1d(x, traj[:, i], 'cubic')
        resampled_traj[:, i] = f(x_new)

    return resampled_traj


def trim_trajectory(traj, start_cut, end_cut):
    """Trim the start and end of the trajectory using `start_cut` and `end_cut`.

    Args:
        traj: The trajectory to trim.
        start_cut (int): The starting point of the trimmed trajectory.
        end_cut (_type_): The end point of the trimmed trajectory.

    Returns:
        The trimmed trajectory.
    """
    # trimmed_traj = traj[start_cut:, :]
    # trimmed_traj = trimmed_traj[:-(end_cut+1),:]

    num_pts = traj.shape[0]
    trimmed_traj = traj[start_cut:num_pts - end_cut, :]

    return trimmed_traj


def median_filter_trajectory(traj, kernel_size):
    """Apply a median filter on the trajectory to remove noise/outliers."""
    filtered_traj = np.zeros_like(traj)
    for i in range(traj.shape[1]):
        filtered_traj[:, i] = medfilt(traj[:, i], kernel_size=kernel_size)

    return filtered_traj


# ------------------------------------
# Batch processing versions


def scale_trajectories(traj_list, scale_factor):
    """Scale each trajectory in `traj_list`."""
    scaled_traj_list = []
    for _, traj in enumerate(traj_list):
        scaled_traj_list.append(scale_factor * traj)
    return scaled_traj_list


def trim_trajectories(traj_list, start_cut, end_cut):
    """Trim each trajectory in `traj_list`."""
    trimmed_traj_list = []
    for _, traj in enumerate(traj_list):
        trimmed_traj = trim_trajectory(traj, start_cut, end_cut)
        trimmed_traj_list.append(trimmed_traj)

    return trimmed_traj_list


def resample_trajectories(traj_list, num_samples):
    """Resample each trajectory to have `num_samples` points.

    Args:
        traj_list (Iterable): The trajectories to resample.
        num_samples (int): The number of samples we want in the resampled trajectories.

    Returns:
        Iterable: The resampled trajectories.
    """
    resampled_traj_list = []

    for _, traj in enumerate(traj_list):
        resampled_traj = resample_trajectory(traj, num_samples)
        resampled_traj_list.append(resampled_traj)

    return resampled_traj_list


def align_trajectories(traj_list, source_idx):
    """
    Time aligns trajectories using dynamic time warping
    :param traj_list:
    :param source_idx:
    :return:
    """
    aligned_traj_list = []
    source_traj = traj_list[source_idx]

    for i, target_traj in enumerate(traj_list):
        if i != source_idx:
            target_traj_aligned = align_traj(source_traj, target_traj)
            aligned_traj_list.append(target_traj_aligned)
        else:
            aligned_traj_list.append(source_traj)

    return aligned_traj_list


def crop_trajectories(traj_list, tol):
    """
    Removes static points at the corners of the trajectories

    :param traj_list:
    :param tol:
    :return:
    """
    cropped_traj_list = [crop_trajectory(traj, tol) for traj in traj_list]
    return cropped_traj_list


def smooth_trajectories(traj_list, window_size):
    """
    Smooth trajectories using Savitzky-Golay filter.

    Args:
        traj_list (_type_): _description_
        window_size (_type_): _description_

    Returns:
        Iterable: List of smoothed trajectories.
    """
    smoothed_traj_list = []

    for traj in traj_list:
        smoothed_traj_list.append(smooth_trajectory(traj, window_size))

    return smoothed_traj_list


def monotonize_trajectories(traj_list, tol):
    """Monotonize a list of trajectories."""
    monotonic_traj_list = []
    for _, traj in enumerate(traj):
        monotonic_traj_list.append(monotonize_trajectory(traj, tol))

    return monotonic_traj_list


def compute_link_trajectories(joint_traj_list, fk_func, output_dim=3):
    """
    Apply forward kinematics `fk_func` to compute
    the link trajectories from the joint trajectories.
    """
    link_traj_list = []

    for joint_traj in joint_traj_list:
        link_traj = compute_link_trajectory(joint_traj, fk_func, output_dim)
        link_traj_list.append(link_traj)

    return link_traj_list


def min_dtw_traj_idx(traj_list):
    """Get the trajectory index which has the minimum DTW distance to all the trajectories."""
    mean_dist_list = []
    for i, source_traj in enumerate(traj_list):
        dist_list = []
        for j, target_traj in enumerate(traj_list):
            if i != j:
                distance, _ = fastdtw(source_traj, target_traj, dist=euclidean)
                dist_list.append(distance)

        mean_dist = np.mean(np.array(dist_list))
        mean_dist_list.append(mean_dist)

    return mean_dist_list.index(min(mean_dist_list))


def transform_trajectories(traj_list, transform_mat):
    """
    Transforms a list of trajectories given either a
        1) single transformation matrix for all trajectories OR
        2) list of transformation matrices (1 per trajectory) OR
        3) list of list of transformation matrices (1 per trajectory point)
    :param traj_list: list of trajectories
    :param transform_mat: transformation matrices
    :return: list of transformed trajectories
    """
    transformed_traj_list = []

    if isinstance(transform_mat, list):
        if len(transform_mat) == len(traj_list):
            for _, traj in enumerate(traj_list):
                transformed_traj = transform_traj(traj, transform_mat[i])
                transformed_traj_list.append(transformed_traj)
        else:
            print(
                "Error! Length of transforms list != length of trajectory list"
            )
            return
    else:
        for _, traj in enumerate(traj_list):
            transformed_traj = transform_traj(traj, transform_mat)
            transformed_traj_list.append(transformed_traj)

    return transformed_traj_list


def median_filter_trajectories(traj_list, kernel_size):
    """Apply the median filter on each trajectory. Helps to remove noise."""
    filtered_traj_list = [
        median_filter_trajectory(traj, kernel_size) for traj in traj_list
    ]
    return filtered_traj_list


def differentiate_trajectories(traj_list, dt, normalize_vel=False):
    """Differentiate the trajectories to get the velocities and accelerations."""
    vel_traj_list = []
    acc_traj_list = []

    for pos_traj in traj_list:
        vel_traj, acc_traj = differentiate_trajectory(
            pos_traj, dt, normalize_vel=normalize_vel)
        vel_traj_list.append(vel_traj)
        acc_traj_list.append(acc_traj)

    return vel_traj_list, acc_traj_list


def find_mean_goal(robot,
                   joint_traj_list,
                   link_name,
                   base_to_tracked_frame_transforms=None):
    """
    Find the goals as the mean of the end point of demos for each link.
    NOTE: If the base_to_tracked_frame_transforms is provides,
        the goals will be in reference of the tracked frames

    Args:
        robot: Robot object.
        joint_traj_list (Iterable): List of trajectories in the configuration space
            (aka joint angles).
        link_name (str): The name of the link to get the mean goal point for.
        base_to_tracked_frame_transforms (np.ndarray, optional): Tranformation matrix
            from base frame to tracking frame. Defaults to None.

    Returns:
        Return the mean goal point of all the trajectories as well as
        the translation biases for each one.
    """
    n_demos = len(joint_traj_list)
    link_task_map = robot.get_task_map(target_link=link_name)

    link_traj_list_base = []
    for joint_traj in joint_traj_list:
        joint_traj = torch.from_numpy(joint_traj).to(
            dtype=torch.get_default_dtype())
        link_traj = link_task_map.psi(joint_traj).numpy()
        link_traj_list_base.append(link_traj)

    if base_to_tracked_frame_transforms:
        # inverting the transforms
        tracked_frame_to_base_transforms = \
         [transform_inv(T_base_to_frame) for T_base_to_frame in base_to_tracked_frame_transforms]

        # transforming to the tracked frame space
        leaf_traj_list = transform_trajectories(
            link_traj_list_base, tracked_frame_to_base_transforms)
    else:
        # use the base link reference frame
        leaf_traj_list = link_traj_list_base

    # Extracting goals in the leaf coordinates
    goals = np.array([traj[-1].tolist() for traj in leaf_traj_list])

    # Finding the mean in the tracked frame coordinates
    mean_goal = np.mean(goals, axis=0).reshape(1, -1)

    # find goal biases (translations)
    demo_goal_biases = []
    for n in range(n_demos):
        # adding a virtual link which ends at the goal
        bias = leaf_traj_list[n][-1].reshape(1, -1) - mean_goal
        demo_goal_biases.append(bias)

    return mean_goal, demo_goal_biases
