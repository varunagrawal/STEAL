import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.linalg import norm


def plot_traj_2D(traj, ls, color, order=2):
    """Plot a 2D trajectory."""
    plt.plot(traj[:, 0], traj[:, 1], linestyle=ls, linewidth=2, color=color)
    plt.plot(traj[0, 0], traj[0, 1], 'ko')
    plt.plot(traj[-1, 0], traj[-1, 1], 'x', color=color)
    if order == 2:
        plt.quiver(traj[0, 0],
                   traj[0, 1],
                   traj[0, 2] + 1e-20,
                   traj[0, 3] + 1e-20,
                   color='k',
                   scale_units='xy',
                   scale=1.)


def plot_trajectories_3D(traj_list, ls, color, ax_handle=None, zorder=100):
    """Plot multiple 3D trajectories."""
    if ax_handle is None:
        ax = plt.gca()
    else:
        ax = ax_handle

    for i, traj in enumerate(traj_list):
        ax.plot3D(traj[:, 0],
                  traj[:, 1],
                  traj[:, 2],
                  linestyle=ls,
                  linewidth=2,
                  color=color,
                  zorder=zorder + i)
        ax.scatter3D(traj[0, 0],
                     traj[0, 1],
                     traj[0, 2],
                     color='green',
                     marker='o')
        ax.scatter3D(traj[-1, 0],
                     traj[-1, 1],
                     traj[-1, 2],
                     color='red',
                     marker='x')
        # plt.quiver(traj[0, 0], traj[1, 0], traj[2, 0] + 1e-20, traj[3, 0] + 1e-20, color


def plot_traj_3D(traj, ls, color, ax_handle=None):
    """Plot a 3D trajectory."""
    if ax_handle is None:
        ax = plt.gca()
    else:
        ax = ax_handle
    ax.plot3D(traj[:, 0],
              traj[:, 1],
              traj[:, 2],
              linestyle=ls,
              linewidth=2,
              color=color)
    ax.scatter3D(traj[0, 0], traj[0, 1], traj[0, 2], color='green', marker='o')
    ax.scatter3D(traj[-1, 0],
                 traj[-1, 1],
                 traj[-1, 2],
                 color='red',
                 marker='x')
    # plt.quiver(traj[0, 0], traj[1, 0], traj[2, 0] + 1e-20, traj[3, 0] + 1e-20, color='k')


def plot_traj_time(time,
                   traj,
                   ls='--',
                   color='k',
                   axs_handles=None,
                   lw=2,
                   labels=('X', 'Y'),
                   title="Traj"):
    shape = traj.shape
    if shape[1] > shape[0]:
        traj = traj.T

    num_dim = traj.shape[1]
    if axs_handles is None:
        axs_handles = [None] * num_dim
    else:
        assert (len(axs_handles) == num_dim)

    for i in range(num_dim):
        if axs_handles[i] is None:
            ax = plt.subplot(num_dim, 1, i + 1)
        else:
            ax = axs_handles[i]
        ax.plot(time, traj[:, i], linestyle=ls, linewidth=lw, color=color)
        ax.set(ylabel=labels[i])
        axs_handles[i] = ax

    plt.xlabel("Time")
    plt.suptitle(title)
    return axs_handles


def plot_robot_2D(robot, q, lw=2, handle_list=None, link_order=None):
    """Plot a Robot in 2D"""
    link_pos = robot.forward_kinematics(q)
    num_links = link_pos.shape[1]

    if link_order is None:
        link_order = [
            (i, j) for i, j in zip(range(0, num_links), range(1, num_links))
        ]

    if handle_list is None:
        handle_list = []

        for link_pair in link_order:
            pt1 = link_pos[:, link_pair[0]]
            pt2 = link_pos[:, link_pair[1]]

            h1 = plt.plot(pt1[0],
                          pt1[1],
                          marker='o',
                          color='black',
                          linewidth=3 * lw)
            handle_list.append(h1[0])
            h2 = plt.plot(pt2[0],
                          pt2[1],
                          marker='o',
                          color='black',
                          linewidth=3 * lw)
            handle_list.append(h2[0])
            h3 = plt.plot(np.array([pt1[0], pt2[0]]),
                          np.array([pt1[1], pt2[1]]),
                          color='blue',
                          linewidth=lw)
            handle_list.append(h3[0])
    else:
        m = 0
        for link_pair in link_order:
            pt1 = link_pos[:, link_pair[0]]
            pt2 = link_pos[:, link_pair[1]]

            handle_list[m].set_data(pt1[0], pt1[1])
            m += 1
            handle_list[m].set_data(pt2[0], pt2[1])
            m += 1
            handle_list[m].set_data(np.array([pt1[0], pt2[0]]),
                                    np.array([pt1[1], pt2[1]]))
            m += 1
    return handle_list,


def plot_robot_3D(robot, q, lw, handle_list=None, link_order=None):
    """Plot a robot in 3D"""
    link_pos = robot.forward_kinematics(q)

    if handle_list is None:
        fig = plt.gcf()
        ax = plt.gca()
        handle_list = []

        if link_order is not None:
            for link_pair in link_order:
                pt1 = link_pos[:, link_pair[0]]
                pt2 = link_pos[:, link_pair[1]]
                h3 = ax.plot3D(np.array([pt1[0], pt2[0]]),
                               np.array([pt1[1], pt2[1]]),
                               np.array([pt1[2], pt2[2]]),
                               color='blue',
                               linewidth=lw,
                               marker='o',
                               markerfacecolor='black',
                               markersize=3 * lw)
                handle_list.append(h3[0])
        else:
            for n in range(link_pos.shape[1] - 1):
                pt1 = link_pos[:, n]
                pt2 = link_pos[:, n + 1]
                h1 = ax.scatter3D(pt1[0],
                                  pt1[1],
                                  pt1[2],
                                  color='black',
                                  linewidths=3 * lw)
                handle_list.append(h1)
                h2 = ax.scatter3D(pt2[0],
                                  pt2[1],
                                  pt2[2],
                                  color='black',
                                  linewidths=3 * lw)
                handle_list.append(h2)
                h3 = ax.plot3D(np.array([pt1[0], pt2[0]]),
                               np.array([pt1[1], pt2[1]]),
                               np.array([pt1[2], pt2[2]]),
                               color='blue',
                               linewidth=lw,
                               marker='o',
                               markerfacecolor='black',
                               markersize=3 * lw)
                handle_list.append(h3)

    else:
        if link_order is not None:
            m = 0
            for link_pair in link_order:
                pt1 = link_pos[:, link_pair[0]]
                pt2 = link_pos[:, link_pair[1]]

                # handle_list[m].set_data(pt1[0], pt1[1], pt1[2])
                # m += 1
                # handle_list[m].set_data(pt2[0], pt2[1], pt2[2])
                # m += 1
                handle_list[m].set_data(np.array([pt1[0], pt2[0]]),
                                        np.array([pt1[1], pt2[1]]))
                handle_list[m].set_3d_properties(np.array([pt1[2], pt2[2]]))
                m += 1
        else:
            m = 0
            for n in range(link_pos.shape[1] - 1):
                pt1 = link_pos[:, n]
                pt2 = link_pos[:, n + 1]

                handle_list[m].set_data(pt1[0], pt1[1], pt1[2])
                m += 1
                handle_list[m].set_data(pt2[0], pt2[1], pt2[2])
                m += 1
                handle_list[m].set_data(np.array([pt1[0], pt2[0]]),
                                        np.array([pt1[1], pt2[1]]),
                                        np.array([pt1[2], pt2[2]]))
                m += 1

    return handle_list,


def plot_tf_from_mat(transformation_mat, axis_scale=0.02, ax_handle=None):
    """Plot transform as 3-axis from tranformation matrix."""
    if ax_handle is None:
        ax = plt.gca()
    else:
        ax = ax_handle
    axis_colors = ['r', 'g', 'b']
    for d in range(3):
        e_d_master = axis_scale * np.eye(3)[:, d].reshape(-1, 1)
        e_d_obj = transform_pt(e_d_master, transformation_mat)
        origin_obj = transformation_mat[:3, -1].reshape(-1, 1)
        line_points = np.concatenate((origin_obj, e_d_obj), axis=1)
        ax.plot3D(line_points[0, :],
                  line_points[1, :],
                  line_points[2, :],
                  linewidth=4,
                  color=axis_colors[d])


def transform_pt(pt, transform_mat):
    """Transform a point to a different reference frame specified by `tranform_mat`."""
    pt = np.concatenate((pt, np.ones((1, 1))))
    pt_transformed = np.dot(transform_mat, pt)[0:3].reshape(-1, 1)

    return pt_transformed


def set_axes_radius(ax, origin, radius):
    """Set the size of the axes."""
    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])


def set_axes_equal(ax):
    """Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])

    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    set_axes_radius(ax, origin, radius)


def cuboid_data2(o, size=(1, 1, 1)):
    X = [[[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 1, 0]],
         [[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0]],
         [[1, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]],
         [[0, 0, 1], [0, 0, 0], [0, 1, 0], [0, 1, 1]],
         [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]],
         [[0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1]]]
    X = np.array(X).astype(float)
    for i in range(3):
        X[:, :, i] *= size[i]
    X += np.array(o)
    return X


def plotCubeAt2(positions, sizes=None, colors=None, **kwargs):
    if not isinstance(colors, (list, np.ndarray)):
        colors = ["C0"] * len(positions)
    if not isinstance(sizes, (list, np.ndarray)):
        sizes = [(1, 1, 1)] * len(positions)
    g = []
    for p, s, c in zip(positions, sizes, colors):
        g.append(cuboid_data2(p, size=s))
    return Poly3DCollection(np.concatenate(g),
                            facecolors=np.repeat(colors, 6),
                            **kwargs)


def plot_sphere(c, r, axis=None, alpha=1.0):
    if axis == None:
        axis = plt.gca()
    N = 50
    stride = 2
    u = np.linspace(0, 2 * np.pi, N)
    v = np.linspace(0, np.pi, N)
    x = c[0] + r * np.outer(np.cos(u), np.sin(v))
    y = c[1] + r * np.outer(np.sin(u), np.sin(v))
    z = c[2] + r * np.outer(np.ones(np.size(u)), np.cos(v))
    axis.plot_surface(x,
                      y,
                      z,
                      linewidth=0.0,
                      cstride=stride,
                      rstride=stride,
                      alpha=alpha)


def plot_cylinder(p0, p1, R, ax=None):
    if ax == None:
        ax = plt.gca()

    v = p1 - p0
    # find magnitude of vector
    mag = norm(v)
    # unit vector in direction of axis
    v = v / mag
    # make some vector not in the same direction as v
    not_v = np.array([1, 0, 0])
    if (v == not_v).all():
        not_v = np.array([0, 1, 0])
    # make vector perpendicular to v
    n1 = np.cross(v, not_v)
    # normalize n1
    n1 /= norm(n1)
    # make unit vector perpendicular to v and n1
    n2 = np.cross(v, n1)
    # surface ranges over t from 0 to length of axis and 0 to 2*pi
    t = np.linspace(0, mag, 100)
    theta = np.linspace(0, 2 * np.pi, 100)
    # use meshgrid to make 2d arrays
    t, theta = np.meshgrid(t, theta)
    # generate coordinates for surface
    X, Y, Z = [
        p0[i] + v[i] * t + R * np.sin(theta) * n1[i] +
        R * np.cos(theta) * n2[i] for i in [0, 1, 2]
    ]
    ax.plot_surface(X, Y, Z)
    # plot axis
    ax.plot(*zip(p0, p1), color='red')
    # ax.set_xlim(0, 10)
    # ax.set_ylim(0, 10)
    # ax.set_zlim(0, 10)
    # plt.show()
