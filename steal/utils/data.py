import torch
from torchdiffeq import odeint


def generate_trajectories(model,
                          x_init,
                          time_dependent=False,
                          order=2,
                          return_label=False,
                          filename=None,
                          t_init=0.,
                          t_final=10.,
                          t_step=0.01,
                          method='rk4'):
    '''
	generate roll-out trajectories of the given dynamical model
	:param model (torch.nn.Module): dynamical model
	:param x_init (torch.Tensor): initial condition
	:param order (int): 1 if first order system, 2 if second order system
	:param return_label (bool): set True to return the velocity (order=1) / acceleration (order=2) along the trajectory
	:param filename (string): if not None, save the generated trajectories to the file
	:param t_init (float): initial time for numerical integration
	:param t_final (float): final time for numerical integration
	:param t_step (float): step for t_eval for numerical integration
	:param method (string): method of integration (ivp or euler)
	:return: state (torch.Tensor) if return_label is False
	:return: state (torch.Tensor), control (torch.Tensor) if return label is True
	'''

    # make sure that the initial condition has dim 1
    if x_init.ndim == 1:
        x_init = x_init.reshape(1, -1)

    # dynamics for numerical integration
    if order == 2:
        # dynamics for the second order system
        def dynamics(t, state):
            # for second order system, the state includes
            # both position and velocity
            n_dims = state.shape[-1] // 2
            x = state[:, :n_dims]
            x_dot = state[:, n_dims:]

            # compute the acceleration under the given model
            if time_dependent:
                y_pred = model(t=t, x=x, xd=x_dot)
            else:
                y_pred = model(x, x_dot)

            # for force model, both the force and the metric are returned
            if isinstance(y_pred, tuple):
                f_pred, g_pred = y_pred

                # compute the acceleration: a = inv(G)*f
                x_ddot = torch.einsum(
                    'bij,bj->bi', torch.inverse(g_pred), f_pred).detach(
                    )  # detach the tensor from the computational graph
            else:
                # otherwise, only the acceleration is returned
                x_ddot = y_pred.detach()

            # the time-derivative of the state includes
            # both velocity and acceleration
            state_dot = torch.cat((x_dot, x_ddot), dim=1)
            return state_dot

    elif order == 1:
        # dynamics for the first order system
        def dynamics(t, state):
            # for first order systems, the state is the position

            # compute the velocity under the given model
            if time_dependent:
                y_pred = model(t=t, x=state)
            else:
                y_pred = model(state)
            if isinstance(y_pred, tuple):
                p_pred, g_pred = y_pred
                x_dot = torch.einsum('bij,bj->bi', torch.inverse(g_pred),
                                     p_pred).detach()
            else:
                x_dot = y_pred.detach()
            return x_dot
    else:
        raise TypeError('Unknown order!')

    # the times at which the trajectory is computed
    t_eval = torch.arange(t_init, t_final, t_step)

    x_data = odeint(dynamics, x_init, t_eval, method=method)

    # if the control inputs along the trajectory are also needed,
    # compute the control inputs (useful for generating datasets)
    if return_label:
        y_data = dynamics(t_eval, x_data)
        data = (x_data, y_data)
    else:
        data = x_data

    if filename is not None:
        torch.save(data, filename)

    return data
