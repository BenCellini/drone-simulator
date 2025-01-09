
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import figurefirst as fifi
import figure_functions as ff
from pybounds import Simulator


class DroneParameters:
    """ Stores drone parameters
    """
    def __init__(self):
        self.g = 9.81  # gravity [m/s^2]
        self.m = 0.086  # mass [kg]
        self.M = 2.529  # mass [kg]
        self.Mm = 4 * self.m + self.M  # total mass [kg]
        self.L = 0.2032  # length [m]
        self.R = 0.1778  # average body radius [m]
        self.Ix = 2 * (self.M * self.R ** 2) / 5 + 2 * self.m * self.L ** 2  # [kg*m^2] moment of inertia about x
        self.Iy = 2 * (self.M * self.R ** 2) / 5 + 2 * self.m * self.L ** 2  # [kg*m^2] moment of inertia about y
        self.Iz = 2 * (self.M * self.R ** 2) / 5 + 4 * self.m * self.L ** 2  # [kg*m^2] moment of inertia about y
        self.Jr = 2 * (self.M * self.R ** 2) / 5  # rotor inertia
        self.b = 1.8311  # thrust coefficient
        self.d = 0.01  # drag constant
        self.Dl = 0.1  # drag coefficient from ground speed plus air speed
        self.Dr = 0.1  # drag coefficient from rotation speed


state_names = ['x',  # x position [m]
               'v_x',  # x velocity [m/s]
               'y',  # y position [m]
               'v_y',  # y velocity [m/s]
               'z',  # z position (altitude) [m]
               'v_z',  # z velocity [m/s]
               'phi',  # roll [rad]
               'phi_dot',  # roll rate [rad/s]
               'theta',  # roll [rad]
               'theta_dot',  # roll rate [rad/s]
               'psi',  # roll [rad]
               'psi_dot',  # roll rate [rad/s]
               'w',  # wind magnitude [m/s]
               'zeta',  # wind direction [rad]
               'w_z',  # wind z velocity [m/s]
               'm',  # mass
               'Ix',  #
               'Iy',  #
               'Iz',  #
               'Jr',  #
               'b',  #
               'd',  #
               'Dl',  #
               'Dr',  #
               ]

input_names = ['u_1',  # rotor #1 speed
               'u_2',  # rotor #2 speed
               'u_3',  # rotor #3 speed
               'u_4',  # rotor #4 speed
               ]


def f(X, U):
    params = DroneParameters()

    # States
    x, v_x, y, v_y, z, v_z, phi, phi_dot, theta, theta_dot, psi, psi_dot, w, zeta, w_z, m, Ix, Iy, Iz, Jr, b, d, Dl, Dr = X

    # Inputs
    u_1, u_2, u_3, u_4 = U

    U1 = b * (u_1 ** 2 + u_2 ** 2 + u_3 ** 2 + u_4 ** 2)
    U2 = b * (u_1 ** 2 + u_4 ** 2 - u_2 ** 2 - u_3 ** 2)
    U3 = b * (u_3 ** 2 + u_4 ** 2 - u_1 ** 2 - u_2 ** 2)
    # U4 = d * (-u_1 ** 2 + u_2 ** 2 - u_3 ** 2 + u_4 ** 2)
    U4 = d * (u_1 ** 2 - u_2 ** 2 + u_3 ** 2 - u_4 ** 2)
    omega = u_2 + u_4 - u_1 - u_3

    # Drag dynamics
    w_x = w * np.cos(zeta)
    w_y = w * np.sin(zeta)
    vr_x = v_x + w_x
    vr_y = v_y + w_y
    vr_z = v_z + w_z

    # Dynamics
    x_dot = v_x
    y_dot = v_y
    z_dot = v_z

    v_x_dot = (np.cos(phi) * np.sin(theta) * np.cos(psi) + np.sin(phi) * np.sin(psi)) * U1 / m - Dl * vr_x / m
    v_y_dot = (np.cos(phi) * np.sin(theta) * np.sin(psi) - np.sin(phi) * np.cos(psi)) * U1 / m - Dl * vr_y / m
    v_z_dot = (np.cos(phi) * np.cos(theta)) * U1 / m - Dl * vr_z / m - params.g

    phi_ddot = theta_dot * psi_dot * (Iy - Iz) / (Ix) - Jr * theta_dot * omega / Ix + U2 * params.L / Ix - phi_dot * Dr / Ix
    theta_ddot = phi_dot * psi_dot * (Iz - Ix) / (Iy) + Jr * phi_dot * omega / Iy + U3 * params.L / Iy - theta_dot * Dr / Iy
    psi_ddot = phi_dot * theta_dot * (Ix - Iy) / (Iz) + U4 / Iz - psi_dot * Dr / Iz

    w_dot = 0.0 * x
    zeta_dot = 0.0 * x
    w_z_dot = 0.0 * x

    m_dot = 0.0 * x
    Ix_dot = 0.0 * x
    Iy_dot = 0.0 * x
    Iz_dot = 0.0 * x
    Jr_dot = 0.0 * x
    b_dot = 0.0 * x
    d_dot = 0.0 * x
    Dl_dot = 0.0 * x
    Dr_dot = 0.0 * x

    # Package and return xdot
    x_dot = [x_dot, v_x_dot, y_dot, v_y_dot, z_dot, v_z_dot,
             phi_dot, phi_ddot, theta_dot, theta_ddot, psi_dot, psi_ddot,
             w_dot, zeta_dot, w_z_dot,
             m_dot, Ix_dot, Iy_dot, Iz_dot, Jr_dot, b_dot, d_dot, Dl_dot, Dr_dot]

    return x_dot


measurement_names = ['x', 'y', 'z',
                     'phi', 'theta', 'psi', 'Psi',
                     'phi_dot', 'theta_dot', 'psi_dot',
                     'v_x', 'v_y', 'v_z', 'v', 'beta',
                     'v_x_dot', 'v_y_dot', 'v_z_dot', 'v_dot', 'alpha',
                     'r_x', 'r_y', 'r_z', 'r',
                     'a_x', 'a_y', 'a_z', 'a', 'gamma',
                     'w', 'zeta',
                     'm', 'Ix', 'Iy', 'Iz', 'Jr', 'b', 'd', 'Dl', 'Dr']


def h(X, U):
    params = DroneParameters()

    # States
    x, v_x, y, v_y, z, v_z, phi, phi_dot, theta, theta_dot, psi, psi_dot, w, zeta, w_z, m, Ix, Iy, Iz, Jr, b, d, Dl, Dr = X

    # Inputs
    u_1, u_2, u_3, u_4 = U

    U1 = b * (u_1 ** 2 + u_2 ** 2 + u_3 ** 2 + u_4 ** 2)
    U2 = b * (u_4 ** 2 + u_1 ** 2 - u_2 ** 2 - u_3 ** 2)
    U3 = b * (u_3 ** 2 + u_4 ** 2 - u_1 ** 2 - u_2 ** 2)
    U4 = d * (-u_1 ** 2 + u_2 ** 2 - u_3 ** 2 + u_4 ** 2)
    omega = u_2 + u_4 - u_1 - u_3

    # Drag dynamics
    w_x = w * np.cos(zeta)
    w_y = w * np.sin(zeta)
    a_x = v_x + w_x
    a_y = v_y + w_y
    a_z = v_z + w_z
    a = np.sqrt(a_x ** 2 + a_y ** 2)

    # Apparent wind angle
    gamma = np.arctan2(a_y, a_x)

    # Course direction
    beta = np.arctan2(v_y, v_x)

    # Optic flow magnitude
    r_x = v_x / z
    r_y = v_y / z
    r_z = v_z / z
    v = np.sqrt(v_x ** 2 + v_y ** 2)
    r = v / z

    # Acceleration
    v_x_dot = (np.cos(phi) * np.sin(theta) * np.cos(psi) + np.sin(phi) * np.sin(psi)) * U1 / m - Dl * a_x / m
    v_y_dot = (np.cos(phi) * np.sin(theta) * np.sin(psi) - np.sin(phi) * np.cos(psi)) * U1 / m - Dl * a_y / m
    v_z_dot = (np.cos(phi) * np.cos(theta)) * U1 / m - Dl * a_z / m - params.g
    v_dot = np.sqrt(v_x ** 2 + v_y ** 2)

    alpha = np.arctan2(v_y_dot, v_x_dot)

    # Global yaw
    yaw_global = calculate_global_yaw(phi, theta, psi)

    # Unwrap angles
    if np.array(phi).ndim > 0:
        if np.array(phi).shape[0] > 1:
            phi = np.unwrap(phi)
            theta = np.unwrap(theta)
            psi = np.unwrap(psi)
            gamma = np.unwrap(gamma)
            beta = np.unwrap(beta)
            alpha = np.unwrap(alpha)

    # Measurements
    Y = [x, y, z,
         phi, theta, psi, yaw_global,
         phi_dot, theta_dot, psi_dot,
         v_x, v_y, v_z, v, beta,
         v_x_dot, v_y_dot, v_z_dot, v_dot, alpha,
         r_x, r_y, r_z, r,
         a_x, a_y, a_z, a, gamma,
         w, zeta,
         m, Ix, Iy, Iz, Jr, b, d, Dl, Dr]

    # Return measurement
    return Y


class DroneSimulator(Simulator):
    def __init__(self, dt=0.1, mpc_horizon=10, r_u=1e-1):
        super().__init__(f, h, dt=dt, mpc_horizon=mpc_horizon,
                         state_names=state_names,
                         input_names=input_names,
                         measurement_names=measurement_names)

        # Set parameters
        self.params = DroneParameters()

        # Define cost function: penalize the squared error between parallel & perpendicular velocity and heading
        cost = (1.0 * (self.model.x['v_x'] - self.model.tvp['v_x_set']) ** 2 +
                1.0 * (self.model.x['v_y'] - self.model.tvp['v_y_set']) ** 2 +
                1.0 * (self.model.x['z'] - self.model.tvp['z_set']) ** 2 +
                1.0 * (self.model.x['psi'] - self.model.tvp['psi_set']) ** 2)

        # Set cost function
        self.mpc.set_objective(mterm=cost, lterm=cost)

        # Set input penalty: make this small for accurate state following
        self.mpc.set_rterm(u_1=r_u, u_2=r_u, u_3=r_u, u_4=r_u)

        # Place limit on controls
        self.mpc.bounds['lower', '_u', 'u_1'] = 0
        self.mpc.bounds['lower', '_u', 'u_2'] = 0
        self.mpc.bounds['lower', '_u', 'u_3'] = 0
        self.mpc.bounds['lower', '_u', 'u_4'] = 0

        # Place limit on states
        self.mpc.bounds['lower', '_x', 'z'] = 0

        self.mpc.bounds['upper', '_x', 'phi'] = np.pi / 4
        self.mpc.bounds['upper', '_x', 'theta'] = np.pi / 4

        self.mpc.bounds['lower', '_x', 'phi'] = -np.pi / 4
        self.mpc.bounds['lower', '_x', 'theta'] = -np.pi / 4

    def update_setpoint(self, vx=None, vy=None, psi=None, z=None, w=None, zeta=None):
        """ Set the set-point variables.
        """

        # Make sure set-points are same size
        if not np.all(np.array([len(vx), len(vy), len(psi), len(z), len(w), len(zeta)])):
            raise ValueError('vx, vy, psi, z, w, & zeta must be of equal length')

        # Set time
        T = self.dt * (len(vx) - 1)
        tsim = np.arange(0, T + self.dt / 2, step=self.dt)

        # Define the set-points to follow
        setpoint = {'x': 0.0 * np.ones_like(tsim),
                    'v_x': vx,
                    'y': 0.0 * np.ones_like(tsim),
                    'v_y': vy,
                    'z': z,
                    'v_z': 0.0 * np.ones_like(tsim),
                    'phi': 0.0 * np.ones_like(tsim),
                    'phi_dot': 0.0 * np.ones_like(tsim),
                    'theta': 0.0 * np.ones_like(tsim),
                    'theta_dot': 0.0 * np.ones_like(tsim),
                    'psi': 1.0 * psi,
                    'psi_dot': 0.0 * np.ones_like(tsim),
                    'w': w,
                    'zeta': zeta,
                    'w_z': 0.1 * np.ones_like(tsim),
                    'm': self.params.Mm * np.ones_like(tsim),
                    'Ix': self.params.Ix * np.ones_like(tsim),
                    'Iy': self.params.Iy * np.ones_like(tsim),
                    'Iz': self.params.Iz * np.ones_like(tsim),
                    'Jr': self.params.Jr * np.ones_like(tsim),
                    'b': self.params.b * np.ones_like(tsim),
                    'd': self.params.d * np.ones_like(tsim),
                    'Dl': self.params.Dl * np.ones_like(tsim),
                    'Dr': self.params.Dr * np.ones_like(tsim),
                    }

        # Update the simulator set-point
        self.update_dict(setpoint, name='setpoint')

    def plot_trajectory(self, start_index=0, dpi=200, size_radius=None):
        """ Plot the trajectory.
        """

        fig, ax = plt.subplots(1, 1, figsize=(3 * 1, 3 * 1), dpi=dpi)

        x = self.y['x'][start_index:]
        y = self.y['y'][start_index:]
        heading = self.y['psi'][start_index:]
        time = self.time[start_index:]

        if size_radius is None:
            size_radius = 0.03 * np.max(np.array([range_of_vals(x), range_of_vals(y)]))
            # print('size radius:', np.round(size_radius, 4))

        ff.plot_trajectory(x, y, heading,
                           color=time,
                           ax=ax,
                           size_radius=size_radius,
                           nskip=0)

        fifi.mpl_functions.adjust_spines(ax, [])


def range_of_vals(x, axis=0):
    return np.max(x, axis=axis) - np.min(x, axis=axis)


def calculate_global_yaw(roll, pitch, magnetometer_yaw):
    """
    Calculate the global yaw (in radians) for a quadcopter using the roll, pitch, and magnetometer yaw.

    Parameters:
    - roll (float): The roll angle of the quadcopter in radians.
    - pitch (float): The pitch angle of the quadcopter in radians.
    - magnetometer_yaw (float): The magnetometer yaw (magnetic heading) in radians.

    Returns:
    - float: The global yaw in radians.
    """
    # Calculate the adjustment based on roll and pitch
    adjustment = np.arctan2(np.sin(roll) * np.sin(pitch), np.cos(pitch))

    # Global yaw is the magnetometer yaw plus the adjustment
    global_yaw = magnetometer_yaw + adjustment

    # Normalize the global yaw to be between -pi and pi
    global_yaw = (global_yaw + np.pi) % (2 * np.pi) - np.pi

    return global_yaw
