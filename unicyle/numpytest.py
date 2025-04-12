import numpy as np
from scipy.interpolate import interp1d

def generate_trajectory_from_waypoints(waypoints, num_points=100):
    """
    Generates a smooth trajectory by interpolating between given waypoints.

    Parameters:
    waypoints (list of tuples): List of (x, y, theta) waypoints.
    num_points (int): Number of points to interpolate.

    Returns:
    tuple: (x_vals, y_vals, theta_vals) interpolated trajectory.
    """
    waypoints = np.array(waypoints)
    t = np.linspace(0, 1, len(waypoints))  # Normalized parameter along waypoints
    t_interp = np.linspace(0, 1, num_points)  # Fine-grained interpolation parameter

    # Interpolating x, y, and theta
    interp_x = interp1d(t, waypoints[:, 0], kind='linear')
    interp_y = interp1d(t, waypoints[:, 1], kind='linear')
    interp_theta = interp1d(t, waypoints[:, 2], kind='linear')

    x_vals = interp_x(t_interp)
    y_vals = interp_y(t_interp)
    theta_vals = interp_theta(t_interp)

    return x_vals, y_vals, theta_vals

waypoints = [
        (0, 0, 0),
        (0.1, 0.2, np.pi / 8),
        (0.3, 0.5, np.pi / 6),
        (0.6, 0.7, np.pi / 4),
        (1.0, 0.8, np.pi / 3),
        (1.3, 0.6, np.pi / 2),
        (1.5, 0.3, 3*np.pi / 4),
        (1.66, 0, np.pi)
    ]
T=30
traj = generate_trajectory_from_waypoints(waypoints, 1000)
t=2
targets = np.array([
        traj[0][t:t+T], traj[1][t:t+T], traj[2][t:t+T], np.zeros(T), np.zeros(T), np.zeros(T)
])
print(targets)
print(targets.T)
