import numpy as np
from utils import *
from scipy.interpolate import interp1d
from numba import int64, float64, boolean    # import the types
from numba.experimental import jitclass

def generate_trajectory_from_waypoints(waypoints, num_points=100):
    """
    Generates a smooth trajectory by interpolating between given waypoints.

    Parameters:
    waypoints (list of tuples): List of (x, y, theta) waypoints.
    num_points (int): Number of points to interpolate.

    Returns:
    tuple: (x_vals, y_vals, theta_vals) interpolated trajectory.
    """

    headings = []
    for i in range(len(waypoints)):
        if i < len(waypoints)-1:
            dx = waypoints[i+1][0] - waypoints[i][0]
            dy = waypoints[i+1][1] - waypoints[i][1]
            headings.append(np.arctan2(dy, dx))
        else:
            headings.append(headings[-1])  # reuse last heading for final point

    # Unwrap for continuity
    headings_unwrapped = np.unwrap(headings)

    # Combine into (x, y, theta)
    waypoints = [(p[0], p[1], th) for p, th in zip(waypoints, headings_unwrapped)]
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
    # np.savetxt("output.txt", np.array(theta_vals), fmt="%.6f")  # 6 decimal places

    return np.dstack(np.array([x_vals, y_vals, np.unwrap(np.array(theta_vals))]))[0]


# Put in list of waypoints with associated time of arrival
# Assign heading to each waypoint to be tangent
# Be able to poll a trajectory at a time to get a position value
spec = [
    ('waypoints', float64[:, :]),
]
@jitclass(spec)
class Trajectory:
    def __init__(self, waypoints):
        headings = np.zeros(len(waypoints)) # Generate headings
        for i in range(len(waypoints)):
            if i < len(waypoints)-1:
                dx = waypoints[i+1][0] - waypoints[i][0]
                dy = waypoints[i+1][1] - waypoints[i][1]
                headings[i] = np.arctan2(dy, dx)
            else:
                headings[i] = headings[-1]  # reuse last heading for final point
        headings = headings.reshape(-1, 1)
        points_angles = np.hstack((waypoints, headings))
        reach_times = np.copy(waypoints[:, 2]).reshape(-1,1)
        print(reach_times)
        points_angles_headings = np.hstack((points_angles, reach_times))
        self.waypoints = points_angles_headings

    def sample_trajectory(self, t):
        mask = self.waypoints[:, 3] >= t
        if np.any(mask):
            smallest_idx = np.argmax(mask) # Find the first waypoint with larger time than requested
            if smallest_idx == 0:
                return self.waypoints[0, 0:3] # Return first waypoint

            return lin_interpolate(self.waypoints[smallest_idx, 0:3], 
                                   self.waypoints[smallest_idx-1, 0:3], 
                                   (t - self.waypoints[smallest_idx -1, 3]) / (self.waypoints[smallest_idx, 3] - self.waypoints[smallest_idx -1, 3])
                                   )
        else:
            return self.waypoints[-1, 0:3]



def generate_trajectory_x_y(waypoints, num_points):
    waypoints = np.array(waypoints)
    t = np.linspace(0, 1, len(waypoints))  # Normalized parameter along waypoints
    t_interp = np.linspace(0, 1, num_points)  # Fine-grained interpolation parameter

    # Interpolating x, y
    interp_x = interp1d(t, waypoints[:, 0], kind='linear')
    interp_y = interp1d(t, waypoints[:, 1], kind='linear')

    x_vals = interp_x(t_interp)
    y_vals = interp_y(t_interp)
    return np.dstack(np.array([x_vals, y_vals]))[0]
