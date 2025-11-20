import numpy as np
from trajectory import Trajectory

waypoints = np.array([
    [0, 0, 0.0],
    [1, 0, 0.5],
    [1, 1, 1.0],
    [0, 1, 1.5]
])

traj = Trajectory(waypoints)
print(traj.sample_trajectory(0.25))