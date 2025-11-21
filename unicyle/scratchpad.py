import numpy as np
from trajectory import *

points = np.array([
    (0.0, 0.0, 0.0),
    (3.0, 0.0, 5.0),
    (3.0, 3.0, 13.0),
    (0.0, 3.0, 22.0),
    (0.0, 0.0, 30.0)
])

traj = Trajectory(points)

times = np.arange(0, 31, 0.1)
for t in times:
    print(t)
    print(traj.sample_trajectory(t))