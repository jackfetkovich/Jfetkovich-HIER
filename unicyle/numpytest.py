import numpy as np



traj = np.linspace(0, 19, 20)
T = 10
Tx = 160
print(traj)
for t in range(Tx):
    print(traj[t+1: min(t+1+T, len(traj))])