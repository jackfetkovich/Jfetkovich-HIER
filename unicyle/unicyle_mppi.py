import numpy as np
from scipy.interpolate import interp1d
from numba import njit
import csv
from animation import animate
from utils import *
from trajectory import *
from mppi import *
from parameters import *

params = Parameters(
    dt = 0.05, # time step
    K = 500,   # number of samples
    T = 8, # time steps (HORIZON)
    sigma = 2,
    lambda_ = 2,
    max_v = 5.1, # max x velocity (m/s)
    max_w = 12.0, # max angular velocity (radians/s)
    max_v_dot = 8.0, # max linear acceleration (m/s^2)
    max_w_dot = 30.0 # max angular acceleration (radians/s^2) (8.0)
)

obstacles = np.array([[3.85, 3.8, 0.5]])

# Main function
def main():
    time = []
    x_pos = []
    y_pos = []
   
    # Original (x, y) points
    points = [
       (0.0, 0.0),
        (0.5, 0.0),
        (1.0, 0.0),
        (1.5, 0.0),
        (2.0, 0.1),
        (2.5, 0.4),
        (3.0, 0.9),
        (3.4, 1.4),
        (3.7, 2.0),
        (3.85, 2.6),
        (3.9, 3.2),
        (3.85, 3.8),
        (3.7, 4.3),
        (3.4, 4.8),
        (3.0, 5.1),
        (2.5, 5.3),
        (2.0, 5.4),
        (1.5, 5.4),
        (1.0, 5.3),
        (0.5, 5.1),
        (0.1, 4.8),
        (-0.2, 4.4),
        (-0.4, 4.0),
        (-0.5, 3.5),
        (-0.5, 3.0),
        (-0.4, 2.5),
        (-0.2, 2.0),
        (0.1, 1.6),
        (0.5, 1.3),
        (1.0, 1.1),
        (1.5, 1.0),
        (2.0, 1.0),
        (2.5, 1.0)
    ]

    # Compute forward tangents
    headings = []
    for i in range(len(points)):
        if i < len(points)-1:
            dx = points[i+1][0] - points[i][0]
            dy = points[i+1][1] - points[i][1]
            headings.append(np.arctan2(dy, dx))
        else:
            headings.append(headings[-1])  # reuse last heading for final point

    # Unwrap for continuity
    headings_unwrapped = np.unwrap(headings)

    # Combine into (x, y, theta)
    waypoints = [(p[0], p[1], th) for p, th in zip(points, headings_unwrapped)]
    
    Tx = int(distance_of_path(np.array(points)) / (params.max_v*0.2941176*params.dt))
    x = np.array([0,0,0,0,0])  # Initial state [x, theta, x_dot, theta_dot] -- tracks current state
    X = np.zeros((Tx, 5)) # list of historical states
    U = np.zeros((Tx, 2)) # list of historical control inputs
    all_weights = np.zeros((Tx, params.K)) # Weights of every generated trajectory, organized by time step
    costs = np.zeros(Tx)
    traj = generate_trajectory_from_waypoints(waypoints, Tx) # trajectory of waypoints
    # traj[:, 2] = new_thetas
    # np.savetxt('trajectory.csv', traj, delimiter=',', fmt='%.4f')

    sample_trajectories = np.zeros((Tx, params.K, 3, params.T))
    sample_trajectories_one = np.zeros((params.K, 3, params.T)) # k sets of (x1, x2, ..., xn), (y1, y2, ..., yn), (w1, w2, ..., wn)
    last_u = np.zeros(2) # the control input from the previous iteration
    for t in range(Tx-1): # From 0 -> 159
        u_nom, X_calc, traj_weight_single = mppi(x, last_u, traj[t+1: min(t+1+params.T, len(traj))], params) # Calculate the optimal control input
        # U[t] = safety_filter(u_nom, x)
        U[t] = u_nom
        x = unicyle_dynamics(x, U[t], params) # Calculate what happens when you apply that input
        X[t + 1, :] = x # Store the new state
        time.append(t)
        x_pos.append(X[t+1, 0]) # Save the x position at this timestep
        y_pos.append(X[t+1, 1]) # Save the y position at this timestep
        last_u = U[t] # Save the control input 
        costs[t] = cost_function(x, U[t], traj[t+1])

        for k in range(params.K):
            for t_ in range (params.T): # Reshaping trajectory weight list for use in animation
                sample_trajectories_one[k, 0, t_] = X_calc[k, t_, 0] #should be 0
                sample_trajectories_one[k, 1, t_] = X_calc[k, t_, 1] #should be 1
        sample_trajectories[t] = sample_trajectories_one # Save the sampled trajectories
        all_weights[t] = traj_weight_single # List of the weights, populated in mppi function
        
        # with open("data3.txt", "a") as f: 
        #     f.write(f"t: {t},  X=(x={x[0]}, y={x[1]}, th={x[2]}), U=(v={u_nom[0]}, w={u_nom[1]})\n")
        #     f.write(f"Close pt x={traj[best_traj_idx][0]}, y={traj[best_traj_idx][1]}, th={traj[best_traj_idx][2]}\n")
        #     f.write(f"idx{best_traj_idx}\n")
        #     f.write("-------------------\n")
    
    with open('circle_discrepancy_time.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Step',  'X_d', 'Y_d', 'Theta_d', 'X', 'Y', 'Theta', 'Cost'])
        for t in range(Tx):
            writer.writerow(np.array([t,traj[t][0], traj[t][1], traj[t][2],X[t][0], X[t][1], X[t][2], costs[t]]))


    animate(x_pos, y_pos, traj[:, 0], traj[:, 1], sample_trajectories, all_weights, params)


if __name__ == "__main__":
    main()

