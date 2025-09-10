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
    dt = 0.05, # time step for MPPI
    safety_dt = 0.025, # time step for safety
    K = 1500,   # number of samples
    T = 8, # time steps (HORIZON)
    sigma = 2,
    lambda_ = 2,
    l = 0.1,
    max_v = 5.1, # max x velocity (m/s)
    max_w = 10.0, # max angular velocity (radians/s)
    max_v_dot = 8.0, # max linear acceleration (m/s^2)
    max_w_dot = 30.0, # max angular acceleration (radians/s^2) (8.0)
    obstacles = np.array([[0.5, 0.05, 0.1]])
)

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

    obstacle_points = [
        (0.5, 0.005),
        (0.5, 0.0),
        (0.5, 0.0),
        (0.5, 0.0),
        (0.5, 0.0),
        (0.5, 0.0),
        (0.5, 0.0),
        (0.5, 0.0),
        (0.5, 0.0),
        (0.5, 0.0),
        (0.5, 0.0),
        (0.5, 0.0),
        (0.5, 0.0),
        (0.5, 0.0),
        (0.5, 0.0),
        (0.5, 0.0),
        (0.5, 0.0),
        (0.5, 0.0),
        (0.5, 0.0),
        (0.5, 0.0),
        (0.5, 0.0),
        (0.5, 0.0),
        (0.5, 0.0),
        (0.5, 0.0),
        (0.5, 0.0),
        (0.5, 0.0),
        (0.5, 0.0),
        (0.5, 0.0),
        (0.5, 0.0),
        (0.5, 0.0),
        (0.5, 0.0),
        (0.5, 0.0),
    ]

    # obstacle_points = [
    #     (0.5, 0.4),
    #     (0.5, 0.3),
    #     (1.7, -0.3),
    #     (1.8, 0.05),
    #     (2.0, 0.45),
    #     (2.8, 0.3),
    #     (3.0, 1.2),
    #     (3.7, 1.3),
    #     (3.5, 2.25),
    #     (4.0, 2.6),
    #     (3.6, 3.3), #11
    #     (0.0, 0.0),
    #     (0.0, 0.0),
    #     (0.0, 0.0),
    #     (0.0, 0.0),
    #     (0.0, 0.0),
    #     (0.0, 0.0),
    #     (0.0, 0.0),
    #     (0.0, 0.0),
    #     (0.0, 0.0),
    #     (0.0, 0.0),
    #     (0.0, 0.0),
    #     (0.0, 0.0),
    #     (0.0, 0.0),
    #     (0.0, 0.0),
    #     (0.0, 0.0),
    #     (0.0, 0.0),
    #     (0.0, 0.0),
    #     (0.0, 0.0),
    #     (0.0, 0.0),
    #     (0.0, 0.0),
    #     (0.0, 0.0),
    #     (0.0, 0.0)
    # ]

    ### Zeroed arrays used for calcuation
    Tx = int(distance_of_path(np.array(points)) / (params.max_v*0.2941176*params.dt))*2
    x = np.array([0,0,0.2,0,0])  # Initial state [x, theta, x_dot, theta_dot] -- tracks current state
    X = np.zeros((Tx, 5)) # list of historical states
    U = np.zeros((Tx, 2)) # list of historical control inputs
    all_weights = np.zeros((Tx, params.K)) # Weights of every generated trajectory, organized by time step
    costs = np.zeros(Tx)
    sample_trajectories = np.zeros((Tx, params.K, 3, params.T))
    sample_trajectories_one = np.zeros((params.K, 3, params.T)) # k sets of (x1, x2, ..., xn), (y1, y2, ..., yn), (w1, w2, ..., wn)
    last_u = np.zeros(2) # the control input from the previous 
    
    x_ob = np.zeros(Tx)
    y_ob = np.zeros(Tx)
    ## Zeroed arrays used for calculation
    
    ## Generation of waypoints for obstacle and robot
    traj = generate_trajectory_from_waypoints(points, int(Tx/2)+1) # trajectory of waypoints
    obstacle_traj = generate_trajectory_x_y(obstacle_points, int(Tx/2)+1) # Moving obstacle trajectory

    # with open('./data/safety_filter.csv', 'w', newline='', encoding='utf-8') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(['Step',  'u_nom_v', 'u_nom_w', 'u_safety_v', 'u_safety_w'])
    
    for t in range(Tx-1):
        if t % 2 == 0:
            u_nom, X_calc, traj_weight_single = mppi(x, last_u, traj[int(t/2)+1: min(int(t/2)+1+params.T, len(traj))], params) # Calculate the optimal control input
            for k in range(params.K):
                for t_ in range (params.T): # Reshaping trajectory weight list for use in animation
                    sample_trajectories_one[k, 0, t_] = X_calc[k, t_, 0] #should be 0
                    sample_trajectories_one[k, 1, t_] = X_calc[k, t_, 1] #should be 1
            sample_trajectories[t] = sample_trajectories_one # Save the sampled trajectories
            all_weights[t] = traj_weight_single # List of the weights, populated in mppi function
            costs[t] = cost_function(x, U[t], traj[int(t/2)+1])
        else:
            # params.obstacles[0] = np.array([obstacle_traj[int(t/2), 0], obstacle_traj[int(t/2),1], params.obstacles[0,2]])
            params.obstacles[0] = np.array([obstacle_traj[int(t/2), 0], 0.05, params.obstacles[0,2]])
            base = np.array([np.ones(params.T) * X[t, 0], np.ones(params.T) * X[t, 1], np.zeros(params.T)])
            sample_trajectories[t] = np.repeat(base[np.newaxis, :, :], params.K, axis=0)
            all_weights[t] = np.ones(params.K)
            costs[t] = cost_function(x, U[t], traj[int(t/2)+1])
        x_ob[t] = params.obstacles[0][0]
        y_ob[t] = params.obstacles[0][1]
        U[t] = safety_filter(u_nom, x, params)
        # U[t] = u_nom
        
        # with open('./data/safety_filter.csv', 'a', newline='', encoding='utf-8') as file:
        #     writer = csv.writer(file)
        #     writer.writerow([t, u_nom[0], u_nom[1], U[t][0], U[t][1]])
        
        x = unicyle_dynamics(x, U[t], params, dt=params.safety_dt) # Calculate what happens when you apply that input
        X[t + 1, :] = x # Store the new state
        time.append(t)
        x_pos.append(X[t+1, 0]) # Save the x position at this timestep
        y_pos.append(X[t+1, 1]) # Save the y position at this timestep
        last_u = U[t] # Save the control input 
        
        
        
        
    animate(x_pos, y_pos, traj[:, 0], traj[:, 1], x_ob, y_ob, sample_trajectories, all_weights, params)


if __name__ == "__main__":
    main()

