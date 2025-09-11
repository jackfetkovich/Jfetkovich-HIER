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
    l = 0.001,
    max_v = 5.1, # max x velocity (m/s)
    max_w = 10.0, # max angular velocity (radians/s)
    max_v_dot = 8.0, # max linear acceleration (m/s^2)
    max_w_dot = 30.0, # max angular acceleration (radians/s^2) (8.0)
    obstacles = np.array([[0.5, 0.05, 0.1], [3.0, 3.0, 0.2]])
)

# Main function
def main():
    time = []
    x_pos = []
    y_pos = []
   
    # Original (x, y) points
    points = [
        (-0.000, -0.011),
        (-0.016, 0.502),
        (-0.009, 1.001),
        (-0.012, 1.578),
        (0.010, 1.987),
        (-0.008, 2.506),
        (-0.008, 3.000),
        (0.499, 3.009),
        (0.988, 3.030),
        (1.483, 3.030),
        (1.970, 3.030),
        (2.439, 3.063),
        (2.960, 3.063),
        (3.443, 3.063),
        (3.999, 3.047),
        (4.009, 2.487),
        (4.005, 2.006),
        (3.980, 1.541),
        (4.017, 1.063),
        (4.002, 0.631),
        (4.021, -0.000),
        (3.519, 0.043),
        (2.986, 0.029),
        (2.480, 0.032),
        (2.024, 0.035),
        (1.499, 0.014),
        (1.021, 0.027),
        (0.550, 0.005),
        (0.008, -0.009),
    ]

    obstacle_points = [[
        (-0.485, 2.830),
        (-0.328, 1.746),
        (-0.018, 1.414),
        (0.431, 1.633),
        (0.431, 2.190),
        (0.354, 2.758),
        (0.360, 2.999),
        (0.693, 2.546),
        (0.988, 2.282),
        (1.324, 1.896),
        (2.566, 2.146),
        (2.926, 3.046),
        (2.926, 3.046),
        (3.527, 3.645),
        (3.739, 3.011),
        (3.388, 1.975),
        (3.992, 2.127),
        (4.017, 2.137),
        (4.017, 2.137),
        (2.761, 0.663),
        (3.767, 0.178),
        (3.767, 0.037),
        (3.767, 0.006),
        (3.767, 0.006),
        (1.470, -0.507),
        (1.673, 0.066),
        (1.673, 0.066),
        (1.673, 0.066),
        (1.673, 0.066)
    ], 
    [
        (3.0, 3.0),
        (3.0, 3.0)
    ]
    ]


    ### Zeroed arrays used for calcuation
    Tx = int(distance_of_path(np.array(points)) / (params.max_v*0.2941176*params.dt))*2
    ## Generation of waypoints for obstacle and robot
    traj = generate_trajectory_from_waypoints(points, int(Tx/2)+1) # trajectory of waypoints
    

    obstacle_traj = np.zeros((len(params.obstacles), int(Tx/2)+1, 2)) # Generate trajectory for each obstacle
    for i in range(len(params.obstacles)):
        obstacle_traj[i] = generate_trajectory_x_y(obstacle_points[i], int(Tx/2)+1)
    

    x = np.array([traj[0,0],traj[0,1],traj[0,2],0,0])  # Initial state [x, theta, x_dot, theta_dot] -- tracks current state
    X = np.zeros((Tx, 5)) # list of historical states
    U = np.zeros((Tx, 2)) # list of historical control inputs
    all_weights = np.zeros((Tx, params.K)) # Weights of every generated trajectory, organized by time step
    costs = np.zeros(Tx)
    sample_trajectories = np.zeros((Tx, params.K, 3, params.T))
    sample_trajectories_one = np.zeros((params.K, 3, params.T)) # k sets of (x1, x2, ..., xn), (y1, y2, ..., yn), (w1, w2, ..., wn)
    last_u = np.zeros(2) # the control input from the previous 
    
    x_ob = np.zeros((len(params.obstacles),Tx))
    y_ob = np.zeros((len(params.obstacles),Tx))
    ## Zeroed arrays used for calculation
    


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
            for i in range(len(params.obstacles)):
                params.obstacles[i] = np.array([obstacle_traj[i, int(t/2), 0], obstacle_traj[i,int(t/2),1], params.obstacles[i,2]])
            
            base = np.array([np.ones(params.T) * X[t, 0], np.ones(params.T) * X[t, 1], np.zeros(params.T)])
            sample_trajectories[t] = np.repeat(base[np.newaxis, :, :], params.K, axis=0)
            all_weights[t] = np.ones(params.K)
            costs[t] = cost_function(x, U[t], traj[int(t/2)+1])

        for i in range(len(params.obstacles)): # Populate obstacle positions with time
            x_ob[i][t] = params.obstacles[i][0]
            y_ob[i][t] = params.obstacles[i][1]
        
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

