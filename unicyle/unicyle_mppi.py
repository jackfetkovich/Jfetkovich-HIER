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
    l = 0.01,
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
        (0.009, -0.022),
        (0.514, 0.035),
        (0.987, 0.436),
        (1.376, 0.743),
        (1.822, 0.855),
        (1.970, 0.855),
        (2.397, 0.812),
        (2.494, 0.714),
        (2.615, 0.422),
        (2.765, 0.178),
        (2.985, 0.007),
        (3.269, 0.017),
        (3.594, 0.008),
        (3.928, 0.008),
        (4.171, 0.052),
        (4.246, 0.117),
        (4.274, 0.207),
        (4.311, 0.307),
        (4.311, 0.460),
        (4.311, 0.570),
        (4.329, 0.843),
        (4.368, 1.124),
        (4.399, 1.544),
        (4.399, 2.492),
        (4.398, 3.271),
        (4.382, 3.508),
        (4.336, 3.607),
        (4.236, 3.623),
        (4.168, 3.623),
        (4.139, 3.633),
        (4.034, 3.674),
        (3.866, 3.696),
        (3.702, 3.696),
        (3.364, 3.690),
        (2.875, 3.711),
        (2.039, 3.752),
        (1.293, 3.752),
        (0.888, 3.795),
        (0.587, 3.795),
        (0.470, 3.766),
        (0.379, 3.724),
        (0.379, 3.649),
        (0.364, 3.577),
        (0.393, 3.554),
        (0.436, 3.491),
        (0.566, 3.399),
        (0.689, 3.316),
        (0.888, 3.217),
        (1.183, 3.032),
        (1.416, 2.899),
        (1.800, 2.732),
        (2.232, 2.560),
        (2.378, 2.511),
        (2.530, 2.385),
        (2.555, 2.304),
        (2.516, 2.262),
        (2.452, 2.225),
        (2.400, 2.225),
        (2.328, 2.205),
        (2.270, 2.205),
        (2.200, 2.184),
        (2.000, 2.184),
        (1.652, 2.184),
        (1.138, 2.184),
        (0.572, 2.205),
        (-0.025, 2.205)
    ]

    obstacle_points = [
        (4.084, 0.480),
        (4.100, 1.058),
        (4.144, 1.583),
        (4.112, 2.311),
        (4.157, 2.625),
        (4.397, 2.835),
        (4.645, 2.536),
        (4.598, 1.920),
        (4.550, 1.211),
        (4.566, 0.626),
        (4.390, 0.555),
        (4.108, 0.951),
        (4.570, 0.994),
        (4.181, 1.309),
        (4.547, 1.325),
        (4.215, 1.533),
        (4.505, 1.554),
        (4.184, 1.877),
        (4.686, 1.920),
        (4.239, 2.091),
        (4.575, 2.135),
        (4.242, 2.476),
        (4.573, 2.508),
        (4.328, 2.253),
        (4.569, 2.196),
        (4.236, 2.107),
        (4.445, 1.977),
        (4.230, 1.897),
        (4.564, 1.779),
        (4.217, 1.695),
        (4.536, 1.594),
        (4.303, 1.489),
        (4.365, 1.259),
        (4.241, 1.218),
        (4.427, 1.135),
        (4.272, 1.113),
        (4.468, 0.969),
        (4.020, 1.052),
        (4.313, 0.766),
        (4.191, 0.848),
        (4.571, 0.785),
        (4.304, 1.016),
        (4.340, 1.016),
        (3.827, 1.331),
        (4.226, 1.202),
        (4.559, 1.181),
        (4.559, 1.181),
        (4.559, 1.181),
        (4.559, 1.181),
        (4.240, 1.522),
        (4.491, 1.353),
        (4.363, 1.805),
        (4.491, 1.419),
        (4.216, 2.287),
        (4.475, 1.679),
        (4.361, 2.655),
        (4.491, 1.875),
        (4.332, 2.581),
        (4.623, 1.756),
        (4.623, 1.756),
        (4.702, 2.161),
        (4.640, 2.182),
        (4.400, 2.123),
        (4.242, 2.123),
        (4.447, 1.765),
        (4.065, 1.851),
        (4.623, 1.595),
        (4.102, 1.554),
        (4.174, 1.527),
        (4.068, 1.527),
        (4.672, 1.273),
        (4.228, 1.338),
        (4.560, 1.148),
        (3.991, 1.361),
        (4.355, 1.042),
        (3.977, 1.317),
        (4.133, 1.001),
        (3.794, 1.254),
        (4.175, 0.871),
        (3.762, 1.022),
        (4.515, 0.848),
        (3.999, 1.347),
        (4.266, 1.383),
        (4.078, 1.593),
        (4.713, 1.785),
        (4.528, 1.868),
        (4.769, 2.006),
        (4.560, 2.028),
        (4.656, 2.115),
        (4.374, 2.137),
        (4.679, 2.348),
        (4.283, 2.307),
        (4.607, 2.578),
        (4.259, 2.515),
        (4.423, 2.891),
        (4.089, 2.743),
        (4.252, 3.110),
        (4.593, 3.644),
        (4.450, 3.200),
        (4.450, 3.200),
        (4.566, 3.278),
        (4.434, 3.183),
        (4.322, 3.074),
        (4.712, 2.777),
        (4.352, 2.878),
        (4.569, 2.256),
        (4.331, 2.449),
        (4.395, 2.090),
        (3.920, 1.974),
        (4.587, 1.293),
        (4.208, 1.429),
        (4.714, 1.322),
        (4.504, 1.587),
        (4.489, 1.501),
        (4.489, 1.501),
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
            params.obstacles[0] = np.array([obstacle_traj[int(t/2), 0], obstacle_traj[int(t/2),1], params.obstacles[0,2]])
            # params.obstacles[0] = np.array([obstacle_traj[int(t/2), 0], 0.05, params.obstacles[0,2]])
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

