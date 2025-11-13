import numpy as np
from scipy.interpolate import interp1d
from numba import njit
import csv
from utils import *
from trajectory import *
from mppi import *
from parameters import *
from animation import animate
from safety_filter import SafetyFilter
import matplotlib.pyplot as plt
import cvxpy as cp
import time as clk
from mpac_cmd import *

params = Parameters(
    dt = 0.025, # time step for MPPI
    safety_dt = 0.001, # time step for safety
    K = 2000,   # number of samples
    T = 18, # time steps (HORIZON)
    sigma = 2,
    lambda_ = 2,
    l = 0.3,
    r = 0.1, 
    max_v = 0.2, # max x velocity (m/s)
    max_w = 1, # max angular velocity (radians/s)
    max_v_dot = 0.1, # max linear acceleration (m/s^2)
    max_w_dot = 3, # max angular acceleration (radians/s^2) (8.0)
    obstacles = np.array([(6.0, 0.0, 0.2), (4.0, 0.0, 0.4)]),
    last_obstacle_pos = np.array([[6.0, 0.0], [4.0, 0.0]]),
    first_filter = True
)

# Main function
def main():
    time = []
    x_pos = []
    y_pos = []

    
   
    # Original (x, y) points
    points = [
        (0.0, 0.0),
        (1.0, 0.0),
        (0.0, 3.0)
    ]

    obstacle_points = [ 
    [
        (1.0, 1.0), 
        (0, 0.0)
    ],
    [
        (1.7, 1.7), 
        (0.7, 0.7)

    ]
    ]

    main_safety_ratio = int(params.dt / params.safety_dt)
    ### Zeroed arrays used for calcuation
    Tx = int(distance_of_path(np.array(points)) / (params.max_v*0.2*params.safety_dt))
    ## Generation of waypoints for obstacle and robot
    traj = generate_trajectory_from_waypoints(points, int(Tx / main_safety_ratio)+1) # trajectory of waypoints
    sf1 = SafetyFilter(params, 3.0, np.diag([200, 1]), params.safety_dt)
    sf2 = SafetyFilter(params,8.0, np.diag([50, 1]), params.safety_dt)
    sf3 = SafetyFilter(params, 8.0, np.diag([45, 1]), params.safety_dt, output=True)
    sf_rollout = SafetyFilter(params, 3.5, np.diag([30, 1]), params.dt)
    print("Is DPP? ", sf1.prob.is_dcp(dpp=True))
    print(f"TX:{Tx}")

    filename = './../../unicycle/Jfetkovich-HIER/unicyle/data/command_vs_mpac_output4.csv'
    with open(filename, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['x_goal', 'x', 'y_goal', 'y', 'u_v', 'v', 'u_w', 'w'])

    def sim():
        obstacle_traj = np.zeros((len(params.obstacles), Tx+1, 2)) # Generate trajectory for each obstacle
        for i in range(len(params.obstacles)):
            obstacle_traj[i] = generate_trajectory_x_y(obstacle_points[i], Tx+1)

        x = np.array([traj[0,0],traj[0,1],traj[0,2],0,0])  # Initial state [x, theta, x_dot, theta_dot] -- tracks current state
        X = np.zeros((Tx, 5)) # list of historical states
        U = np.zeros((Tx, 2)) # list of historical control inputs
        all_weights = np.zeros((Tx, params.K), dtype=np.float32) # Weights of every generated trajectory, organized by time step
        # sample_trajectories = np.zeros((Tx, params.K, 3, params.T), dtype=np.float32)
        # sample_trajectories_one = np.zeros((params.K, 3, params.T), dtype=np.float32) # k sets of (x1, x2, ..., xn), (y1, y2, ..., yn), (w1, w2, ..., wn)
        last_u = np.zeros(2) # the control input from the previous 

        x_ob = np.zeros(len(params.obstacles), dtype=np.float32)
        y_ob = np.zeros(len(params.obstacles), dtype=np.float32)
        ## Zeroed arrays used for calculation
        print("Tx", Tx)
        print("Main safety ratio", main_safety_ratio)
        print("Traj size", traj.size)
        print("Obstacle traj size", obstacle_traj.size)

        safe_outputs = np.zeros((3, 2), dtype=np.float32)
        stand_idqp()
        clk.sleep(4)
        
        for t in range(Tx-1):
            start_time = clk.perf_counter()
            tel = get_tlm_data()

            if t % main_safety_ratio == 0:
                u_nom, X_calc, traj_weight_single, optimizations = mppi(x, safe_outputs, traj[int(t/main_safety_ratio)+1: min(int(t/main_safety_ratio)+1+params.T, len(traj))], params) # Calculate the optimal control input
                # for k in range(params.K):
                #     for t_ in range (params.T): # Reshaping trajectory weight list for use in animation
                #         sample_trajectories_one[k, 0, t_] = X_calc[k, t_, 0] #should be 0
                #         sample_trajectories_one[k, 1, t_] = X_calc[k, t_, 1] #should be 1
                # sample_trajectories[t] = sample_trajectories_one # Save the sampled trajectories
                all_weights[t] = traj_weight_single # List of the weights, populated in mppi function

            else:
                base = np.array([np.ones(params.T) * X[t, 0], np.ones(params.T) * X[t, 1], np.zeros(params.T)])
                # sample_trajectories[t] = np.repeat(base[np.newaxis, :, :], params.K, axis=0)
                all_weights[t] = np.ones(params.K)

            for i in range(len(params.obstacles)):
                params.obstacles[i] = np.array([obstacle_traj[i, t, 0], obstacle_traj[i,t,1], params.obstacles[i,2]])
            for i in range(len(params.obstacles)): # Populate obstacle positions with time
                x_ob[i] = params.obstacles[i][0]
                y_ob[i] = params.obstacles[i][1]
            
            # safe_outputs[0] = sf1.filter(u_nom, x, params, last_u)
            # safe_outputs[1] = sf2.filter(u_nom, x, params, last_u)
            # safe_outputs[2] = sf3.filter(u_nom, x, params, last_u)
            # U[t] = safe_outputs[0]
            U[t] = u_nom
            walk_idqp(vx=u_nom[0],vy=0,vrz=u_nom[1])
            print(f"Goal: ({traj[int(t/main_safety_ratio)+1, 0]},{traj[int(t/main_safety_ratio)+1, 1]})")
            print(f"Pos: ({tel["q"][0]},{tel["q"][1]})")
            print(f"Command: ({u_nom[0]},{u_nom[1]})")
            print(f"Output: ({tel["qd"][0]},{tel["qd"][2]})")

            with open(filename, 'a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow([traj[int(t/main_safety_ratio)+1, 0], tel["q"][0],
                                traj[int(t/main_safety_ratio)+1, 1], tel["q"][1],
                                u_nom[0], tel["qd"][0],
                                u_nom[1], tel["qd"][2]
                ])
            
            x = np.array([tel["q"][0], tel["q"][1], tel["q"][2], tel["qd"][0], tel["qd"][2]])
            X[t + 1, :] = x # Store the new state
            end_time = clk.perf_counter()
            print(f"loop time: {end_time-start_time}")

            
    output_frames = sim()

    


if __name__ == "__main__":
    main()

