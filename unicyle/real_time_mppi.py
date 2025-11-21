import sys
import os

# Get the absolute path to the directory containing the module
module_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'mpac_go2', 'atnmy'))

# Add the directory to sys.path
sys.path.insert(0, module_dir) 
print(sys.path)


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
    dt = 0.02, # time step for MPPI
    safety_dt = 0.005, # time step for safety
    K = 2000,   # number of samples
    T = 50, # time steps (HORIZON)
    sigma = 2,
    lambda_ = 2,
    l = 0.3,
    r = 0.1, 
    max_v = 0.35, # max x velocity (m/s)
    max_w = 0.5, # max angular velocity (radians/s)
    max_v_dot = 0.1, # max linear acceleration (m/s^2)
    max_w_dot = 0.6, # max angular acceleration (radians/s^2) (8.0)
    obstacles = np.array([(6.0, 0.0, 0.2), (4.0, 0.0, 0.4)]),
    last_obstacle_pos = np.array([[6.0, 0.0], [4.0, 0.0]]),
    first_filter = True
)

# Main function
def main():

    points = np.array([
        (0.0, 0.0, 0.0),
        (3.0, 0.0, 3.0),
        (3.0, 3.0, 7.0),
        (0.0, 3.0, 11.0),
        (0.0, 0.0, 14),
    ])


    obstacle_points = np.array([ 
        np.array([(1.0, 1.0, 0.0), (0, 0.0, 2.0)]),
        np.array([(1.7, 1.7, 0.0), (0.7, 0.7, 2.0)])
    ])


    main_safety_ratio = int(params.dt / params.safety_dt)
    dist = distance_of_path(np.array(points)) # Total travelled distance of trajectory

    vehicle_traj = Trajectory(points)
    # obstacle_trajs = np.array([Trajectory(obstacle_points[0, 0]), Trajectory([obstacle_points[1,0]])])

    traj_time = 14
    
    # Safety Filter Creation
    sf1 = SafetyFilter(params, 3.0, np.diag([200, 1]), params.safety_dt)
    sf2 = SafetyFilter(params,8.0, np.diag([50, 1]), params.safety_dt)
    sf3 = SafetyFilter(params, 8.0, np.diag([45, 1]), params.safety_dt, output=True)
    sf_rollout = SafetyFilter(params, 3.5, np.diag([30, 1]), params.dt)

    # Optimization printout
    print("Is DPP? ", sf1.prob.is_dcp(dpp=True))

    # filename = './../../unicycle/Jfetkovich-HIER/unicyle/data/command_vs_mpac_outputreel.csv'
    # with open(filename, 'w', newline='', encoding='utf-8') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(['x_goal', 'x', 'y_goal', 'y', 'u_v', 'v', 'u_w', 'w'])

    def controller():
        x = np.array([vehicle_traj.sample_trajectory(0)[0],vehicle_traj.sample_trajectory(0)[1],vehicle_traj.sample_trajectory(0)[2],0,0])  # Initial state [x, theta, x_dot, theta_dot] -- tracks current state
        last_u = np.zeros(2) # the control input from the previous 

        x_ob = np.zeros(len(params.obstacles), dtype=np.float32)
        y_ob = np.zeros(len(params.obstacles), dtype=np.float32)
        ## Zeroed arrays used for calculation
        print("Main safety ratio", main_safety_ratio)

        safe_outputs = np.zeros((3, 2), dtype=np.float32)
        stand_idqp()
        clk.sleep(2)
        
        traj_time_start = clk.perf_counter()
        # Main Loop
        while (clk.perf_counter() - traj_time_start) < traj_time: # Continue while trajectory time is not complete
            start_time = clk.perf_counter()
            time = traj_time_start - start_time
            tel = get_tlm_data()

            u_nom, X_calc, traj_weight_single, optimizations = mppi(x, safe_outputs, vehicle_traj, time, params) # Calculate the optimal control input

            # for i in range(len(params.obstacles)):
            #     params.obstacles[i] = np.array([obstacle_trajs[0].sample(time), obstacle_trajs[1].sample(time), params.obstacles[i,2]])

            
            # safe_outputs[0] = sf1.filter(u_nom, x, params, last_u)
            # safe_outputs[1] = sf2.filter(u_nom, x, params, last_u)
            # safe_outputs[2] = sf3.filter(u_nom, x, params, last_u)
            # U[t] = safe_outputs[0]
            walk_idqp(vx=u_nom[0],vy=0,vrz=u_nom[1])
            print(f"Pos: ({tel['q'][0]},{tel['q'][1]})")
            print(f"Command: ({u_nom[0]},{u_nom[1]})")
            print(f"Output: ({tel['qd'][0]},{tel['qd'][5]})")

            
            x = np.array([tel["q"][0], tel["q"][1], tel["q"][5], tel["qd"][0], tel["qd"][5]])
            end_time = clk.perf_counter()
            print(f"loop time: {end_time-start_time}")
            
    controller()

if __name__ == "__main__":
    main()

