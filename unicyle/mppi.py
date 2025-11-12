from numba import njit
from utils import *
import numpy as np
import cvxpy as cp
import time
import csv
from math import ceil

@njit
def mppi(x, prev_safe, targets, params):
    X_calc = np.zeros((params.K, params.T + 1, 5))
    # U1 = gen_normal_control_seq(prev_safe[0, 0], 6, prev_safe[0, 1], params.max_w/4, int(ceil(params.K/3)), params.T) # Generate control sequences
    # U2 = gen_normal_control_seq(prev_safe[1, 0], 6, prev_safe[1, 1], params.max_w/4, int(ceil(params.K/3)), params.T)
    # U3 = gen_normal_control_seq(prev_safe[2, 0], 6, prev_safe[2, 1], params.max_w/4, params.K - 2*int(ceil(params.K/3)), params.T)
    # U = np.vstack((U1, U2, U3))

    U = gen_normal_control_seq(0.3, 1, 0, params.max_w*3, params.K, params.T) #

    num_optimizations = 0

    for k in range(params.K):
        X_calc[k, 0, :] = x  # Initialize all trajectories with the current state
            
    costs = np.zeros(params.K) # initialize all costs
    last_u = np.zeros(2)
    for k in range(params.K):
        path_safe = True
        for t in range(len(targets)-1):
            u_nom = U[k,t]
            u_safe = u_nom
            X_calc[k, t + 1, :] = unicyle_dynamics(X_calc[k, t, :], u_safe, params)
            next_x = X_calc[k, t+1, :]
            for o in params.obstacles: # check for obstacle collision
                if (next_x[0]-o[0] + params.l*np.cos(next_x[2])) ** 2 + (next_x[1] - o[1] + params.l*np.sin(next_x[2])) ** 2 <= (o[2])**2:
                    # path_safe = False
                    num_optimizations += 1
                    # costs[k]+=np.inf
                    break
                   
            current_target = targets[t]
            cost = cost_function(X_calc[k, t+1, :], u_safe, current_target)
            costs[k] += cost
            last_u = u_safe
        if path_safe:
            final_target = targets[-1]    
            terminal_cost_val = terminal_cost(X_calc[k, params.T, :], final_target) #Terminal cost of final state
            costs[k] += terminal_cost_val
        last_u = np.zeros(2)
        
    # Calculate weights for each trajectory
    weights = np.exp(-(costs - np.min(costs)) / params.lambda_)
    sum_weights = np.sum(weights)
    if sum_weights < 1e-10:
        weights = np.ones_like(weights) / len(weights)  # fallback to uniform
    else:
        weights /= sum_weights
    
    traj_weight_single = np.zeros(params.K)
    traj_weight_single[:] = weights

    # Compute the weighted sum of control inputs
    u_star = np.sum(weights[:, None, None] * U, axis=0)
    return u_star[0], X_calc, traj_weight_single, num_optimizations

# Cost function
@njit
def cost_function(x, u, target):
    Q = np.diag(np.array([16, 16, 0.0, 0.00, 0.00]))  # State costs
    R = np.diag(np.array([0.0005,0.001]))  # Input costs

    x_des = np.array([target[0], target[1], target[2], 0, 0])
    state_diff = x_des - x
    state_diff[2] = (state_diff[2] + np.pi) % (2 * np.pi) - np.pi
    state_cost = np.dot(state_diff.T, np.dot(Q,state_diff))

    cost = state_cost + np.dot(u.T, np.dot(R, u))
    return cost

# Terminal Cost Function
@njit
def terminal_cost(x, target):
    Q = np.diag(np.array([20, 20, 0.0, 0.00, 0.00]))
    x_des= np.array([target[0], target[1], target[2], 0, 0])
    state_diff = x_des - x
    state_diff[2] = (state_diff[2] + np.pi) % (2 * np.pi) - np.pi
    terminal_cost = np.dot(state_diff.T, np.dot(Q,state_diff))
    return terminal_cost 

# Unicyle dynamics
@njit
def unicyle_dynamics(x, u, params, dt=-1.0):    
    if(dt == -1.0):
        dt = params.dt
    
    v = max(min(u[0], params.max_v), -params.max_v)
    w = max(min(u[1], params.max_w), -params.max_w )
    # v = u[0]
    # w = u[1]

    # Aceleration Limiting
    last_v = x[3]
    if abs(v - last_v) > params.max_v_dot * dt: 
        v = x[3] + params.max_v_dot * dt * np.sign(v - x[3])
    
    if abs(w - x[4]) > params.max_w_dot * dt:
        w  = x[4] + params.max_w_dot * dt * np.sign(w - x[4])

    x_star = np.zeros(5)
    x_star[0] = x[0] + v * np.cos(x[2]) * dt
    x_star[1] = x[1] + v * np.sin(x[2]) * dt
    x_star[2] = x[2] + w                * dt
    x_star[3] = v
    x_star[4] = w

    return x_star
