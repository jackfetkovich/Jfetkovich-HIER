from numba import njit
from utils import *
import numpy as np
import cvxpy as cp

@njit
def mppi(x, prev_U, targets, params):
    X_calc = np.zeros((params.K, params.T + 1, 5))
    
    U = gen_normal_control_seq(0.3, 6, 0, params.max_w*2, params.K, params.T) # Generate control sequences

    for k in range(params.K):
        X_calc[k, 0, :] = x  # Initialize all trajectories with the current state
            
    costs = np.zeros(params.K) # initialize all costs
    for k in range(params.K):
        for t in range(len(targets)-1):
            X_calc[k, t + 1, :] = unicyle_dynamics(X_calc[k, t, :], U[k, t], params)
            current_target = targets[t]
            cost = cost_function(X_calc[k, t+1, :], U[k, t], current_target)
            costs[k] += cost
        final_target = targets[-1]    
        terminal_cost_val = terminal_cost(X_calc[k, params.T, :], final_target) #Terminal cost of final state
        costs[k] += terminal_cost_val
        
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
    return u_star[0], X_calc, traj_weight_single

# Cost function
@njit
def cost_function(x, u, target):
    Q = np.diag(np.array([16, 16, 1.0, 0.00, 0.00]))  # State costs
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
    Q = np.diag(np.array([20, 20, 1.6, 0.00, 0.00]))
    x_des= np.array([target[0], target[1], target[2], 0, 0])
    state_diff = x_des - x
    state_diff[2] = (state_diff[2] + np.pi) % (2 * np.pi) - np.pi
    terminal_cost = np.dot(state_diff.T, np.dot(Q,state_diff))
    return terminal_cost 

# Unicyle dynamics
@njit
def unicyle_dynamics(x, u, params):    
    v = max(min(u[0], params.max_v), -params.max_v)
    w = max(min(u[1], params.max_w), -params.max_w )

    ## Aceleration Limiting
    last_v = x[3]
    if abs(v - last_v) > params.max_v_dot * params.dt: 
        v = x[3] + params.max_v_dot * params.dt * np.sign(v - x[3])
    
    if abs(w - x[4]) > params.max_w_dot * params.dt:
        w  = x[4] + params.max_w_dot * params.dt * np.sign(w - x[4])

    x_star = np.zeros(5)
    x_star[0] = x[0] + v * np.cos(x[2]) * params.dt
    x_star[1] = x[1] + v * np.sin(x[2]) * params.dt
    x_star[2] = x[2] + w                * params.dt
    x_star[3] = v
    x_star[4] = w

    return x_star

def safety_filter(u_nom, x):
    c = np.array([3.85, 3.8])       # obstacle center
    r = 0.5                        # obstacle radius

    # Variables
    u = cp.Variable(2)        # [v, omega]

    # Compute h and derivatives
    dx = x[0] - c[0]
    dy = x[1] - c[1]
    h = dx**2 + dy**2 - r**2
    Lg_h = np.array([[2*dx*np.cos(x[2]) + 2*dy*np.sin(x[2]), 0]])
    alpha = 1.0
    constraint = Lg_h @ u + alpha * h >= 0
    # Define QP
    cost = cp.sum_squares(u - u_nom)
    prob = cp.Problem(cp.Minimize(cost), [constraint])

    # Solve
    prob.solve(solver=cp.OSQP)

    return u.value