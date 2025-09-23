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
        path_safe = True
        for t in range(len(targets)-1):
            X_calc[k, t + 1, :] = unicyle_dynamics(X_calc[k, t, :], U[k, t], params)
            next_x = X_calc[k, t+1, :]
            for o in params.obstacles: # check for obstacle collision
                if (next_x[0]-o[0] + params.l*np.cos(next_x[2])) ** 2 + (next_x[1] - o[1] + params.l*np.sin(next_x[2])) ** 2 <= (o[2]+params.r)**2:
                    path_safe = False
                    costs[k] = np.inf
                    break
            current_target = targets[t]
            cost = cost_function(X_calc[k, t+1, :], U[k, t], current_target)
            costs[k] += cost
        if path_safe:
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
    v = u[0]
    w = u[1]

    ## Aceleration Limiting
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

def safety_filter(u_nom, x, params, last_u):
    # Variables
    u = cp.Variable(2)        # [v, omega]
    alpha = 5.0

    constraints = []

    # Loop through all obstacles
    for i in range(len(params.obstacles)):
        c = params.obstacles[i][0:2]   # obstacle center (x, y)
        r = params.obstacles[i][2]     # obstacle radius

        dx = x[0] - c[0] + params.l * np.cos(x[2])
        dy = x[1] - c[1] + params.l * np.sin(x[2])

        if params.first_filter:
            vx_obs = 0.0
            vy_obs = 0.0
        else:
            vx_obs = (c[0] - params.last_obstacle_pos[i][0])*params.safety_dt
            vy_obs = (c[1] - params.last_obstacle_pos[i][1])*params.safety_dt
        params.last_obstacle_pos[i] = np.array([c[0], c[1]])

        print(vx_obs)
        v_obs = np.array([vx_obs, vy_obs])

        # Barrier function
        h = (dx + params.l * np.cos(x[2]))**2 + (dy + params.l * np.sin(x[2]))**2 - r**2
        # Lie derivative term
        Lg_h = np.array([
            2*dx*np.cos(x[2]) + 2*dy*np.sin(x[2]),
            -2*dx*params.l*np.sin(x[2]) + 2*dy*params.l*np.cos(x[2])
        ])

        dh_dt = -2*(x[0]-c[0])*vx_obs - 2*(x[1] - c[1])*vy_obs

        # Add inequality constraint: Lg_h @ u + alpha * h >= 0
        constraints.append(Lg_h @ u + dh_dt + alpha * h >= 0)

    constraints.append(u[0] <= params.max_v)
    constraints.append(u[0] >= -params.max_v)
    constraints.append(u[1] <= params.max_w)
    constraints.append(u[1] >= -params.max_w)
    constraints.append(u[0] - last_u[0] <= params.max_v_dot)
    constraints.append(u[0] - last_u[0] >= -params.max_v_dot)
    constraints.append(u[1] - last_u[1] <= params.max_w_dot)
    constraints.append(u[1] - last_u[1] >= -params.max_w_dot)

    
    if np.isnan(u_nom[0]) or np.isnan(u_nom[1]):
        print("NAN")
        print("x", x)
        print("Ob 1 pos:", params.obstacles[0, :])
        # print("Ob 2 pos:", params.obstacles[1, :])


    # Define QP
    cost = cp.sum_squares(u - u_nom)
    # Q = np.diag([5.0, 1.0])  # weight on v is 1.0, weight on w is alpha
    # cost = cp.quad_form(u - u_nom, Q)
    prob = cp.Problem(cp.Minimize(cost), constraints)
    try:
        prob.solve(solver=cp.OSQP, warm_start=True)
        if prob.status not in ["optimal", "optimal_inaccurate"]:
            raise cp.error.SolverError("Infeasible or failed solve")
        u = u.value
    except cp.error.SolverError:
    # Fallback strategy
        u = np.array([0, 0])
    
    print("------")
    # print("h1", h1)
    # print("constraint", Lg_h1 @ u + alpha * h1)
    # print("gtz", Lg_h1 @ u + alpha * h1 >= 0)
    print("------")
    params.first_filter = False
    # Solve
    return u