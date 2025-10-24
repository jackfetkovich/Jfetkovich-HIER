from numba import njit
from utils import *
import numpy as np
import cvxpy as cp
import time
import csv

@njit
def mppi(x, prev_safe, targets, params):
    
    X_calc = np.zeros((params.K, params.T + 1, 5))
    U1 = gen_normal_control_seq(prev_safe[0, 0], 6, prev_safe[0, 1], params.max_w*2, 667, params.T) # Generate control sequences
    U2 = gen_normal_control_seq(prev_safe[1, 0], 6, prev_safe[1, 1], params.max_w*2, 667, params.T)
    U3 = gen_normal_control_seq(prev_safe[2, 0], 6, prev_safe[2, 1], params.max_w*2, 666, params.T)
    U = np.vstack((U1, U2, U3))

    # U = gen_normal_control_seq(0.3, 6, 0, params.max_w*2, params.K, params.T) #

    num_discarded_paths = 0

    for k in range(params.K):
        X_calc[k, 0, :] = x  # Initialize all trajectories with the current state
            
    costs = np.zeros(params.K) # initialize all costs
    for k in range(params.K):
        path_safe = True
        for t in range(len(targets)-1):
            X_calc[k, t + 1, :] = unicyle_dynamics(X_calc[k, t, :], U[k, t], params)
            next_x = X_calc[k, t+1, :]
            # for o in params.obstacles: # check for obstacle collision
                # if (next_x[0]-o[0] + params.l*np.cos(next_x[2])) ** 2 + (next_x[1] - o[1] + params.l*np.sin(next_x[2])) ** 2 <= (o[2])**2:
                #     path_safe = False
                #     num_discarded_paths += 1
                #     costs[k] = np.inf
                #     break
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
    return u_star[0], X_calc, traj_weight_single, num_discarded_paths

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


# -----------------------------
# Config / scratch (as before)
# -----------------------------
num_obstacles = 2
filter_outputs = np.zeros((3, 2), dtype=np.float32)

# -----------------------------
# Metric projection onto one half-space:
#   minimize (u - u_nom)^T Q (u - u_nom)  s.t.  a^T u >= b
# Closed-form:
#   u* = u + max(0, (b - a^T u) / (a^T Q^{-1} a)) * (Q^{-1} a)
# -----------------------------
def project_halfspace(u, a, b, Q_inv):
    # violation = b - a^T u
    viol = b - float(a @ u)
    denom = float(a @ (Q_inv @ a))
    if viol > 0 and denom > 1e-12:
        u = u + (viol / denom) * (Q_inv @ a)
    return u

# -----------------------------
# One pass of sequential projections over many constraints
# (optionally sorted by most violated first)
# -----------------------------
def project_many(u, A, b, Q_inv, passes=2):
    for _ in range(passes):
        # sort by violation magnitude (most violated first)
        viol = b - (A @ u)
        order = np.argsort(-viol)  # descending
        for i in order:
            if viol[i] > 0:
                u = project_halfspace(u, A[i], b[i], Q_inv)
    return u

# -----------------------------
# Build obstacle CBF constraints a_i^T u >= b_i
# Using your Lg_h, dh_dt, and alpha * h terms (time-varying obstacles supported)
# -----------------------------
def build_obstacle_constraints(x, params, alpha):
    A_list = []
    b_list = []

    # Robot pose
    px, py, th = x[0], x[1], x[2]
    cth, sth = np.cos(th), np.sin(th)

    for i in range(num_obstacles):
        c = params.obstacles[i][0:2]   # (x_i, y_i)
        r = params.obstacles[i][2]     # radius

        # Look-ahead point on the body
        dx = px - c[0] + params.l * cth
        dy = py - c[1] + params.l * sth

        # Obstacle velocity estimate
        if params.first_filter:
            vx_obs = 0.0
            vy_obs = 0.0
        else:
            vx_obs = (c[0] - params.last_obstacle_pos[i][0]) / params.safety_dt
            vy_obs = (c[1] - params.last_obstacle_pos[i][1]) / params.safety_dt
        params.last_obstacle_pos[i] = np.array([c[0], c[1]])

        # Barrier function
        h = dx*dx + dy*dy - (r + 0.1)**2

        # Lg_h (matches your code; depends on v and omega)
        Lg_h = np.array([
            2.0*dx*cth + 2.0*dy*sth,
            -2.0*dx*params.l*sth + 2.0*dy*params.l*cth
        ], dtype=float)

        # Time-varying obstacle term ∂h/∂t
        dh_dt = -2.0*dx*vx_obs - 2.0*dy*vy_obs

        # Inequality: Lg_h @ u + dh_dt + alpha*h >= 0  ⇒  a^T u ≥ b
        a = Lg_h
        b = -dh_dt - alpha*h

        A_list.append(a)
        b_list.append(b)

    if len(A_list) == 0:
        return np.zeros((0, 2), dtype=float), np.zeros((0,), dtype=float)

    return np.vstack(A_list), np.array(b_list, dtype=float)

# -----------------------------
# Add box and slew-rate constraints as half-spaces a^T u >= b
# v ≤ vmax      ⇒ (-1, 0)·u ≥ -vmax
# v ≥ vmin      ⇒ ( 1, 0)·u ≥  vmin
# ω ≤ ωmax      ⇒ ( 0,-1)·u ≥ -ωmax
# ω ≥ ωmin      ⇒ ( 0, 1)·u ≥  ωmin
# v ≤ last_v + Δv  ⇒ (-1,0)·u ≥ -(last_v + Δv)
# v ≥ last_v - Δv  ⇒ ( 1,0)·u ≥  (last_v - Δv)
# similarly for ω
# -----------------------------
def build_box_and_rate_constraints(params, last_u):
    A = []
    b = []

    # Box limits
    vmax = params.max_v
    vmin = -params.max_v
    wmax = params.max_w
    wmin = -params.max_w

    A += [np.array([-1.0,  0.0]), np.array([ 1.0, 0.0]),
          np.array([ 0.0, -1.0]), np.array([ 0.0, 1.0])]
    b += [-vmax, vmin, -wmax, wmin]

    # Rate limits
    dv = params.max_v_dot * params.safety_dt
    dw = params.max_w_dot * params.safety_dt

    # v ≤ last_v + dv  ⇒ (-1,0)·u ≥ -(last_v + dv)
    A.append(np.array([-1.0, 0.0])); b.append(-(last_u[0] + dv))
    # v ≥ last_v - dv  ⇒ ( 1,0)·u ≥  (last_u - dv)
    A.append(np.array([ 1.0, 0.0])); b.append( (last_u[0] - dv))

    # w ≤ last_w + dw  ⇒ (0,-1)·u ≥ -(last_w + dw)
    A.append(np.array([0.0, -1.0])); b.append(-(last_u[1] + dw))
    # w ≥ last_w - dw  ⇒ (0, 1)·u ≥  (last_w - dw)
    A.append(np.array([0.0,  1.0])); b.append( (last_u[1] - dw))

    return np.vstack(A), np.array(b, dtype=float)

# -----------------------------
# The safety filter (drop-in replacement)
# Returns filter_outputs[3,2] like your original, with three Q metrics
# -----------------------------
def safety_filter(u_in, x, params, last_u):
    alpha = 15.0
    # For parity with your prints:
    print(u_in)

    # Build obstacle constraints
    A_obs, b_obs = build_obstacle_constraints(x, params, alpha)

    # Build box + rate constraints
    A_lim, b_lim = build_box_and_rate_constraints(params, last_u)

    # Stack all constraints
    if A_obs.shape[0] > 0:
        A_all = np.vstack([A_obs, A_lim])
        b_all = np.concatenate([b_obs, b_lim])
    else:
        A_all = A_lim
        b_all = b_lim

    # Handle NaN nominal control (diagnostic parity with your code)
    if np.isnan(u_in[0]) or np.isnan(u_in[1]):
        print("NAN")
        print("x", x)
        print("Ob 1 pos:", params.obstacles[0, :])

    # Three different Q metrics (matching your varying v-weight idea)
    Q_list = [
        np.diag([40.0*((1)/3), 1.0]),
        np.diag([40.0*((2)/3), 1.0]),
        np.diag([40.0*((3)/3), 1.0]),
    ]

    out = np.zeros_like(filter_outputs)
    for j, Q in enumerate(Q_list):
        try:
            start_time = time.perf_counter()

            # Metric projection loop
            Q_inv = np.linalg.inv(Q)
            u = np.array(u_in, dtype=float)

            # Sequential projections (2 passes is usually enough)
            u = project_many(u, A_all, b_all, Q_inv, passes=2)

            # (Optional) final clamp to hard bounds for extra robustness
            u[0] = np.clip(u[0], -params.max_v, params.max_v)
            u[1] = np.clip(u[1], -params.max_w, params.max_w)

            end_time = time.perf_counter()
            print(f"solve time (projection): {end_time - start_time:.6f}s")

            out[j] = u
        except Exception as e:
            # Fallback
            print("Projection error:", e)
            out[j] = np.array([0.0, 0.0], dtype=float)

    params.first_filter = False
    return out
