import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.interpolate import interp1d
from scipy.spatial import KDTree
import cvxpy as cp
from numba import njit
import csv
matplotlib.use("TkAgg")

# system parameters
dt = 0.05 # time step
K = 500   # number of samples
T = 30 # time steps (HORIZON)
sigma = 2
lambda_ = 2

# Simulation Parameters
init_x = 0.0
init_y = 0.0
init_theta = 0
init_x_dot = 0.0
init_y_dot = 0.0
init_theta_dot = 0.0

obstacles = np.array([[3.85, 3.8, 0.5]])

max_v = 2.0 # max x velocity (m/s)
max_w = 10.0 # max angular velocity (radians/s)
max_v_dot = 2.0 # max linear acceleration (m/s^2)
max_w_dot = 10.0 # max angular acceleration (radians/s^2) (8.0)

# Unicyle dynamics
@njit
def unicyle_dynamics(x, u):    
    v = max(min(u[0], max_v), -max_v)
    w = max(min(u[1], max_w), -max_w )
    # v = u[0]
    # w = u[1]

    ## Aceleration Limiting
    last_v = x[3]
    if abs(v - last_v) > max_v_dot * dt: 
        v = x[3] + max_v_dot * dt * np.sign(v - x[3])
    
    if abs(w - x[4]) > max_w_dot * dt:
        w  = x[4] + max_w_dot * dt * np.sign(w - x[4])

    x_star = np.zeros(5)
    x_star[0] = x[0] + v * np.cos(x[2]) * dt
    x_star[1] = x[1] + v * np.sin(x[2]) * dt
    x_star[2] = x[2] + w                * dt
    x_star[3] = v
    x_star[4] = w

    return x_star

@njit
def closest_point_on_path(waypoints, point, last_index):
    """
    waypoints: (N,2) array
    point: (2,) array
    Returns: (closest_x, closest_y, distance, seg_idx, t)
    """
    best_d2 = 1e18
    best_point = np.zeros(2)
    best_idx = last_index
    best_t = 0.0

    for i in range(max(best_idx, 0), min(best_idx + 40, waypoints.shape[0]-2)):
        A = waypoints[i]
        B = waypoints[i + 1]
        AB = B - A
        AP = point - A
        denom = AB[0] * AB[0] + AB[1] * AB[1]
        if denom > 1e-12:
            t = (AP[0] * AB[0] + AP[1] * AB[1]) / denom
            if t < 0.0:
                t = 0.0
            elif t > 1.0:
                t = 1.0
        else:
            t = 0.0
        proj = A + t * AB
        dx = point[0] - proj[0]
        dy = point[1] - proj[1]
        d2 = dx * dx + dy * dy
        if d2 < best_d2:
            best_d2 = d2
            best_point = proj
            best_idx = i
            best_t = t

    return best_point[0], best_point[1], best_idx

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

@njit
def gen_normal_control_seq(mu_v, sigma_v, mu_w, sigma_w, K, T):
    return np.dstack((
        np.random.normal(loc=mu_v, scale=sigma_v, size=(K, T)), #v
        np.random.normal(loc=mu_w, scale=sigma_w, size=(K, T)), #omega
    )) 

# @njit 
# def remove_points_colliding_with_obstacle(path, obstacles):
#     num_points_removed = 0
#     filtered_path = np.copy(path)
#     for pt in filtered_path:
#         pt_x = pt[0]
#         pt_y = pt[1]
#         for ob in obstacles:
#             if (pt_x - ob[0])**2 + (pt_y - ob[1])**2 - ob[3]**2 <= 0:
#                 pt[0] = np.inf
#                 pt[1] = np.inf
#                 num_points_removed += 1
                
#     return filtered_path, num_points_removed

@njit
def point_in_obstacle(point, obstacles):
    pt_x = point[0]
    pt_y = point[1]
    for ob in obstacles:
        if (pt_x - ob[0])**2 + (pt_y - ob[1])**2 - ob[3]**2 <= 0:
            return True
    return False
    

# MPPI control
@njit
def mppi(x, prev_U, traj_x_y, traj, starting_traj_idx):
    X_calc = np.zeros((K, T + 1, 5))
    
    U = gen_normal_control_seq(0.3, 0.2, 0, max_w, K, T) # Generate control sequences

    for k in range(K):
        X_calc[k, 0, :] = x  # Initialize all trajectories with the current state
            
    costs = np.zeros(K) # initialize all costs
    for k in range(K):
        target_idx = starting_traj_idx
        for t in range(T-1):
            X_calc[k, t + 1, :] = unicyle_dynamics(X_calc[k, t, :], U[k, t])
            current_target_raw = closest_point_on_path(traj_x_y, X_calc[k, t+1, 0:2], target_idx)
            current_target = np.array([current_target_raw[0], current_target_raw[1], traj[target_idx][2]])
            target_idx = current_target_raw[2]
            cost = cost_function(X_calc[k, t+1, :], U[k, t], current_target)
            costs[k] += cost
        final_target_raw = closest_point_on_path(traj_x_y, X_calc[k, T-1, 0:2], target_idx)
        target_idx = final_target_raw[2]
        final_target = np.array([final_target_raw[0], final_target_raw[1], traj[target_idx][2]])  
        terminal_cost_val = terminal_cost(X_calc[k, T, :], final_target) #Terminal cost of final state
        costs[k] += terminal_cost_val
        
    # Calculate weights for each trajectory
    weights = np.exp(-(costs - np.min(costs)) / lambda_)
    sum_weights = np.sum(weights)
    if sum_weights < 1e-10:
        weights = np.ones_like(weights) / len(weights)  # fallback to uniform
    else:
        weights /= sum_weights
    
    traj_weight_single = np.zeros(K)
    traj_weight_single[:] = weights

    # Compute the weighted sum of control inputs
    u_star = np.sum(weights[:, None, None] * U, axis=0)
    return u_star[0], X_calc, traj_weight_single

def generate_trajectory_from_waypoints(waypoints, num_points=100):
    """
    Generates a smooth trajectory by interpolating between given waypoints.

    Parameters:
    waypoints (list of tuples): List of (x, y, theta) waypoints.
    num_points (int): Number of points to interpolate.

    Returns:
    tuple: (x_vals, y_vals, theta_vals) interpolated trajectory.
    """
    waypoints = np.array(waypoints)
    t = np.linspace(0, 1, len(waypoints))  # Normalized parameter along waypoints
    t_interp = np.linspace(0, 1, num_points)  # Fine-grained interpolation parameter

    # Interpolating x, y, and theta
    interp_x = interp1d(t, waypoints[:, 0], kind='linear')
    interp_y = interp1d(t, waypoints[:, 1], kind='linear')
    interp_theta = interp1d(t, waypoints[:, 2], kind='linear')

    x_vals = interp_x(t_interp)
    y_vals = interp_y(t_interp)
    theta_vals = interp_theta(t_interp)
    # np.savetxt("output.txt", np.array(theta_vals), fmt="%.6f")  # 6 decimal places

    return np.dstack(np.array([x_vals, y_vals, np.unwrap(np.array(theta_vals))]))[0]

def animate(x_vals, y_vals, x_traj, y_traj, sample_trajs, weights, goal_points_x, goal_points_y):
    """
    Animates the movement of an object in 2D space given its state variables over time.
    Also plots a given trajectory as a dotted line.

    Parameters:
    x_vals (list or np.array): X positions over time.
    y_vals (list or np.array): Y positions over time.
    theta_vals (list or np.array): Orientations (in radians) over time.
    x_traj (list or np.array, optional): X values of the reference trajectory.
    y_traj (list or np.array, optional): Y values of the reference trajectory.
    """
    # Set up the figure
    # fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=100)
    fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=100)
    ax.set_xlim(min(x_vals) - 1, max(x_vals) + 1)
    ax.set_ylim(min(y_vals) - 1, max(y_vals) + 1)
    # ax.set_xlim(-3, 3)
    # ax.set_ylim(-3, 3)
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")

    # plt.plot(3.85, 3.8, 'yo') #obstacle
    # circle1 = plt.Circle((3.85, 3.8), 0.5, color='r')
    # ax.add_patch(circle1)

    # Plot the trajectory if provided
    if x_traj is not None and y_traj is not None:
        ax.plot(x_traj, y_traj, 'k--', linewidth=1.5, label="Trajectory")  # Dotted reference path

    samples = []
    for i in range(K):
        samples.append(ax.plot([], [], color=[0.5, 0.5, 0.5], linewidth=0.5)[0])

    # Initialize plot elements
    line, = ax.plot([], [], 'r-', linewidth=2)  # History line
    point, = ax.plot([], [], 'bo', markersize=8)  # Current position
    ghost,  = ax.plot([], [], 'gx', markersize=6)  # Desired position


    # Update function
    def update(frame):
        x, y = x_vals[frame], y_vals[frame]

        # Update history path
        line.set_data(x_vals[:frame + 1], y_vals[:frame + 1])
                # Update point position
        point.set_data([x], [y])

        max_intensity = -1
        for i in range(K):
            this_intensity = weights[frame][i]
            if this_intensity > max_intensity:
                max_intensity = this_intensity
        # Update generated trajectories
        for i in range(K):
            intensity = min(weights[frame][i], 1)
            samples[i].set_data([], [])  # Clears previous data
            samples[i].set_color([0, intensity/max_intensity , 0, intensity/max_intensity])
            samples[i].set_data(sample_trajs[frame, i, 0, 0 : T], sample_trajs[frame, i, 1, 0 : T])

        # Update ghost point if reference trajectory exists
        if x_traj is not None and y_traj is not None:
            ghost.set_data([goal_points_x[frame]], [goal_points_y[frame]])


        return [line, point, ghost].append(samples)

    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=len(x_vals), interval=15, blit=False)
    plt.title(f"K={K}, T={T} Position Based")
    plt.legend()
    filename=f"unicyle{K}-{T}-position_based.gif"
    ani.save(filename, writer='pillow', fps=20, )
    print(f"Animation saved as {filename}")
    plt.show()

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

def distance_of_path(p):
    distance = 0
    for x in range(len(p)-1):
        distance += np.sqrt((p[x+1,0] - p[x, 0])**2 + (p[x+1,1] - p[x, 1])**2)
    return distance



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
    new_thetas = np.hstack([
        np.zeros(14),
        np.pi/4 * np.ones(13),
        np.pi/2 * np.ones(13),
        3*np.pi /4 * np.ones(13),
        np.pi * np.ones(13),
        5*np.pi /4 * np.ones(13),
        3 * np.pi / 2 *np.ones(13),
        7*np.pi /4 * np.ones(14)
    ])
    Tx = 1000
    x = np.array([0,0,0, 0, 0])  # Initial state [x, theta, x_dot, theta_dot] -- tracks current state
    X = np.zeros((Tx, 5)) # list of historical states
    U = np.zeros((Tx, 2)) # list of historical control inputs
    all_weights = np.zeros((Tx, K)) # Weights of every generated trajectory, organized by time step

    traj = generate_trajectory_from_waypoints(waypoints, Tx) # trajectory of waypoints
    traj_x_y = traj[:, :2]

    sample_trajectories = np.zeros((Tx, K, 3, T))
    sample_trajectories_one = np.zeros((K, 3, T)) # k sets of (x1, x2, ..., xn), (y1, y2, ..., yn), (w1, w2, ..., wn)
    last_u = np.zeros(2) # the control input from the previous iteration
    best_traj_idx = 0
    trajectory_indices = np.zeros(Tx)
    goal_points = np.zeros((2, Tx))
    costs = np.zeros(Tx)
    for t in range(Tx-1): # From 0 -> 159
        trajectory_indices[t] = best_traj_idx
        u_nom, X_calc, traj_weight_single = mppi(x, last_u, traj_x_y, traj, best_traj_idx) # Calculate the optimal control input
        # U[t] = safety_filter(u_nom, x)
        U[t] = u_nom
        x = unicyle_dynamics(x, U[t]) # Calculate what happens when you apply that input
        X[t + 1, :] = x # Store the new state
        best_traj_idx = closest_point_on_path(traj_x_y, x[:2], best_traj_idx)[2]
        print(best_traj_idx)
        goal_points[0,t] = traj_x_y[best_traj_idx, 0]
        goal_points[1,t] = traj_x_y[best_traj_idx, 1]
        time.append(t)
        x_pos.append(X[t+1, 0]) # Save the x position at this timestep
        y_pos.append(X[t+1, 1]) # Save the y position at this timestep
        costs[t] = cost_function(x, U[t], traj[best_traj_idx])
        last_u = U[t] # Save the control input 


        for k in range(K):
            for t_ in range (T): # Reshaping trajectory weight list for use in animation
                sample_trajectories_one[k, 0, t_] = X_calc[k, t_, 0] #should be 0
                sample_trajectories_one[k, 1, t_] = X_calc[k, t_, 1] #should be 1
        sample_trajectories[t] = sample_trajectories_one # Save the sampled trajectories
        all_weights[t] = traj_weight_single # List of the weights, populated in mppi function
        
        # with open("data3.txt", "a") as f: 
        #     f.write(f"t: {t},  X=(x={x[0]}, y={x[1]}, th={x[2]}), U=(v={u_nom[0]}, w={u_nom[1]})\n")
        #     f.write(f"Close pt x={traj[best_traj_idx][0]}, y={traj[best_traj_idx][1]}, th={traj[best_traj_idx][2]}\n")
        #     f.write(f"idx{best_traj_idx}\n")
        #     f.write("-------------------\n")
    
    with open('circle_discrepancy_pos.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Step',  'X_d', 'Y_d', 'Theta_d', 'X', 'Y', 'Theta', 'Cost'])
        for t in range(Tx):
            idx = int(trajectory_indices[t])
            writer.writerow(np.array([t,traj[idx][0], traj[idx][1], traj[idx][2],X[t][0], X[t][1], X[t][2], costs[t]]))


    animate(x_pos, y_pos, traj[:, 0], traj[:, 1], sample_trajectories, all_weights, goal_points[0, :], goal_points[1, :])


if __name__ == "__main__":
    main()

