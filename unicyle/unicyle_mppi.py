import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.interpolate import interp1d
from scipy.spatial import KDTree
import cvxpy as cp
from numba import njit
matplotlib.use("TkAgg")

# system parameters
dt = 0.05 # time step
K = 500   # number of samples
T = 20 # time steps (HORIZON)
sigma = 2
lambda_ = 2

# Simulation Parameters
init_x = 0.0
init_y = 0.0
init_theta = 0
init_x_dot = 0.0
init_y_dot = 0.0
init_theta_dot = 0.0

obstacle = np.array([0.3, 0.5])

max_v = 2 # max x velocity
max_w = 10 # max angular velocity
max_v_dot = 2 # max linear acceleration
max_w_dot = 10 # max angular acceleration

# Unicyle dynamics
@njit
def unicyle_dynamics(x, u):    
    # v = np.clip(u[0], -max_v, max_v)
    v = max(min(u[0], max_v), -max_v)
    # w  = np.clip(u[1], -max_w, max_w)
    w = max(min(u[1], max_w), -max_w )

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

    x_star[2] = (x_star[2] + np.pi) % (2 * np.pi) - np.pi

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

    for i in range(max(best_idx, 0), min(best_idx + 60, waypoints.shape[0]-2)):
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
    Q = np.diag(np.array([8, 8, 1.2, 0.00, 0.00]))  # State costs
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
    Q = np.diag(np.array([20, 20, 3.3, 0.00, 0.00]))
    x_des= np.array([target[0], target[1], target[2], 0, 0])
    state_diff = x_des - x
    state_diff[2] = (state_diff[2] + np.pi) % (2 * np.pi) - np.pi
    terminal_cost = np.dot(state_diff.T, np.dot(Q,state_diff))
    return terminal_cost 


# MPPI control
@njit
def mppi(x, prev_U, traj, starting_traj_idx):
    X_calc = np.zeros((K, T + 1, 5))
    
    U = np.dstack((
        np.random.normal(loc=0.3, scale=0.2, size=(K, T)), #v
        np.random.normal(loc=0, scale=6, size=(K, T)), #omega
    )) # Generate random (normal) control inputs

    for k in range(K):
        X_calc[k, 0, :] = x  # Initialize all trajectories with the current state

    costs = np.zeros(K) # initialize all costs
    for k in range(K):
        target_idx = starting_traj_idx
        for t in range(T-1):
            X_calc[k, t + 1, :] = unicyle_dynamics(X_calc[k, t, :], U[k, t])
            current_target_raw = closest_point_on_path(traj, X_calc[k, t+1, 0:2], target_idx) 
            target_idx = current_target_raw[2]    
            current_target = np.array([current_target_raw[0], current_target_raw[1], traj[target_idx][2]])
            costs[k] += cost_function(X_calc[k, t+1, :], U[k, t], current_target)
        final_target_raw = closest_point_on_path(traj, X_calc[k, T-1, 0:2], target_idx)
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

    return np.dstack(np.array([x_vals, y_vals, theta_vals]))[0]


def animate(x_vals, y_vals, x_traj, y_traj, sample_trajs, weights):
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
    fig, ax = plt.subplots()
    ax.set_xlim(min(x_vals) - 1, max(x_vals) + 1)
    ax.set_ylim(min(y_vals) - 1, max(y_vals) + 1)
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")

    # plt.plot(3.85, 3.8, 'yo') #obstacle
    circle1 = plt.Circle((3.85, 3.8), 0.5, color='r')
    ax.add_patch(circle1)

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
            ghost.set_data([x_traj[frame]], [y_traj[frame]])


        return [line, point, ghost].append(samples)

    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=len(x_vals), interval=50, blit=False)
    plt.title(f"K={K}, T={T}")
    plt.legend()
    # filename=f"unicyle{K}-{T}-green.gif"
    # ani.save(filename, writer='pillow', fps=20)
    # print(f"Animation saved as {filename}")
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


# Main function
def main():
    # Tx = 1000
    Tx=1000
    x = np.array([0,0,0, 0, 0])  # Initial state [x, theta, x_dot, theta_dot] -- tracks current state
    X = np.zeros((Tx, 5)) # list of historical states
    U = np.zeros((Tx, 2)) # list of historical control inputs
    all_weights = np.zeros((Tx+1, K)) # Weights of every generated trajectory, organized by time step
    
    time = []
    x_pos = []
    y_pos = []
    waypoints = [
        (0.0, 0.0, 0.0),
        (0.5, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.5, 0.0, 0.0),
        (2.0, 0.1, 0.19739555984988078),
        (2.5, 0.4, 0.3805063771123649),
        (3.0, 0.9, 0.5880026035475675),
        (3.4, 1.4, 0.7853981633974483),
        (3.7, 2.0, 1.1071487177940904),
        (3.85, 2.6, 1.3734007669450158),
        (3.9, 3.2, 1.4711276743037347),
        (3.85, 3.8, 1.6814535479687923),
        (3.7, 4.3, 1.8157749899217608),
        (3.4, 4.8, 2.1112158270654806),
        (3.0, 5.1, 2.356194490192345),
        (2.5, 5.3, 2.677945044588987),
        (2.0, 5.4, 2.9441970937399127),
        (1.5, 5.4, 3.141592653589793),
        (1.0, 5.3, -3.078760800517997),
        (0.5, 5.1, -2.9441970937399127),
        (0.1, 4.8, -2.677945044588987),
        (-0.2, 4.4, -2.356194490192345),
        (-0.4, 4.0, -2.1112158270654806),
        (-0.5, 3.5, -1.8925468811915387),
        (-0.5, 3.0, -1.5707963267948966),
        (-0.4, 2.5, -1.3734007669450158),
        (-0.2, 2.0, -1.1071487177940904),
        (0.1, 1.6, -0.7853981633974483),
        (0.5, 1.3, -0.5880026035475675),
        (1.0, 1.1, -0.3805063771123649),
        (1.5, 1.0, -0.19739555984988078),
        (2.0, 1.0, 0.0),
        (2.5, 1.0, 0.0)
    ]

    traj = generate_trajectory_from_waypoints(waypoints, 1000+T) # trajectory of waypoints
    traj_x_y = traj[:, :2]
    sample_trajectories = np.zeros((Tx, K, 3, T))
    sample_trajectories_one = np.zeros((K, 3, T)) # k sets of (x1, x2, ..., xn), (y1, y2, ..., yn), (w1, w2, ..., wn)
    last_u = np.zeros(2) # the control input from the previous iteration
    best_traj_idx = 0
    for t in range(Tx - 1):
        u_nom, X_calc, traj_weight_single = mppi(x, last_u, traj_x_y, best_traj_idx) # Calculate the optimal control input
        # U[t] = safety_filter(u_nom, x)
        U[t] = u_nom
        x = unicyle_dynamics(x, U[t]) # Calculate what happens when you apply that input
        best_traj_idx = closest_point_on_path(traj_x_y, x[:2], best_traj_idx)[2]
        print(best_traj_idx)
        X[t + 1, :] = x # Store the new state
        time.append(t)
        x_pos.append(X[t+1, 0]) # Save the x position at this timestep
        y_pos.append(X[t+1, 1]) # Save the y position at this timestep

        last_u = U[t] # Save the control input 
        for k in range(K):
            for t_ in range (T): # Reshaping trajectory weight list for use in animation
                sample_trajectories_one[k, 0, t_] = X_calc[k, t_, 0] #should be 0
                sample_trajectories_one[k, 1, t_] = X_calc[k, t_, 1] #should be 1
        sample_trajectories[t] = sample_trajectories_one # Save the sampled trajectories
        all_weights[t] = traj_weight_single # List of the weights, populated in mppi function

    

    animate(x_pos, y_pos, traj[:, 0], traj[:, 1], sample_trajectories, all_weights)


if __name__ == "__main__":
    main()

