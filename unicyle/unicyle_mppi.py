import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.interpolate import interp1d
import random
matplotlib.use("TkAgg")


# system parameters
dt = 0.01 # time step
K = 500   # number of samples
T = 15 # time steps (HORIZON)
sigma = 1.0
lambda_ = 1

# Simulation Parameters
init_x = 0.0
init_y = 0.0
init_theta = 0
init_x_dot = 0.0
init_y_dot = 0.0
init_theta_dot = 0.0

target_x = -0.3
target_y = 0.0
target_theta = 0.5
target_x_dot = 0.0
target_y_dot = 0.0
target_theta_dot = 0.0


# Unicyle dynamics
def unicyle_dynamics(x, u):    
    # Next states that depend on time differential
    td_states = np.array([u[0], u[1]]) # should be cos, sin
    
     # Next states that don't depend on time differentia
    # ntd_states = np.array([x[0], x[1], x[2], u[0]*cos_theta, u[0]*sin_theta, u[1]])
    # x_star = td_states*dt + ntd_states
    x_star = x + td_states*dt

    return x_star

# Cost function
def cost_function(x, u, target):
    Q = np.diag([10, 10])  # State costs
    R = np.diag([0.0,0.0])  # Input costs

    x_des= np.array([target[0], target[1]])
    state_diff = x_des - x
    state_cost = np.dot(state_diff.T, np.dot(Q,state_diff))

    cost = state_cost + np.dot(u.T, np.dot(R, u))
    return cost

def terminal_cost(x, target):
    Q = np.diag([50, 50]);
    x_des= np.array([target[0], target[1]])
    state_diff = x_des - x
    terminal_cost = np.dot(state_diff.T, np.dot(Q,state_diff))
    return terminal_cost

X_calc = np.zeros((K, T + 1, 2))
traj_weight_single = np.zeros(K)
# MPPI control
def mppi(x, target, prev_U):
    #U = np.random.randn(K, T, 2) * sigma  # Random initial control inputs

    U = np.stack([
        np.random.normal(loc=1, scale=3, size=(K, T)), #vx
        np.random.normal(loc=0, scale=3, size=(K, T)), #vy
    ], axis=-1)

    for k in range(K):
        X_calc[k, 0, :] = x  # Initialize all trajectories with the current state

    costs = np.zeros(K)
    
    for k in range(K):
        for t in range(T):
            X_calc[k, t + 1, :] = unicyle_dynamics(X_calc[k, t, :], U[k, t])
            current_target = np.array([target[0][t],target[1][t]])
            costs[k] += cost_function(X_calc[k, t + 1, :], U[k, t], current_target)
        final_target = np.array([target[0][T-1],target[1][T-1]])
        terminal_cost_val = terminal_cost(X_calc[k, T, :], final_target) #Terminal cost of final state
        costs[k] += terminal_cost_val

    # Calculate weights for each trajectory
    weights = np.exp(-1/lambda_ * (costs))
    weights /= np.sum(weights)
    global traj_weight_single
    traj_weight_single[:] = weights

    # Compute the weighted sum of control inputs
    u_star = np.sum(weights[:, None, None] * U, axis=0)

    
    return u_star[0]

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

    return x_vals, y_vals


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



# Main function
def main():
    Tx = 1000
    x = np.array([0,0])  # Initial state [x, theta, x_dot, theta_dot]
    X = np.zeros((Tx, 2))
    U = np.zeros((Tx, 2))
    all_weights = np.zeros((Tx+1, K))
    
    time = []
    x_pos = []
    y_pos = []
    theta = []
    x_vel = []
    y_vel = []
    omega = []
    waypoints = [
        (0, 0, 0),
        (0.1, 0.2, np.pi / 8),
        (0.3, 0.5, np.pi / 6),
        (0.6, 0.7, np.pi / 4),
        (1.0, 0.8, np.pi / 3),
        (1.3, 0.6, np.pi / 2),
        (1.5, 0.3, 3*np.pi / 4),
        (1.66, 0, np.pi)
    ]


    traj = generate_trajectory_from_waypoints(waypoints, 1000+T)
    sample_trajectories = np.zeros((Tx, K, 2, T))
    sample_trajectories_one = np.zeros((K, 2, T))

    
    last_u = np.zeros(2)
    for t in range(Tx - 1):
        targets = np.array([
            traj[0][t:t+T], traj[1][t:t+T]
        ])
        U[t] = mppi(x, targets, last_u)
        x = unicyle_dynamics(x, U[t])
        X[t + 1, :] = x
        time.append(t)
        x_pos.append(X[t + 1, 0])
        y_pos.append(X[t+1, 1])

        last_u = U[t]
        for k in range(K):
            for t_ in range (T):
                sample_trajectories_one[k, 0, t_] = X_calc[k, t_, 0] #should be 0
                sample_trajectories_one[k, 1, t_] = X_calc[k, t_, 1] #should be 1
        sample_trajectories[t] = sample_trajectories_one
        all_weights[t] = traj_weight_single

    

    animate(x_pos, y_pos, traj[0], traj[1], sample_trajectories, all_weights)


if __name__ == "__main__":
    main()

