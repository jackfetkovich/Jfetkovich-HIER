import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.interpolate import interp1d
import random
matplotlib.use("TkAgg")


# system parameters
dt = 0.01 # time step
K = 100   # number of samples
T = 10 # time steps (HORIZON)
sigma = 1.0
lambda_ = 1.0

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

# target_states = np.array([target_x, target_y, target_theta, target_x_dot, target_y_dot, target_theta_dot])

# Generate a random number in the range [0, 1]
def gen_rand():
    return random.uniform(0.0, 10.0)

# Unicyle dynamics
def unicyle_dynamics(x, u):
    theta = x[2]
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    
    # Next states that depend on time differential
    td_states = np.array([u[0]*cos_theta, u[0]*sin_theta, u[1], 0, 0, 0]) 
    
     # Next states that don't depend on time differentia
    # ntd_states = np.array([x[0], x[1], x[2], u[0]*cos_theta, u[0]*sin_theta, u[1]])
    # x_star = td_states*dt + ntd_states
    x_star = x + td_states*dt

    return x_star

# Cost function
def cost_function(x, u, target):
    Q = np.diag([1.0, 1.0, 0.0, 0, 0, 0])  # State costs
    R = np.diag([0,0])  # Input costs

    x_des= np.array([target[0], target[1], 0, 0, 0, 0])
    state_diff = np.abs(x_des - x)
    state_cost = np.dot(state_diff.T, np.dot(Q,state_diff))

    cost = state_cost + np.dot(u.T, np.dot(R, u))
    return cost

def terminal_cost(x, target):
    Q = np.diag([100.0, 100.0, 0, 0, 0, 0]);
    x_des= np.array([target[0], target[1], 0, 0, 0, 0])
    state_diff = np.abs(x_des - x)
    terminal_cost = np.dot(state_diff.T, np.dot(Q,state_diff))
    return terminal_cost

# MPPI control
def mppi(x, target):
    #U = np.random.randn(K, T, 2) * sigma  # Random initial control inputs

    U = np.stack([
        np.random.normal(loc=0, scale=10, size=(K, T)),
        np.random.normal(loc=0, scale=5*np.pi, size=(K, T))
    ], axis=-1)

    X = np.zeros((K, T + 1, 6))  # Array to store states
    for k in range(K):
        X[k, 0, :] = x  # Initialize all trajectories with the current state

    costs = np.zeros(K)
    
    for k in range(K):
        for t in range(T):
            X[k, t + 1, :] = unicyle_dynamics(X[k, t, :], U[k, t])
            current_target = np.array([target[0][t],target[1][t], target[2][t], target[3][t], target[4][t], target[5][t]])
            costs[k] += cost_function(X[k, t + 1, :], U[k, t], current_target)
        final_target = np.array([target[0][T-1],target[1][T-1], target[2][T-1], target[3][T-1], target[4][T-1], target[5][T-1]])
        terminal_cost_val = terminal_cost(X[k, T, :], final_target) #Terminal cost of final state
        costs[k] += terminal_cost_val

    # Calculate weights for each trajectory
    weights = np.exp(-1/lambda_ * (costs))
    weights /= np.sum(weights)
    print(weights)
    print(sum(weights))

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

    return x_vals, y_vals, theta_vals

# Plotting function
def plot(time, x_pos, y_pos, theta, x_vel, y_vel, omega):
    plt.title("Unicycle States")
    #plt.plot(time, x_pos, label="x")
    #plt.plot(time, y_pos, label="y")
    #plt.plot(time, theta, label="theta")
    plt.plot(time, x_vel, label="x velocity")
    plt.plot(time, y_vel, label="y velocity")
    plt.plot(time, omega, label="omega")
    plt.xlabel("time")
    plt.ylabel("Y")
    plt.legend()
    plt.show()

def animate(x_vals, y_vals, theta_vals, x_traj=None, y_traj=None):
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

    # Initialize plot elements
    line, = ax.plot([], [], 'r-', linewidth=2)  # History line
    point, = ax.plot([], [], 'bo', markersize=8)  # Current position
    ghost,  = ax.plot([], [], 'gx', markersize=6)  # Desired position
    arrow = ax.quiver([], [], [], [], angles='xy', scale_units='xy', scale=30, color='y')  # Orientation arrow

    # Update function
    def update(frame):
        x, y, theta, x_des, y_des = x_vals[frame], y_vals[frame], theta_vals[frame], x_traj[frame], y_traj[frame]

        # Update history path
        line.set_data(x_vals[:frame+1], y_vals[:frame+1])
        
        # Update point position
        point.set_data([x], [y])
        ghost.set_data([x_des], [y_des])
        
        # Update arrow orientation
        arrow.set_offsets([[x, y]])
        arrow.set_UVC([np.cos(theta)], [np.sin(theta)])

        return line, point, arrow, ghost

    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=len(x_vals), interval=50, blit=True)
    
    plt.legend()
    plt.show()

# Main function
def main():
    Tx = 1000
    x = np.zeros(6)  # Initial state [x, theta, x_dot, theta_dot]
    X = np.zeros((Tx, 6))
    U = np.zeros((Tx, 2))
    
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


    for t in range(Tx - 1):
        # targets = [traj[0][t:t+T], traj[1][t:t+T], traj[2][t:t+T], 0, 0, 0]
        targets = np.array([
            traj[0][t:t+T], traj[1][t:t+T], traj[2][t:t+T], np.zeros(T), np.zeros(T), np.zeros(T)
        ])
        U[t] = mppi(x, targets)
        x = unicyle_dynamics(x, U[t])
        X[t + 1, :] = x
        time.append(t)
        x_pos.append(X[t + 1, 0])
        y_pos.append(X[t+1, 1])
        theta.append(X[t+1, 2])
        x_vel.append(X[t + 1, 3])
        y_vel.append(X[t + 1, 4])
        omega.append(X[t + 1, 5])

    #plot(time, x_pos, y_pos, theta, x_vel, y_vel, omega)
    animate(x_pos, y_pos, theta, traj[0], traj[1])

if __name__ == "__main__":
    main()

