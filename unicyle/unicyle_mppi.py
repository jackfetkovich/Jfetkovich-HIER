import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.interpolate import interp1d
import random
matplotlib.use("TkAgg")


# system parameters
dt = 0.01 # time step
K = 600   # number of samples
T = 10 # time steps (HORIZON)
sigma = 1.0
lambda_ = 6

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

obstacle = np.array([0.3, 0.5])


# Unicyle dynamics
def unicyle_dynamics(x, u):    
    # Next states that depend on time differential
    td_states = np.array([u[0]*np.cos(x[2]), u[0]*np.sin(x[2]), u[1]]) # 
    x_star = x + td_states*dt

    return x_star

# Cost function
def cost_function(x, u, target):
    Q = np.diag([5, 5,0.0005])  # State costs
    R = np.diag([0.001,0.001])  # Input costs
    O = 100

    x_des = np.array([target[0], target[1], target[2]])
    state_diff = x_des - x
    state_cost = np.dot(state_diff.T, np.dot(Q,state_diff))
    
    obstacle_cost = 0
    # if(abs(x[0] - obstacle[0]) < 0.1 and abs(x[1] - obstacle[1]) < 0.1):
    #     obstacle_cost = O
    

    cost = state_cost + np.dot(u.T, np.dot(R, u)) + obstacle_cost
    return cost

def terminal_cost(x, target):
    Q = np.diag([6, 6, 0.0005])
    x_des= np.array([target[0], target[1], target[2]])
    state_diff = x_des - x
    O = 100
    obstacle_cost = 0
    # if(abs(x[0] - obstacle[0]) < 0.1 and abs(x[1] - obstacle[1])< 0.1):
    #     obstacle_cost = O
    terminal_cost = np.dot(state_diff.T, np.dot(Q,state_diff)) + obstacle_cost
    return terminal_cost 

X_calc = np.zeros((K, T + 1, 3))
traj_weight_single = np.zeros(K)
# MPPI control
def mppi(x, target, prev_U):
    #U = np.random.randn(K, T, 2) * sigma  # Random initial control inputs

    U = np.stack([
        np.random.normal(loc=0, scale=10, size=(K, T)), #v
        np.random.normal(loc=0, scale=50, size=(K, T)), #omega
    ], axis=-1) # Generate random (normal) control inputs

    for k in range(K):
        X_calc[k, 0, :] = x  # Initialize all trajectories with the current state

    costs = np.zeros(K) # initialize all costs
    
    for k in range(K):
        for t in range(T-1):
            X_calc[k, t + 1, :] = unicyle_dynamics(X_calc[k, t, :], U[k, t])
            current_target = np.array([target[0][t],target[1][t], target[2][t]])
            costs[k] += cost_function(X_calc[k, t+1, :], U[k, t], current_target)
        final_target = np.array([target[0][T-1],target[1][T-1], target[2][T-1]])
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

    return x_vals, y_vals, theta_vals


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

    plt.plot(0.3, 0.5, 'yo') #obstacle

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
    x = np.array([0,0,0])  # Initial state [x, theta, x_dot, theta_dot] -- tracks current state
    X = np.zeros((Tx, 3)) # list of historical states
    U = np.zeros((Tx, 2)) # list of historical control inputs
    all_weights = np.zeros((Tx+1, K)) # Weights of every generated trajectory, organized by time step
    
    time = []
    x_pos = []
    y_pos = []
    theta = []
    x_vel = []
    y_vel = []
    omega = []
    # waypoints = [
    #     (0.0, 0.0, 1.1071487177940904),
    #     (0.1, 0.2, 0.982793723247329),
    #     (0.3, 0.5, 0.5880026035475675),
    #     (0.6, 0.7, 0.24497866312686414),
    #     (1.0, 0.8, -0.46364760900080615),
    #     (1.3, 0.6, -0.5880026035475675),
    #     (1.5, 0.3, -1.0636978224025597),
    #     (1.66, 0.0, -1.0899093659292262)
    # ]
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
    sample_trajectories = np.zeros((Tx, K, 3, T))
    sample_trajectories_one = np.zeros((K, 3, T)) # k sets of (x1, x2, ..., xn), (y1, y2, ..., yn), (w1, w2, ..., wn)

    
    last_u = np.zeros(2) # the control input from the previous iteration
    for t in range(Tx - 1):
        targets = np.array([ # Get the target state at this timestep
            traj[0][t+1:t+1+T], traj[1][t+1:t+1+T], traj[2][t+1:t+1+T]
        ])
        U[t] = mppi(x, targets, last_u) # Calculate the optimal control input
        x = unicyle_dynamics(x, U[t]) # Calculate what happens when you apply that input
        X[t + 1, :] = x # Store the new state
        time.append(t)
        x_pos.append(X[t + 1, 0]) # Save the x position at this timestep
        y_pos.append(X[t+1, 1]) # Save the y position at this timestep

        last_u = U[t] # Save the control input 
        for k in range(K):
            for t_ in range (T): # Reshaping trajectory weight list for use in animation
                sample_trajectories_one[k, 0, t_] = X_calc[k, t_, 0] #should be 0
                sample_trajectories_one[k, 1, t_] = X_calc[k, t_, 1] #should be 1
        sample_trajectories[t] = sample_trajectories_one # Save the sampled trajectories
        all_weights[t] = traj_weight_single # List of the weights, populated in mppi function

    

    animate(x_pos, y_pos, traj[0], traj[1], sample_trajectories, all_weights)


if __name__ == "__main__":
    main()

