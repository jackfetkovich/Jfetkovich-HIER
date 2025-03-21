import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
matplotlib.use("TkAgg")


# system parameters
dt = 0.01 # time step
K = 50   # number of samples
T = 10   # time steps (HORIZON)
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

target_states = np.array([target_x, target_y, target_theta, target_x_dot, target_y_dot, target_theta_dot])

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
    ntd_states = np.array([x[0], x[1], x[2], u[0]*cos_theta, u[0]*sin_theta, u[1]])
    x_star = td_states*dt + ntd_states

    return x_star

# Cost function
def cost_function(x, u):
    Q = np.diag([1.0, 0.0, 0.0, 0.0, 0, 0])  # State costs
    R = np.diag([0,0])  # Input costs
    
    cost = np.dot(x.T, np.dot(Q, x)) + np.dot(u.T, np.dot(R, u))
    return cost

def terminal_cost(x):
    Q = np.diag([100.0, 0, 1000, 0, 0, 0]);
    state_diff = np.abs(target_states - x)
    terminal_cost = np.dot(state_diff.T, np.dot(Q,state_diff))
    return terminal_cost

# MPPI control
def mppi(x):
    #U = np.random.randn(K, T, 2) * sigma  # Random initial control inputs

    U = np.stack([
        np.random.normal(loc=0, scale=1, size=(K, T)),
        np.random.normal(loc=0, scale=np.pi, size=(K, T))
    ], axis=-1)

    X = np.zeros((K, T + 1, 6))  # Array to store states
    for k in range(K):
        X[k, 0, :] = x  # Initialize all trajectories with the current state

    costs = np.zeros(K)
    
    for k in range(K):
        for t in range(T):
            X[k, t + 1, :] = unicyle_dynamics(X[k, t, :], U[k, t])
            costs[k] += cost_function(X[k, t + 1, :], U[k, t])
        #terminal_cost_val = terminal_cost(X[k, T, :]) #Terminal cost of final state
        #costs[k] += terminal_cost_val

    # Calculate weights for each trajectory
    weights = np.exp(-1/lambda_ * (costs)) #TODO: updating this cost function
    weights /= np.sum(weights)

    # Compute the weighted sum of control inputs
    u_star = np.sum(weights[:, None, None] * U, axis=0)
    
    return u_star[0]

# Plotting function
def plot(time, x_pos, y_pos, theta, x_vel, y_vel, omega):
    plt.title("Unicycle States")
    plt.plot(time, x_pos, label="x")
    plt.plot(time, y_pos, label="y")
    plt.plot(time, theta, label="theta")
    #plt.plot(time, x_vel, label="x velocity")
   # plt.plot(time, y_vel, label="y velocity")
    #plt.plot(time, omega, label="omega")
    plt.xlabel("time")
    plt.ylabel("Y")
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

    for t in range(Tx - 1):
        U[t] = mppi(x)
        print(U[t])
        x = unicyle_dynamics(x, U[t])
        X[t + 1, :] = x
        time.append(t)
        x_pos.append(X[t + 1, 0])
        y_pos.append(X[t+1, 1])
        theta.append(X[t+1, 2])
        x_vel.append(X[t + 1, 3])
        y_vel.append(X[t + 1, 4])
        omega.append(X[t + 1, 5])

    plot(time, x_pos, y_pos, theta, x_vel, y_vel, omega)

if __name__ == "__main__":
    main()

