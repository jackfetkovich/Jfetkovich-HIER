import numpy as np
import matplotlib.pyplot as plt
import random

# system parameters
g = 9.81  # gravitational acceleration
l = 1.0   # length of the pole
m = 0.1   # mass of the pole
M = 1.0   # mass of the cart
dt = 0.01 # time step
K = 100   # number of samples
T = 400   # time steps
sigma = 1.0
lambda_ = 1.0

# Simulation Parameters
init_x = 0.0
init_theta = 0.0
init_x_dot = 0.0
init_theta_dot = 0.0

target_x = 0.3
target_x_dot = 0.0
target_theta = 0.5
target_theta_dot = 0.0

target_states = np.array([target_x, target_x_dot, target_theta, target_theta_dot])

# Generate a random number in the range [0, 1]
def gen_rand():
    return random.uniform(0.0, 1.0)

# Cart-pole dynamics
def cart_pole_dynamics(x, u):
    theta = x[1]
    theta_dot = x[3]
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    a = (1.0 / (M + m - m * cos_theta * cos_theta)) * (u + m * sin_theta * (l * theta_dot * theta_dot + g * cos_theta))
    theta_ddot = (1.0 / (l * (M + m - m * cos_theta * cos_theta))) * (-u * cos_theta - m * l * theta_dot * theta_dot * sin_theta * cos_theta - (M + m) * g * sin_theta)
    
    x_dot = np.array([x[2], x[3], a, theta_ddot])
    x_next = x + x_dot * dt
    return x_next

# Cost function
def cost_function(x, u):
    Q = np.diag([1.0, 1.0, 1.0, 10.0])  # Diagonal matrix Q
    R = 0.1  # Scalar R
    
    cost = np.dot(x.T, np.dot(Q, x)) + u * R * u
    return cost

def terminal_cost(x):
    Q = np.diag([100.0, 10000, 0, 0]);
    state_diff = np.abs(target_states - x)
    terminal_cost = np.dot(state_diff.T, np.dot(Q,state_diff))
    return terminal_cost

# MPPI control
def mppi(x):
    dt = 0.01
    T = 10
    K = 100

    U = np.random.randn(K, T) * sigma  # Random initial control inputs
    X = np.zeros((K, T + 1, 4))  # Array to store states
    for k in range(K):
        X[k, 0, :] = x  # Initialize all trajectories with the current state

    costs = np.zeros(K)
    
    for k in range(K):
        for t in range(T):
            X[k, t + 1, :] = cart_pole_dynamics(X[k, t, :], U[k, t])
            costs[k] += cost_function(X[k, t + 1, :], U[k, t])
        terminal_cost_val = terminal_cost(X[k, T, :]) #Terminal cost of final state
        costs[k] += terminal_cost_val

    # Calculate weights for each trajectory
    weights = np.exp(-lambda_ * (costs - np.min(costs)))
    weights /= np.sum(weights)

    # Compute the weighted sum of control inputs
    u_star = np.sum(weights[:, None] * U, axis=0)
    
    return u_star[0]

# Plotting function
def plot(time, cart_pos, pole_angle, cart_velo, pole_velo):
    plt.title("Cart Pole States")
    plt.plot(time, cart_pos, label="cart_pos")
    plt.plot(time, pole_angle, label="pole_angle")
    plt.plot(time, cart_velo, label="cart_velo")
    plt.plot(time, pole_velo, label="pole_velo")
    plt.xlabel("time")
    plt.ylabel("Y")
    plt.legend()
    plt.show()

# Main function
def main():
    Tx = 400
    x = np.array([init_x, init_theta, init_x_dot, init_theta_dot])  # Initial state [x, theta, x_dot, theta_dot]
    X = np.zeros((T, 4))
    U = np.zeros(T)
    
    time = []
    cart_pos = []
    pole_angle = []
    cart_velo = []
    pole_velo = []

    for t in range(Tx - 1):
        U[t] = mppi(x)
        #print(x)
        x = cart_pole_dynamics(x, U[t])
        X[t + 1, :] = x
        time.append(t)
        cart_pos.append(X[t + 1, 0])
        pole_angle.append(X[t + 1, 1])
        cart_velo.append(X[t + 1, 2])
        pole_velo.append(X[t + 1, 3])

    plot(time, cart_pos, pole_angle, cart_velo, pole_velo)

if __name__ == "__main__":
    main()

