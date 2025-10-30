import numpy as np
from numba import njit
from utils import *
from trajectory import *
from mppi import *
from parameters import *
from safety_filter import SafetyFilter
import matplotlib.pyplot as plt

params = Parameters(
    dt = 0.025, # time step for MPPI
    safety_dt = 0.001, # time step for safety
    K = 1000,   # number of samples
    T = 18, # time steps (HORIZON)
    sigma = 2,
    lambda_ = 2,
    l = 0.3,
    r = 0.1, 
    max_v = 6.0, # max x velocity (m/s)
    max_w = 45.0, # max angular velocity (radians/s)
    max_v_dot = 8.0, # max linear acceleration (m/s^2)
    max_w_dot = 45.0, # max angular acceleration (radians/s^2) (8.0)
    obstacles = np.array([(6.0, 0.0, 0.2), (4.0, 0.0, 0.4)]),
    last_obstacle_pos = np.array([[6.0, 0.0], [4.0, 0.0]]),
    first_filter = True
)

def main():
    sf = SafetyFilter(params, 2.0, np.diag([25, 1]), params.safety_dt)
    