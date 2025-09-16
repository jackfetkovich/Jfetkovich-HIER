import numpy as np
from numba import int64, float64    # import the types
from numba.experimental import jitclass

spec = [
    ('dt', float64),
    ('safety_dt', float64),
    ('K', int64), 
    ('T', int64), 
    ('sigma', int64), 
    ('lambda_', int64),   
    ('l', float64),
    ('r', float64),            
    ('max_v', float64),  
    ('max_w', float64),  
    ('max_v_dot', float64), 
    ('max_w_dot', float64) ,
    ('obstacles', float64[:,:])
]

@jitclass(spec)
class Parameters(object):
    def __init__(self, dt, safety_dt, K, T, sigma, lambda_, l, r, max_v, max_w, max_v_dot, max_w_dot, obstacles):
        self.dt = dt
        self.safety_dt = safety_dt
        self.K = K
        self.T = T
        self.sigma = sigma
        self.lambda_ = lambda_
        self.l = l
        self.r = r
        self.max_v = max_v
        self.max_w = max_w
        self.max_v_dot = max_v_dot
        self.max_w_dot = max_w_dot
        self.obstacles = obstacles
