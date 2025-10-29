import cvxpy as cp
import numpy as np
from numba import int32, float32
from numba.experimental import jitclass
import time
from parameters import Parameters


class SafetyFilter():
    def __init__(self, params, alpha, q, dt):
        self.num_obstacles = len(params.obstacles)
        self.u_nom = cp.Parameter(2) # MPPI Output (what's changing)
        self.last_u = cp.Parameter(2)
        self.x = cp.Parameter(5)
        self.lgh1 = [cp.Parameter() for _ in range(self.num_obstacles)]
        self.lgh2 = [cp.Parameter() for _ in range(self.num_obstacles)]
        self.lfh = [cp.Parameter() for _ in range(self.num_obstacles)]
        self.h = [cp.Parameter() for _ in range(self.num_obstacles)]
        self.dh_dt = [cp.Parameter() for _ in range(self.num_obstacles)]
        self.u = cp.Variable(2) # Control output (decision variable)
        self.q = q
        self.cost = cp.quad_form(self.u - self.u_nom, q)
        self.filter_outputs = np.zeros((3,2), dtype=np.float32)
        self.alpha = alpha
        self.last_u_var = np.zeros(2)
        self.dt = dt
        self.last_obstacle_pos = np.copy(params.last_obstacle_pos)

        self.constraints = [
            self.u[0] <= params.max_v, # Box constraints
            self.u[0] >= -params.max_v,
            self.u[1] <= params.max_w,
            self.u[1] >= -params.max_w,
            self.u[0] - self.last_u[0] <= params.max_v_dot * dt, # Slew-rate limiting
            self.u[0] - self.last_u[0] >= -params.max_v_dot * dt,
            self.u[1] - self.last_u[1] <= params.max_w_dot * dt,
            self.u[1] - self.last_u[1] >= -params.max_w_dot * dt,
        ]

        for i in range(self.num_obstacles):
            Lg_h = cp.hstack([
                self.lgh1[i], 
                self.lgh2[i]
            ])
            self.constraints.append(Lg_h @ self.u + self.dh_dt[i] + alpha * self.h[i] >= 0)

        self.objective = cp.Minimize(self.cost)
        self.prob = cp.Problem(self.objective, self.constraints)

    def filter(self, u_in, x, params, last_u):
        self.u_nom.value = u_in
        # print(u_in)
        self.x.value = x
        self.last_u.value = last_u

        print(f"u_nom: {self.u_nom.value}")
        print(f"x: {self.x.value}")
        print(f"last_U: {self.last_u.value}")

        # Loop through all obstacles
        for i in range(self.num_obstacles):
            c = params.obstacles[i][0:2]   # obstacle center (x, y)
            r = params.obstacles[i][2]     # obstacle radius

            dx = x[0] - c[0] + params.l * np.cos(x[2])
            dy = x[1] - c[1] + params.l * np.sin(x[2])
            print(f"i={i}")
            print(f"c[{i}] = {c}]")
            print(f"r[{i}] = {r}")
            print(f"dx[{i}] = {dx}")
            print(f"dy[{i}] = {dy}")
            print(f"last_obs[{i}]={self.last_obstacle_pos[i]}")

            vx_obs = (c[0] - self.last_obstacle_pos[i][0]) / self.dt
            print(f"vx_obs:{vx_obs}")
            vy_obs = (c[1] - self.last_obstacle_pos[i][1]) / self.dt
            print(f"vy_obs:{vy_obs}")
            self.last_obstacle_pos[i] = np.array([c[0], c[1]])

            # Barrier function
            self.h[i].value = (dx)**2 + (dy)**2 - (r+0.1)**2
            print(f"h[{i}]: {self.h[i].value}")
            # Lie derivative term
            self.lfh[i].value = 2*x[3]*(dx*np.cos(x[2])+dy*np.sin(x[2]))
            # print(f"lfh[{i}]: {self.lfh[i].value}")
            self.lgh1[i].value = 2*dx*np.cos(x[2]) + 2*dy*np.sin(x[2])
            print(f"lgh1[{i}]: {self.lgh1[i].value}")
            self.lgh2[i].value = -2*dx*params.l*np.sin(x[2]) + 2*dy*params.l*np.cos(x[2])
            print(f"lgh2[{i}]: {self.lgh2[i].value}")
            self.dh_dt[i].value = -2*(dx)*vx_obs - 2*(dy)*vy_obs # Obstacle time_varying
            print(f"dhdt[{i}]: {self.dh_dt[i].value}")


        if np.isnan(u_in[0]) or np.isnan(u_in[1]):
            print("NAN")
            print("x", x)
            print("Ob 1 pos:", params.obstacles[0, :])
            # print("Ob 2 pos:", params.obstacles[1, :])

        try:
            self.prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)
            if self.prob.status not in ["optimal", "optimal_inaccurate"]:
                raise cp.error.SolverError("Infeasible or failed solve")
            u_out = self.u.value
        except cp.error.SolverError:
        # Fallback strategy
            u_out = np.array([0, 0])
        # Solve
        print(f"u_out: {u_out}")
        print(f"status:{self.prob.status}")
        print("**************************")
        return u_out